#!/usr/bin/env python3
"""
Evaluate a STKLM0-trained BertPCa model on the Milan (OSR) dataset.

Uses Milan train/val/test CSVs (already in STKLM0 feature format, produced by
prepare_milan.py) and applies Milan-fitted min-max scaling.  The STKLM0-trained
model is then evaluated on the Milan test split.

Note: STKLM0 training used STKLM0-fitted scaling (not saved to disk); Milan-fitted
scaling is used here as the closest available approximation — both datasets share
the same feature set and similar clinical value ranges.

Prerequisites:
  python stklm0/scripts/prepare_milan.py --outcome csm   # produces milan_csm_*.csv

Run from repo root:
  python stklm0/scripts/eval_milan.py
  python stklm0/scripts/eval_milan.py --outcome bcr
  python stklm0/scripts/eval_milan.py --model stklm0/outputs/models/my_model.keras
"""

import csv
import json
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bertpca", "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bertpca"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "stklm0"))

from bertpca import calculate_time_dependent_c_index, load_and_preprocess_data
from bertpca.loss import weibull_loss
from config.load_config import load_yaml_config

_CONFIG_PATH = os.path.join(_REPO_ROOT, "stklm0", "config", "config_stklm0.yaml")
_DATA_DIR    = os.path.join(_REPO_ROOT, "stklm0", "data")
_MODEL_DIR   = os.path.join(_REPO_ROOT, "stklm0", "outputs", "models")
_RESULTS_DIR = os.path.join(_REPO_ROOT, "stklm0", "outputs", "results")


def _newest_stklm0_model():
    import glob
    candidates = sorted(
        glob.glob(os.path.join(_MODEL_DIR, "app_trained_stklm0_csm*.keras")),
        key=os.path.getmtime,
    )
    return candidates[-1] if candidates else None


def run(outcome: str = "csm", model_path: str = None):
    config = load_yaml_config(_CONFIG_PATH)

    # ---- Resolve model path -------------------------------------------------------
    if model_path is None:
        model_path = _newest_stklm0_model()
        if model_path is None:
            raise FileNotFoundError(
                f"No app_trained_stklm0_csm*.keras found in {_MODEL_DIR}.\n"
                "Train one first via the Streamlit app or train_eval_stklm0.py."
            )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"Model:   {os.path.basename(model_path)}")

    # ---- Check Milan data ---------------------------------------------------------
    train_path = os.path.join(_DATA_DIR, f"milan_{outcome}_train.csv")
    val_path   = os.path.join(_DATA_DIR, f"milan_{outcome}_val.csv")
    test_path  = os.path.join(_DATA_DIR, f"milan_{outcome}_test.csv")
    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing: {p}\n"
                f"Run: python stklm0/scripts/prepare_milan.py --outcome {outcome}"
            )

    # ---- Detect features from train CSV ------------------------------------------
    sample = pd.read_csv(train_path, nrows=2, index_col="id")
    dynamic_features = ["times", "psa"]
    static_features  = [c for c in sample.columns
                        if c not in dynamic_features + ["tte", "label"]]
    n_features = len(static_features) + len(dynamic_features)
    print(f"Features: {n_features} ({len(static_features)} static + {len(dynamic_features)} dynamic)")
    print(f"Static:   {static_features}")

    # ---- Load and preprocess Milan data (scaling fit on Milan train) --------------
    print(f"\nLoading Milan {outcome.upper()} data ...")
    train_ds, val_ds, test_ds, y_train_struct, y_val_struct, y_test_struct = (
        load_and_preprocess_data(
            train_path, val_path, test_path,
            static_features, dynamic_features,
            config.SEQ_LENGTH, config.BATCH_SIZE,
            config.T_MAX, augment=False, scale_features=config.SCALE_FEATURES,
        )
    )

    def _n_patients(ds):
        return len(np.unique([r for r in range(len(ds["features"]))]))

    # Count from CSV directly (more reliable)
    def _csv_stats(path):
        df = pd.read_csv(path, index_col="id")
        n_pts = df.index.nunique()
        n_ev  = int(df.groupby(level=0)["label"].first().sum())
        return n_pts, n_ev

    n_tr, ev_tr = _csv_stats(train_path)
    n_te, ev_te = _csv_stats(test_path)
    print(f"  Train: {n_tr} patients, {ev_tr} events")
    print(f"  Test:  {n_te} patients, {ev_te} events")

    # ---- Load model ---------------------------------------------------------------
    print(f"\nLoading model ...")
    model = tf.keras.models.load_model(
        model_path, custom_objects={"weibull_loss": weibull_loss}
    )

    # ---- Evaluate -----------------------------------------------------------------
    p_times = np.array(config.EVALUATION_CONFIG["p_times"])
    e_times = np.array(config.EVALUATION_CONFIG["e_times"])

    print(f"Computing C-index (STKLM0 model → Milan {outcome.upper()}) ...")
    c_matrix = calculate_time_dependent_c_index(
        np.array(test_ds["features"]),
        y_train_struct,
        y_test_struct,
        model,
        p_times=p_times,
        e_times=e_times,
        t_max=config.EVALUATION_CONFIG["t_max"],
        return_mean=False,
    )

    mean_c = float(np.nanmean(np.where(c_matrix == -1.0, np.nan, c_matrix)))
    print(f"\nExternal validation C-index (STKLM0 → Milan {outcome.upper()}):")
    header = ["p_time"] + [f"e={int(e) // 365}y" for e in e_times]
    print("  " + "  ".join(f"{h:>10}" for h in header))
    for i, p in enumerate(p_times):
        vals = [
            f"{c_matrix[i, j]:.4f}" if c_matrix[i, j] != -1.0 else "    —   "
            for j in range(len(e_times))
        ]
        print(f"  {int(p) // 365:>4}y    " + "  ".join(f"{v:>10}" for v in vals))
    print(f"\nMean C-index: {mean_c:.4f}")

    # ---- Save results -------------------------------------------------------------
    out_dir = os.path.join(_RESULTS_DIR, f"stklm0_to_milan_{outcome}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "mean_c_index.txt"), "w") as f:
        f.write(f"{mean_c:.6f}\n")

    table_path = os.path.join(out_dir, "c_index_table.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p_time"] + [f"e_time_{int(e)}" for e in e_times])
        for i, p in enumerate(p_times):
            row = [int(p)]
            for j in range(len(e_times)):
                v = c_matrix[i, j]
                row.append(f"{v:.6f}" if v != -1.0 else "")
            writer.writerow(row)
    print(f"\nSaved: {table_path}")

    # Summary CSV alongside existing model_comparison files
    summary_path = os.path.join(_RESULTS_DIR, "stklm0_to_milan_summary.csv")
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows = []
    for i, p in enumerate(p_times):
        for j, e in enumerate(e_times):
            v = c_matrix[i, j]
            summary_rows.append({
                "timestamp": ts,
                "model": os.path.basename(model_path),
                "direction": f"stklm0_to_milan_{outcome}",
                "p_time": int(p),
                "e_time": int(e),
                "c_index": round(float(v), 6) if v != -1.0 else None,
            })
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Summary: {summary_path}")

    return c_matrix, mean_c


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate STKLM0-trained BertPCa on Milan (OSR) data"
    )
    parser.add_argument("--outcome", choices=["bcr", "csm"], default="csm",
                        help="Which Milan split to evaluate on (default: csm)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to STKLM0-trained .keras model (default: newest app_trained_stklm0_csm*.keras)")
    args = parser.parse_args()

    def abs_path(p):
        return os.path.join(_REPO_ROOT, p) if p and not os.path.isabs(p) else p

    run(outcome=args.outcome, model_path=abs_path(args.model))
    print("\nDone.")


if __name__ == "__main__":
    main()
