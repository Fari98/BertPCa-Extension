#!/usr/bin/env python3
"""
Evaluate a Milan-trained BertPCa model on the STKLM0 test set.

Loads the model and the Milan scaling parameters saved by train_milan.py,
applies Milan's feature scaling to the STKLM0 test data, and computes the
time-dependent C-index (external validation).

Prerequisites:
  python stklm0/scripts/prepare_stklm0.py --input data/stklm0.csv
  python stklm0/scripts/train_milan.py --outcome bcr   # or csm

Run from repo root:
  python stklm0/scripts/eval_stklm0.py --outcome bcr
  python stklm0/scripts/eval_stklm0.py --outcome csm
  python stklm0/scripts/eval_stklm0.py --outcome bcr \\
      --model stklm0/outputs/models/my_model.keras \\
      --scaling stklm0/outputs/models/milan_bcr_scaling.json
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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bertpca", "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bertpca"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "stklm0"))

from bertpca import calculate_time_dependent_c_index
from bertpca.data import preprocess_data
from bertpca.loss import weibull_loss
from config.load_config import load_yaml_config

_CONFIG_PATH = os.path.join(_REPO_ROOT, "stklm0", "config", "config_stklm0.yaml")
_DATA_DIR    = os.path.join(_REPO_ROOT, "stklm0", "data")
_MODEL_DIR   = os.path.join(_REPO_ROOT, "stklm0", "outputs", "models")


def _load_csv(path):
    df = pd.read_csv(path)
    df.set_index("id", inplace=True)
    return df.astype(float)


def _build_structured_labels(df_long):
    dt   = np.dtype([("Status", "?"), ("Survival_in_days", "<f8")])
    last = df_long.groupby(level=0).last()
    return np.array(list(zip(last["label"].astype(bool), last["tte"])), dtype=dt)


def _scale_df(df, features_to_scale, train_max, train_min, t_max):
    denom = (train_max - train_min).replace(0, 1)
    df = df.copy()
    present = [f for f in features_to_scale if f in df.columns]
    df[present] = (df[present] - train_min[present]) / denom[present]
    df[["tte", "times"]] = df[["tte", "times"]] / t_max
    return df


def run(outcome: str, model_path: str = None, scaling_path: str = None,
        y_train_path: str = None):

    config = load_yaml_config(_CONFIG_PATH)
    t_max  = config.T_MAX

    # Default paths (produced by train_milan.py)
    if model_path is None:
        model_path = os.path.join(_MODEL_DIR, f"best_model_milan_{outcome}.keras")
    if scaling_path is None:
        scaling_path = os.path.join(_MODEL_DIR, f"milan_{outcome}_scaling.json")
    if y_train_path is None:
        y_train_path = os.path.join(_MODEL_DIR, f"milan_{outcome}_y_train.npz")

    for p, label in [(model_path, "model"), (scaling_path, "scaling params"),
                     (y_train_path, "training labels")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {label}: {p}\n"
                f"Run: python stklm0/scripts/train_milan.py --outcome {outcome}"
            )

    stklm0_test_path  = os.path.join(_DATA_DIR, "stklm0_test.csv")
    stklm0_train_path = os.path.join(_DATA_DIR, "stklm0_train.csv")
    stklm0_val_path   = os.path.join(_DATA_DIR, "stklm0_val.csv")
    for p in [stklm0_test_path, stklm0_train_path, stklm0_val_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing: {p}\n"
                "Run: python stklm0/scripts/prepare_stklm0.py --input data/stklm0.csv"
            )

    # ---- Load scaling parameters ------------------------------------------------
    print(f"Loading scaling params from {scaling_path} ...")
    with open(scaling_path) as f:
        sp = json.load(f)

    available_static  = sp["available_static"]
    dynamic_features  = sp["dynamic_features"]
    features_to_scale = sp["features_to_scale"]
    train_max = pd.Series(sp["train_max"])
    train_min = pd.Series(sp["train_min"])
    print(f"  Features ({len(available_static)}): {available_static}")

    # ---- Load STKLM0 test set and apply Milan scaling ---------------------------
    print("\nLoading STKLM0 test set ...")
    stklm0_test_raw = _load_csv(stklm0_test_path)

    # Keep only columns the model was trained with (+ tte, label, times, psa)
    needed_cols = available_static + dynamic_features + ["tte", "label"]
    stklm0_test = stklm0_test_raw[[c for c in needed_cols
                                    if c in stklm0_test_raw.columns]].copy()
    stklm0_test = _scale_df(stklm0_test, features_to_scale, train_max, train_min, t_max)

    stklm0_test_ds, _ = preprocess_data(
        stklm0_test, available_static, dynamic_features,
        "label", config.SEQ_LENGTH, config.BATCH_SIZE,
    )
    stklm0_test_struct = _build_structured_labels(stklm0_test)
    print(f"  Test patients: {stklm0_test.index.nunique()}, "
          f"events: {int(stklm0_test.groupby(level=0)['label'].first().sum())}")

    # ---- Load IPCW base labels (STKLM0 train+val, same population as test) -----
    stklm0_trainval = pd.concat([_load_csv(stklm0_train_path),
                                  _load_csv(stklm0_val_path)])
    stklm0_base_struct = _build_structured_labels(stklm0_trainval)

    # ---- Load model -------------------------------------------------------------
    print(f"\nLoading model from {model_path} ...")
    model = tf.keras.models.load_model(
        model_path, custom_objects={"weibull_loss": weibull_loss}
    )

    # ---- Evaluate ---------------------------------------------------------------
    p_times = np.array(config.EVALUATION_CONFIG["p_times"])
    e_times = np.array(config.EVALUATION_CONFIG["e_times"])

    print(f"Computing C-index (Milan {outcome.upper()} → STKLM0) ...")
    c_matrix = calculate_time_dependent_c_index(
        np.array(stklm0_test_ds["features"]),
        stklm0_base_struct,
        stklm0_test_struct,
        model,
        p_times=p_times,
        e_times=e_times,
        t_max=t_max,
        return_mean=False,
    )

    print(f"\nExternal validation C-Index (Milan {outcome.upper()} → STKLM0):")
    print(c_matrix)
    mean_c = float(np.nanmean(c_matrix))
    print(f"Mean C-Index: {mean_c:.6f}")

    # ---- Save results -----------------------------------------------------------
    results_dir = os.path.join(_REPO_ROOT, "stklm0", "outputs", "results",
                                f"milan_{outcome}_to_stklm0")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "mean_c_index.txt"), "w") as f:
        f.write(f"{mean_c:.6f}\n")

    table_path = os.path.join(results_dir, "c_index_table.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p_time"] + [f"e_time_{int(e)}" for e in e_times])
        for i, p in enumerate(p_times):
            writer.writerow([int(p)] + [f"{c_matrix[i, j]:.6f}" for j in range(len(e_times))])
    print(f"C-index table saved to {table_path}")

    return c_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Milan-trained BertPCa on STKLM0 test set"
    )
    parser.add_argument("--outcome",  choices=["bcr", "csm"], required=True,
                        help="Milan training outcome to evaluate")
    parser.add_argument("--model",    type=str, default=None,
                        help="Path to saved .keras model (default: auto from outcome)")
    parser.add_argument("--scaling",  type=str, default=None,
                        help="Path to milan_*_scaling.json (default: auto from outcome)")
    parser.add_argument("--y-train",  type=str, default=None,
                        help="Path to milan_*_y_train.npz (default: auto from outcome)")
    args = parser.parse_args()

    def abs_path(p):
        return os.path.join(_REPO_ROOT, p) if p and not os.path.isabs(p) else p

    run(
        outcome=args.outcome,
        model_path=abs_path(args.model),
        scaling_path=abs_path(args.scaling),
        y_train_path=abs_path(args.y_train),
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
