#!/usr/bin/env python3
"""
Run all baselines (CAPRA-S, MSKCC, CoxPH, RSF, DDH) on EF and/or UC outcomes
and optionally compare against a pre-trained BertPCa model.

Usage (from repo root):
  python functional_outcomes/scripts/run_baselines.py --outcome ef
  python functional_outcomes/scripts/run_baselines.py --outcome uc
  python functional_outcomes/scripts/run_baselines.py --outcome all
  python functional_outcomes/scripts/run_baselines.py --outcome ef --bertpca-model functional_outcomes/outputs/models/best_model_ef.keras
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bertpca", "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bertpca"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "functional_outcomes"))

from bertpca.metrics import weighted_c_index
from src.baselines.capras import capras_score
from src.baselines.mskcc import mskcc_score
from src.baselines.coxph_rsf import train_coxph, train_rsf, evaluate_static_model
from src.baselines.ddh import train_ddh, evaluate_ddh

DATA_DIR = os.path.join(_REPO_ROOT, "functional_outcomes", "data")
OUT_DIR = os.path.join(_REPO_ROOT, "functional_outcomes", "outputs")

OUTCOME_CFG = {
    "ef": {
        "train": "ef_train.csv",
        "val":   "ef_val.csv",
        "test":  "ef_test.csv",
        "static_cols": [
            "nerve_sparing", "IIEF_EFdomain_pre", "age", "tpsa", "bmi",
            "pathgg_group", "ece_bin", "svi_bin", "psm", "lni_bin",
            "neo_adjHT", "pstage",
        ],
        "p_times": np.array([90.0, 120.0, 180.0]),   # IIEF>=26 events peak later
        "e_times": np.array([180.0, 270.0, 365.0]),
        "t_max": 365.0,
    },
    "uc": {
        "train": "uc_train.csv",
        "val":   "uc_val.csv",
        "test":  "uc_test.csv",
        "static_cols": [
            "nerve_sparing", "IPSS_pre", "age", "tpsa", "bmi",
            "pathgg_group", "ece_bin", "svi_bin", "psm",
            "prostate_vol", "operative_time",
        ],
        "p_times": np.array([7.0, 14.0, 30.0]),
        "e_times": np.array([30.0, 90.0, 180.0]),
        "t_max": 365.0,
    },
}


def load_splits(cfg: dict) -> tuple:
    """Load train/val/test CSVs, return DataFrames indexed by patient ID."""
    train = pd.read_csv(os.path.join(DATA_DIR, cfg["train"])).set_index("id")
    val   = pd.read_csv(os.path.join(DATA_DIR, cfg["val"])).set_index("id")
    test  = pd.read_csv(os.path.join(DATA_DIR, cfg["test"])).set_index("id")
    return train, val, test


def patient_level(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    return df.groupby(level=0)[cols + ["tte", "label"]].first()


def run_nomogram(name: str, score_fn, train_pt, test_pt, e_times, t_max) -> dict:
    """Evaluate a score function (CAPRA-S or MSKCC) with weighted C-index."""
    risk = score_fn(test_pt)
    # Higher score → worse prognosis → later/no recovery → negate
    risk = -np.array(risk, dtype=float)
    results = {}
    for e in e_times:
        valid = ~np.isnan(risk)
        if valid.sum() < 10:
            results[e] = np.nan
            continue
        results[e] = weighted_c_index(
            train_pt["tte"].values, train_pt["label"].values,
            risk[valid],
            test_pt["tte"].values[valid], test_pt["label"].values[valid],
            e,
        )
    return results


def run_bertpca(model_path: str, train_df, test_df, static_cols, p_times, e_times, t_max) -> np.ndarray:
    """Evaluate a saved BertPCa model using calculate_time_dependent_c_index.

    Applies the same min-max scaling (fit on train) and t_max normalisation that
    load_and_preprocess_data uses during training, so features and labels are in
    the same [0,1] space that evaluate.py expects.
    """
    import tensorflow as tf
    from bertpca import calculate_time_dependent_c_index
    from bertpca.loss import weibull_loss
    from bertpca.data import preprocess_data

    print(f"  Loading BertPCa model from {model_path} ...")
    model = tf.keras.models.load_model(model_path, custom_objects={"weibull_loss": weibull_loss})

    dynamic_features = ["times", "psa"]
    features_to_scale = [f for f in static_cols + dynamic_features if f != "times"]

    # Fit scaling on train, apply to both splits (mirrors load_and_preprocess_data)
    train_s = train_df.copy().astype(float)
    test_s  = test_df.copy().astype(float)
    train_max = train_s[features_to_scale].max()
    train_min = train_s[features_to_scale].min()
    denom = (train_max - train_min).replace(0, 1)
    for df in (train_s, test_s):
        df[features_to_scale] = (df[features_to_scale] - train_min) / denom
        df["times"] = df["times"] / t_max
        df["tte"]   = df["tte"]   / t_max

    # Build scaled structured label arrays (Survival_in_days = tte/t_max)
    dt = np.dtype([("Status", "?"), ("Survival_in_days", "<f8")])
    train_pt = train_s.groupby(level=0)[["tte", "label"]].first()
    test_pt  = test_s.groupby(level=0)[["tte", "label"]].first()
    y_train = np.array(list(zip(train_pt["label"].astype(bool), train_pt["tte"])), dtype=dt)
    y_test  = np.array(list(zip(test_pt["label"].astype(bool),  test_pt["tte"])),  dtype=dt)

    # Build feature tensors from scaled data
    test_ds, _ = preprocess_data(
        test_s, static_cols, dynamic_features, "label",
        seq_length=16, batch_size=len(test_pt),
    )
    features = np.array(test_ds["features"])

    c_matrix = calculate_time_dependent_c_index(
        features, y_train, y_test, model,
        p_times=p_times, e_times=e_times,
        t_max=t_max, return_mean=False,
    )
    return c_matrix


def format_table(results: dict, p_times: np.ndarray, e_times: np.ndarray) -> pd.DataFrame:
    """
    Format results dict into a DataFrame.

    results: {method_name: np.ndarray (len(p_times), len(e_times)) or dict {e_time: value}}
    """
    rows = []
    col_names = [f"p{int(p)}_e{int(e)}" for p in p_times for e in e_times] + ["mean"]
    for method, vals in results.items():
        if isinstance(vals, np.ndarray):
            flat = vals.flatten().tolist()
            mean = float(np.nanmean(vals))
        elif isinstance(vals, dict):
            # nomograms: only e_times, no p_times — replicate across p_times
            flat = []
            for p in p_times:
                for e in e_times:
                    flat.append(vals.get(e, np.nan))
            mean = float(np.nanmean(list(vals.values())))
        else:
            flat = [np.nan] * (len(p_times) * len(e_times))
            mean = np.nan
        rows.append([method] + flat + [mean])
    return pd.DataFrame(rows, columns=["method"] + col_names)


def run_outcome(outcome: str, bertpca_model_path: str = None):
    cfg = OUTCOME_CFG[outcome]
    static_cols = cfg["static_cols"]
    p_times = cfg["p_times"]
    e_times = cfg["e_times"]
    t_max = cfg["t_max"]

    print(f"\n{'='*60}")
    print(f"Outcome: {outcome.upper()}")
    print(f"{'='*60}")

    train_df, val_df, test_df = load_splits(cfg)
    train_val_df = pd.concat([train_df, val_df])

    train_pt = patient_level(train_df, static_cols)
    test_pt  = patient_level(test_df, static_cols)

    print(f"Train: {len(train_pt)} patients | Val: {len(patient_level(val_df, static_cols))} | Test: {len(test_pt)}")
    print(f"Test events: {int(test_pt['label'].sum())} / {len(test_pt)}")

    all_results = {}

    # --- CAPRA-S ---
    print("\n[1/5] CAPRA-S ...")
    capras_res = run_nomogram("CAPRA-S", capras_score, train_pt, test_pt, e_times, t_max)
    all_results["CAPRA-S"] = capras_res
    print(f"  C-indices: { {int(k): round(v, 4) for k, v in capras_res.items()} }")

    # --- MSKCC ---
    print("\n[2/5] MSKCC nomogram ...")
    mskcc_res = run_nomogram("MSKCC", mskcc_score, train_pt, test_pt, e_times, t_max)
    all_results["MSKCC"] = mskcc_res
    print(f"  C-indices: { {int(k): round(v, 4) for k, v in mskcc_res.items()} }")

    # --- CoxPH ---
    print("\n[3/5] CoxPH ...")
    try:
        cox_model = train_coxph(train_df, static_cols)
        cox_res = evaluate_static_model(cox_model, train_df, test_df, static_cols, e_times)
        all_results["CoxPH"] = cox_res
        print(f"  C-indices: { {int(k): round(v, 4) for k, v in cox_res.items()} }")
    except Exception as exc:
        print(f"  CoxPH failed: {exc}")
        all_results["CoxPH"] = {}

    # --- RSF ---
    print("\n[4/5] RSF ...")
    try:
        rsf_model = train_rsf(train_df, static_cols)
        rsf_res = evaluate_static_model(rsf_model, train_df, test_df, static_cols, e_times)
        all_results["RSF"] = rsf_res
        print(f"  C-indices: { {int(k): round(v, 4) for k, v in rsf_res.items()} }")
    except Exception as exc:
        print(f"  RSF failed: {exc}")
        all_results["RSF"] = {}

    # --- DDH ---
    print("\n[5/5] Dynamic-DeepHit ...")
    try:
        ddh_model = train_ddh(
            train_df, val_df, static_cols,
            seq_length=16, n_bins=36, t_max=t_max,
            epochs=100, batch_size=32, patience=10,
        )
        ddh_res = evaluate_ddh(
            ddh_model, train_val_df, test_df, static_cols,
            p_times, e_times,
        )
        all_results["DDH"] = ddh_res
        print(f"  Mean C-index: {float(np.nanmean(ddh_res)):.4f}")
    except Exception as exc:
        print(f"  DDH failed: {exc}")
        all_results["DDH"] = np.full((len(p_times), len(e_times)), np.nan)

    # --- BertPCa (optional) ---
    if bertpca_model_path and os.path.exists(bertpca_model_path):
        print(f"\n[+] BertPCa from {bertpca_model_path} ...")
        try:
            bp_res = run_bertpca(bertpca_model_path, train_df, test_df, static_cols, p_times, e_times, t_max)
            all_results["BertPCa"] = bp_res
            print(f"  Mean C-index: {float(np.nanmean(bp_res)):.4f}")
        except Exception as exc:
            print(f"  BertPCa eval failed: {exc}")

    # --- Summary table ---
    table = format_table(all_results, p_times, e_times)
    print(f"\n{'='*60}")
    print(f"Results summary ({outcome.upper()}):")
    print(table.to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"baseline_results_{outcome}.csv")
    table.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    return table


def main():
    parser = argparse.ArgumentParser(description="Run functional outcome baselines")
    parser.add_argument("--outcome", choices=["ef", "uc", "all"], default="all")
    parser.add_argument("--bertpca-model", type=str, default=None,
                        help="Path to a pre-trained BertPCa .keras model to include in comparison")
    args = parser.parse_args()

    outcomes = ["ef", "uc"] if args.outcome == "all" else [args.outcome]

    for outcome in outcomes:
        model_path = args.bertpca_model
        if model_path is None:
            # Try default location
            default = os.path.join(OUT_DIR, "models", f"best_model_{outcome}.keras")
            if os.path.exists(default):
                model_path = default
        run_outcome(outcome, bertpca_model_path=model_path)


if __name__ == "__main__":
    main()
