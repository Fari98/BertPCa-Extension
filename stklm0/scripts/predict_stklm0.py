#!/usr/bin/env python3
"""
Run inference with a saved BertPCa model on new STKLM0-format patient data.

No training is performed. Outputs predicted survival probabilities
at specified time horizons for each patient.

Run from repo root:
  python stklm0/scripts/predict_stklm0.py \\
      --input new_patients.csv \\
      --model stklm0/outputs/models/best_model_stklm0.keras \\
      --params stklm0/data/preprocessing_params.json \\
      --e-times 365 1825 3650 \\
      --out stklm0/outputs/predictions/predictions.csv
"""

import os
import sys
import json
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

from bertpca.data import preprocess_data
from bertpca.loss import weibull_loss

# Import STKLM0 preprocessing helpers from prepare_stklm0.py
sys.path.insert(0, os.path.join(_REPO_ROOT, "stklm0", "scripts"))
from prepare_stklm0 import encode_stklm0_features, build_psa_long_stklm0, STATIC_COLS


def load_params(params_path: str) -> dict:
    with open(params_path, "r") as f:
        return json.load(f)


def preprocess_for_inference(df_raw: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Apply STKLM0 encoding + median imputation + feature scaling.
    Uses the preprocessing parameters saved during training (no re-fitting).
    """
    t_max   = params["t_max"]
    psa_max = params["psa_max"]
    medians = params["imputer_medians"]
    static_features = params["static_features"]

    # Encode features
    df_static = encode_stklm0_features(df_raw)

    # Build PSA long format
    psa_long = build_psa_long_stklm0(df_raw, t_max=t_max)
    if len(psa_long) == 0:
        raise ValueError("No valid PSA observations found in input data.")

    # Assemble long format with placeholder tte/label (not needed for inference)
    valid_ids = df_static.index.intersection(psa_long["id"].unique())
    df_s = df_static.loc[valid_ids, static_features].copy()
    df_s["tte"]   = t_max   # placeholder
    df_s["label"] = 0       # placeholder
    df_s["t_last"] = (
        psa_long[psa_long["id"].isin(valid_ids)]
        .groupby("id")["times"].max()
    )

    psa_sub = psa_long[psa_long["id"].isin(valid_ids)].set_index("id")

    records = []
    for pid, grp in psa_sub.groupby(level=0):
        if pid not in df_s.index:
            continue
        row = df_s.loc[pid]
        for _, prow in grp.iterrows():
            rec = {"id": pid, "tte": row["tte"], "label": row["label"],
                   "times": prow["times"], "psa": prow["psa"]}
            for col in static_features:
                rec[col] = row[col]
            records.append(rec)

    df_long = pd.DataFrame(records).set_index("id")

    # Apply saved median imputation (no re-fitting)
    for col in static_features:
        if col in medians:
            df_long[col] = df_long[col].fillna(medians[col])

    # Apply the same min-max scaling that load_and_preprocess_data() applies during training.
    # train_max and train_min were computed from the post-imputation training split and
    # saved to preprocessing_params.json by prepare_stklm0.py.
    train_max = params.get("train_max", {})
    train_min = params.get("train_min", {})
    features_to_scale = [f for f in static_features + ["psa"] if f != "times"]
    for col in features_to_scale:
        if col in train_max and col in train_min:
            denom = train_max[col] - train_min[col]
            if denom == 0:
                denom = 1.0
            df_long[col] = (df_long[col] - train_min[col]) / denom
        elif col == "psa":
            # Fallback: scale psa by psa_max if train_max/train_min not in params
            df_long[col] = df_long[col] / psa_max

    df_long["times"] = df_long["times"] / t_max
    df_long["tte"]   = df_long["tte"]   / t_max

    return df_long, df_s["t_last"]


def compute_survival(alpha_raw: np.ndarray, beta_raw: np.ndarray,
                     t_last_scaled: np.ndarray, e_times: list,
                     t_max: float) -> np.ndarray:
    """
    Compute conditional survival P(T > e_time | T > t_last) for each patient.

    Weibull parameterisation matches weibull_loss:
      alpha_eff = alpha_raw + 1
      beta_eff  = beta_raw  + 1
      P(T > t | T > t_last) = exp(-((t - t_last) / alpha_eff)^beta_eff)
    """
    alpha_eff = alpha_raw + 1.0   # shape (n_patients,)
    beta_eff  = beta_raw  + 1.0

    probs = np.zeros((len(alpha_raw), len(e_times)))
    for j, et in enumerate(e_times):
        et_scaled = et / t_max
        interval  = np.maximum(et_scaled - t_last_scaled, 1e-5)
        probs[:, j] = np.exp(-np.power(interval / alpha_eff, beta_eff))

    return probs


def run(input_path: str, model_path: str, params_path: str,
        e_times: list, output_path: str):

    # ---- Load preprocessing params -----------------------------------------------
    print(f"Loading preprocessing params from {params_path} ...")
    params = load_params(params_path)
    t_max   = params["t_max"]
    static_features = params["static_features"]
    print(f"  t_max={t_max}, static features: {static_features}")

    # ---- Load and preprocess input data ------------------------------------------
    print(f"\nLoading input data from {input_path} ...")
    df_raw = pd.read_csv(input_path)
    id_col = "id" if "id" in df_raw.columns else df_raw.columns[0]
    df_raw = df_raw.set_index(id_col)
    df_raw.index.name = "id"
    print(f"  {len(df_raw)} patients")

    df_long, t_last_series = preprocess_for_inference(df_raw, params)
    patient_ids = df_long.index.unique().tolist()
    print(f"  {len(patient_ids)} patients with valid PSA data")

    # ---- Build feature tensor ----------------------------------------------------
    seq_length = 16  # must match training config; could be loaded from params
    ds, _ = preprocess_data(
        df_long, static_features, ["times", "psa"], "label",
        seq_length=seq_length, batch_size=len(patient_ids),
    )
    features = np.array(ds["features"])  # (n_patients, n_features, seq_length)

    # ---- Load model and predict ---------------------------------------------------
    print(f"\nLoading model from {model_path} ...")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"weibull_loss": weibull_loss},
    )
    print("Running inference ...")
    raw_preds = model.predict(features, verbose=0)  # (n_patients, 2)
    alpha_raw = raw_preds[:, 0]
    beta_raw  = raw_preds[:, 1]

    # ---- Compute survival probabilities ------------------------------------------
    t_last_days = t_last_series.loc[patient_ids].values
    t_last_scaled = t_last_days / t_max

    print(f"Computing survival at e_times: {e_times} days ...")
    probs = compute_survival(alpha_raw, beta_raw, t_last_scaled, e_times, t_max)

    # ---- Build output DataFrame --------------------------------------------------
    out = pd.DataFrame({"patient_id": patient_ids})
    out["alpha_raw"]   = alpha_raw
    out["beta_raw"]    = beta_raw
    out["t_last_days"] = t_last_days

    for j, et in enumerate(e_times):
        label = f"P(T>{int(et)}d)"
        out[label] = probs[:, j]

    # Risk score: 1 - P(survive to first e_time)
    out["risk_score"] = 1.0 - probs[:, 0]

    # ---- Save -------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    print(out.head(5).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Run BertPCa inference on new STKLM0-format patient data"
    )
    parser.add_argument("--input",   required=True,
                        help="Path to new patient CSV (STKLM0 schema)")
    parser.add_argument("--model",   required=True,
                        help="Path to saved .keras model")
    parser.add_argument("--params",  required=True,
                        help="Path to preprocessing_params.json")
    parser.add_argument("--e-times", nargs="+", type=float,
                        default=[365.0, 1825.0, 3650.0],
                        help="Evaluation time horizons in days (default: 365 1825 3650)")
    parser.add_argument("--out",     default=os.path.join(
                            _REPO_ROOT, "stklm0", "outputs", "predictions", "predictions.csv"),
                        help="Output CSV path")
    args = parser.parse_args()

    run(
        input_path=os.path.join(_REPO_ROOT, args.input) if not os.path.isabs(args.input) else args.input,
        model_path=os.path.join(_REPO_ROOT, args.model) if not os.path.isabs(args.model) else args.model,
        params_path=os.path.join(_REPO_ROOT, args.params) if not os.path.isabs(args.params) else args.params,
        e_times=args.e_times,
        output_path=os.path.join(_REPO_ROOT, args.out) if not os.path.isabs(args.out) else args.out,
    )


if __name__ == "__main__":
    main()
