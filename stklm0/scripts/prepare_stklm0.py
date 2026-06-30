#!/usr/bin/env python3
"""
Prepare STKLM0 dataset (wide CSV) for BertPCa training and inference.

Outcome: cancer-specific mortality (crmort == 1); other deaths (crmort == 2)
treated as censored.

Outputs one row per PSA observation per patient in BertPCa-compatible format:
  columns: id, tte, label, times, psa, <static_features>

Also saves preprocessing_params.json (imputer medians + t_max + psa_max)
for use by the inference script.

Run from repo root:
  python stklm0/scripts/prepare_stklm0.py --input data/stklm0.csv --out-dir stklm0/data
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "stklm0", "data")
T_MAX           = 3650.0   # 10-year cap (adjust once actual follow-up range is known)
RANDOM_STATE    = 42
N_PSA           = 135      # PSA1...PSA135

# Encoding maps — must match the STKLM0 codebook
T_CLEAN_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 9: np.nan}
PT_MAP      = {0: np.nan, 2: 0, 5: 1, 3: 1, 6: 2, 4: 3, 9: np.nan}
PR_MAP      = {1: 0, 2: 1, 3: np.nan, 98: np.nan}
PN_MAP      = {0: 0, 1: 1, 9: np.nan, 98: np.nan}

STATIC_COLS = [
    "d_diaage", "d_spsa", "isup_gealson", "t_clean_ord",
    "isup_RP",  "pT_ord", "pR_bin",       "pRlenght",   "pN_bin",
]


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def encode_stklm0_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply STKLM0 codebook encodings and produce a clean static feature DataFrame.
    Input df must have one row per patient, indexed by patient ID.
    """
    out = pd.DataFrame(index=df.index)
    out["d_diaage"]    = pd.to_numeric(df["d_diaage"], errors="coerce")
    out["d_spsa"]      = pd.to_numeric(df["d_spsa"],   errors="coerce")
    out["isup_gealson"]= pd.to_numeric(df["isup_gealson"], errors="coerce")
    out["isup_RP"]     = pd.to_numeric(df["isup_RP"],  errors="coerce")

    out["t_clean_ord"] = pd.to_numeric(df["t_clean"], errors="coerce").map(T_CLEAN_MAP)
    out["pT_ord"]      = pd.to_numeric(df["pT"],      errors="coerce").map(PT_MAP)
    out["pR_bin"]      = pd.to_numeric(df["pR"],      errors="coerce").map(PR_MAP)
    out["pN_bin"]      = pd.to_numeric(df["pN"],      errors="coerce").map(PN_MAP)

    # Margin length: 0 when PSM negative, NaN-preserving otherwise
    pRlenght = pd.to_numeric(df.get("pRlenght", pd.Series(np.nan, index=df.index)),
                              errors="coerce")
    out["pRlenght"] = np.where(out["pR_bin"] == 0, 0.0, pRlenght)

    return out


# ---------------------------------------------------------------------------
# PSA wide → long conversion
# ---------------------------------------------------------------------------

def build_psa_long_stklm0(df: pd.DataFrame, t_max: float = T_MAX, n_psa: int = N_PSA) -> pd.DataFrame:
    """
    Convert PSA1...PSA{n_psa} + psadate1...psadate{n_psa} to long format.
    Surgery reference date: exp_date.
    Caps at t_max, enforces strictly monotone timestamps.
    """
    psa_cols  = [f"PSA{i}"     for i in range(1, n_psa + 1) if f"PSA{i}"     in df.columns]
    date_cols = [f"psadate{i}" for i in range(1, n_psa + 1) if f"psadate{i}" in df.columns]

    dos      = pd.to_datetime(df["exp_date"], errors="coerce")
    df_dates = df[date_cols].apply(lambda col: pd.to_datetime(col, errors="coerce"))
    df_days  = df_dates.subtract(dos, axis=0).apply(
        lambda col: col.dt.days if hasattr(col, "dt") else col.map(
            lambda x: x.days if pd.notna(x) else np.nan
        )
    )
    df_psa = df[psa_cols].copy().apply(pd.to_numeric, errors="coerce")

    neg_mask = df_days < 0
    df_days[neg_mask] = np.nan
    df_psa[neg_mask]  = np.nan

    records = []
    for pid in df.index:
        days_row = df_days.loc[pid].values.astype(float)
        psa_row  = df_psa.loc[pid].values.astype(float)
        valid = ~(np.isnan(days_row) | np.isnan(psa_row))
        t, p  = days_row[valid], psa_row[valid]
        if len(t) == 0:
            continue
        order  = np.argsort(t)
        t, p   = t[order], p[order]
        keep   = np.ones(len(t), dtype=bool)
        for i in range(1, len(t)):
            if t[i] <= t[i - 1]:
                keep[i] = False
        t, p = t[keep], p[keep]
        cap  = t <= t_max
        t, p = t[cap], p[cap]
        for ti, pi in zip(t, p):
            records.append({"id": pid, "times": ti, "psa": pi})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_long_format(df_static: pd.DataFrame,
                          df_outcome: pd.DataFrame,
                          psa_long: pd.DataFrame,
                          static_cols: list) -> pd.DataFrame:
    """
    Merge static features + outcome + PSA long format into BertPCa CSV format.
    PSA observations are NOT filtered by tte — the full trajectory within t_max
    is retained for use as model context (PSA is masked at prediction time
    inside calculate_time_dependent_c_index).
    """
    valid_ids = df_static.index.intersection(df_outcome.index).intersection(
        psa_long["id"].unique()
    )
    df_s = df_static.loc[valid_ids, static_cols].copy()
    df_s["label"] = df_outcome.loc[valid_ids, "label"]
    df_s["tte"]   = df_outcome.loc[valid_ids, "tte"]
    df_s = df_s[df_s["tte"] > 0]

    psa_sub = psa_long[psa_long["id"].isin(df_s.index)].copy().set_index("id")

    records = []
    for pid, grp in psa_sub.groupby(level=0):
        if pid not in df_s.index:
            continue
        row = df_s.loc[pid]
        for _, prow in grp.iterrows():
            rec = {"id": pid, "tte": row["tte"], "label": row["label"],
                   "times": prow["times"], "psa": prow["psa"]}
            for col in static_cols:
                rec[col] = row[col]
            records.append(rec)

    return pd.DataFrame(records).set_index("id")


# ---------------------------------------------------------------------------
# Split + impute
# ---------------------------------------------------------------------------

def split_and_impute(df_long: pd.DataFrame, static_cols: list,
                     val_frac: float = 0.10, test_frac: float = 0.10,
                     random_state: int = RANDOM_STATE):
    """
    80/10/10 patient-level stratified split; median imputation fit on train.
    Returns (train, val, test, imputer).
    """
    # .to_numpy() avoids PyArrow-backed Index errors in sklearn's _safe_indexing
    unique_ids = df_long.index.unique().to_numpy()
    labels     = df_long.groupby(level=0)["label"].first()
    total_test = val_frac + test_frac

    train_ids, tmp_ids = train_test_split(
        unique_ids, test_size=total_test, random_state=random_state,
        stratify=labels[unique_ids].to_numpy(),
    )
    val_ids, test_ids = train_test_split(
        tmp_ids, test_size=test_frac / total_test, random_state=random_state,
        stratify=labels[tmp_ids].to_numpy(),
    )

    train = df_long.loc[train_ids].copy()
    val   = df_long.loc[val_ids].copy()
    test  = df_long.loc[test_ids].copy()

    train_first = train.groupby(level=0)[static_cols].first()
    imp = SimpleImputer(strategy="median")
    imp.fit(train_first)

    for split in [train, val, test]:
        imputed = imp.transform(split[static_cols])
        split[static_cols] = imputed

    return train, val, test, imp


def save_preprocessing_params(imputer: SimpleImputer, static_cols: list,
                               t_max: float, train_df_imputed: pd.DataFrame,
                               dynamic_features: list, out_path: str):
    """
    Save preprocessing statistics to JSON for use by predict_stklm0.py.
    Includes imputer medians AND feature scaling parameters (train_max / train_min)
    computed from the post-imputation training split — this mirrors exactly what
    load_and_preprocess_data() does internally.
    """
    features_to_scale = [f for f in static_cols + dynamic_features if f != "times"]
    train_max = train_df_imputed[features_to_scale].max()
    train_min = train_df_imputed[features_to_scale].min()

    params = {
        "static_features":  static_cols,
        "dynamic_features": dynamic_features,
        "imputer_medians":  dict(zip(static_cols, imputer.statistics_.tolist())),
        "train_max":        train_max.to_dict(),
        "train_min":        train_min.to_dict(),
        "t_max":            t_max,
        "psa_max":          float(train_max.get("psa", 1.0)),
    }
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved preprocessing params to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare STKLM0 data for BertPCa")
    parser.add_argument("--input",   required=True, help="Path to STKLM0 CSV file")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--t-max",   type=float, default=T_MAX,
                        help="Max follow-up days (default 3650 = 10 years)")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)

    # Use first column as patient ID if not named 'id'
    id_col = "id" if "id" in df.columns else df.columns[0]
    df = df.set_index(id_col)
    df.index.name = "id"
    print(f"  {len(df)} patients, {len(df.columns)} columns")

    # --- Outcome ---
    df["label"] = (pd.to_numeric(df["crmort"], errors="coerce") == 1).astype(int)
    exp_date = pd.to_datetime(df["exp_date"], errors="coerce")
    t_end    = pd.to_datetime(df["t_end"],    errors="coerce")
    df["tte"] = (t_end - exp_date).dt.days.clip(lower=1, upper=args.t_max)
    df_outcome = df[["label", "tte"]].copy()

    n_ev = int(df["label"].sum())
    print(f"  Events: {n_ev} / {len(df)} ({100*n_ev/len(df):.1f}%)")

    # --- Static features ---
    print("\nEncoding features ...")
    df_static = encode_stklm0_features(df)
    print(f"  Static features: {list(df_static.columns)}")

    # --- PSA long format ---
    print(f"\nBuilding PSA long format (cap={args.t_max} days) ...")
    psa_long = build_psa_long_stklm0(df, t_max=args.t_max)
    psa_max  = float(psa_long["psa"].max()) if len(psa_long) > 0 else 1.0
    print(f"  {len(psa_long)} observations, {psa_long['id'].nunique()} patients")
    print(f"  PSA max (train scaling anchor): {psa_max:.2f}")

    # --- Assemble ---
    print("\nAssembling long-format dataset ...")
    df_long = assemble_long_format(df_static, df_outcome, psa_long, STATIC_COLS)
    n_pat = df_long.index.nunique()
    n_ev2 = int(df_long.groupby(level=0)["label"].first().sum())
    print(f"  Patients: {n_pat} | Events: {n_ev2} ({100*n_ev2/n_pat:.1f}%)")
    print(f"  Rows: {len(df_long)}")

    # --- Split + impute ---
    print("\nSplitting and imputing ...")
    train, val, test, imp = split_and_impute(df_long, STATIC_COLS)

    os.makedirs(args.out_dir, exist_ok=True)
    for split_name, split in [("stklm0_train", train), ("stklm0_val", val), ("stklm0_test", test)]:
        path = os.path.join(args.out_dir, f"{split_name}.csv")
        split.reset_index().to_csv(path, index=False)
        np_ = split.index.nunique()
        ne_ = int(split.groupby(level=0)["label"].first().sum())
        print(f"  Saved {path}: {np_} patients, {len(split)} rows, {ne_} events")

    # --- Save preprocessing params (includes train_max/train_min for inference scaling) ---
    params_path = os.path.join(args.out_dir, "preprocessing_params.json")
    save_preprocessing_params(imp, STATIC_COLS, args.t_max,
                               train_df_imputed=train,
                               dynamic_features=["times", "psa"],
                               out_path=params_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
