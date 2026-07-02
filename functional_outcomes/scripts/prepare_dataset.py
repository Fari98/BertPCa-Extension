#!/usr/bin/env python3
"""
Prepare EF and UC datasets from Master_Prostate_Milan RData file.

Outputs one row per PSA observation per patient, in BertPCa-compatible CSV format:
  columns: id, tte, label, times, psa, <static features>

Run from the repo root:
  python functional_outcomes/scripts/prepare_dataset.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

RDATA_PATH = os.path.join("data", "Master_Prostate_Milan_2025-09-22.RData")
OUT_DIR = os.path.join("functional_outcomes", "data")
T_MAX = 365.0  # 1 year in days (max observed ttIIEF / ttICIQ ~ 395 days)
MIN_PSA_OBS = 3
RANDOM_STATE = 42

PSA_DERIVED = ["psa_nadir", "time_to_nadir", "psa_at_last_obs", "psa_slope", "n_psa_obs"]
EXTRA_STATIC = ["QoL_pre", "drs_max"]

EF_STATIC = [
    "nerve_sparing", "IIEF_EFdomain_pre", "age", "tpsa", "bmi",
    "pathgg_group", "ece_bin", "svi_bin", "psm", "lni_bin",
    "neo_adjHT", "pstage",
] + EXTRA_STATIC + PSA_DERIVED

UC_STATIC = [
    "nerve_sparing", "IPSS_pre", "age", "tpsa", "bmi",
    "pathgg_group", "ece_bin", "svi_bin", "psm",
    "prostate_vol", "operative_time",
] + EXTRA_STATIC + PSA_DERIVED

WRONG_DATE_VALS = {
    "20013": "2013", "2919": "2019", "3013": "2013", "2917": "2017",
    "2103": "2013", "2208": "2008", "2026": "2016", "2118": "2018",
    "2201": "2014", "2029": "2020", "2200": "2009",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_gleason_group(row, primary_col="pathgg_primary", secondary_col="pathgg_secondary"):
    p, s = row[primary_col], row[secondary_col]
    if pd.isna(p) or pd.isna(s):
        return np.nan
    score = p + s
    if score <= 6:
        return 1
    if p == 3 and s == 4:
        return 2
    if p == 4 and s == 3:
        return 3
    if p == 4 and s == 4:
        return 4
    if score >= 9:
        return 5
    return np.nan


def build_psa_long(df):
    """
    Convert wide PSA columns to a long DataFrame with columns
    [patientID, times (days since surgery), psa].
    Removes non-monotone timestamps and caps at T_MAX.

    Date columns from pyreadr are already datetime.date objects,
    so no string correction is needed.
    """
    psa_cols = [f"psa_{i}" for i in range(1, 69)]
    date_cols = [f"date_psa_{i}" for i in range(1, 69)]

    dos = pd.to_datetime(df["dos"], errors="coerce")

    # Convert datetime.date objects to pandas Timestamps
    df_dates = df[date_cols].apply(lambda col: pd.to_datetime(col, errors="coerce"))

    # Days since surgery
    df_days = df_dates.subtract(dos, axis=0).apply(
        lambda col: col.dt.days if hasattr(col, "dt") else col.map(
            lambda x: x.days if pd.notna(x) else np.nan
        )
    )

    df_psa = df[psa_cols].copy()

    # Zero out negative-day readings
    neg_mask = df_days < 0
    df_days[neg_mask] = np.nan
    df_psa[neg_mask] = np.nan

    # Build per-patient lists
    records = []
    for pid in df.index:
        days_row = df_days.loc[pid].values
        psa_row = df_psa.loc[pid].values
        valid = ~(np.isnan(days_row) | np.isnan(psa_row))
        t = days_row[valid]
        p = psa_row[valid]
        if len(t) == 0:
            continue
        # Sort by time
        order = np.argsort(t)
        t, p = t[order], p[order]
        # Remove non-strictly-increasing timestamps
        keep = np.ones(len(t), dtype=bool)
        for i in range(1, len(t)):
            if t[i] <= t[i - 1]:
                keep[i] = False
        t, p = t[keep], p[keep]
        # Cap at T_MAX
        cap = t <= T_MAX
        t, p = t[cap], p[cap]
        for ti, pi in zip(t, p):
            records.append({"id": pid, "times": ti, "psa": pi})

    return pd.DataFrame(records)


def compute_psa_features(psa_long: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-patient PSA summary statistics from the long-format PSA DataFrame.

    Returns a DataFrame indexed by patient ID with columns:
      psa_nadir, time_to_nadir, psa_at_last_obs, psa_slope, n_psa_obs

    psa_slope is NaN for patients with only 1 observation (median-imputed downstream).
    """
    records = []
    for pid, grp in psa_long.groupby("id"):
        t = grp["times"].values  # sorted ascending (guaranteed by build_psa_long)
        p = grp["psa"].values
        n = len(t)
        nadir_idx = int(np.argmin(p))
        slope = float(linregress(t, p).slope) if n >= 2 else np.nan
        records.append({
            "id":              pid,
            "psa_nadir":       float(p[nadir_idx]),
            "time_to_nadir":   float(t[nadir_idx]),
            "psa_at_last_obs": float(p[-1]),
            "psa_slope":       slope,
            "n_psa_obs":       float(n),
        })
    return pd.DataFrame(records).set_index("id")


def assemble_outcome(df_static, psa_long, outcome_col, time_col, static_cols, label_name="label"):
    """
    Build a per-observation DataFrame for one outcome.

    tte = time_col (contains both event time for recovered patients and
    censoring time for non-recovered patients — both are present in the data).

    PSA observations are NOT filtered by tte: the model uses the full PSA
    trajectory as context. At evaluation, PSA is masked to prediction time p
    inside calculate_time_dependent_c_index.
    """
    # Patients with both outcome variable and at least one PSA observation
    has_outcome = df_static[outcome_col].notna() & df_static[time_col].notna()
    psa_ids = set(psa_long["id"].unique())
    valid_ids = df_static[has_outcome].index.intersection(list(psa_ids))

    df_s = df_static.loc[valid_ids, static_cols + [outcome_col, time_col]].copy()
    df_s["label"] = df_s[outcome_col].astype(int)
    df_s["tte"] = df_s[time_col].clip(upper=T_MAX)
    # Drop the few cases where tte == 0 (no meaningful follow-up)
    df_s = df_s[df_s["tte"] > 0]

    # Merge with PSA long format (all PSA within T_MAX, regardless of tte)
    psa_sub = psa_long[psa_long["id"].isin(df_s.index)].copy()
    psa_sub = psa_sub.set_index("id")

    merged_rows = []
    for pid, grp in psa_sub.groupby(level=0):
        if pid not in df_s.index:
            continue
        row = df_s.loc[pid]
        for _, psa_row in grp.iterrows():
            rec = {"id": pid, "tte": row["tte"], "label": row["label"],
                   "times": psa_row["times"], "psa": psa_row["psa"]}
            for col in static_cols:
                rec[col] = row[col]
            merged_rows.append(rec)

    out = pd.DataFrame(merged_rows)
    out = out.set_index("id")
    return out


def split_and_impute(df_long, static_cols, random_state=RANDOM_STATE):
    """
    Split by unique patient IDs (80/10/10 stratified on label),
    then median-impute static columns (fit on train).
    """
    unique_ids = df_long.index.unique()
    labels = df_long.groupby(level=0)["label"].first()

    train_ids, tmp_ids = train_test_split(
        unique_ids, test_size=0.20, random_state=random_state,
        stratify=labels[unique_ids],
    )
    val_ids, test_ids = train_test_split(
        tmp_ids, test_size=0.50, random_state=random_state,
        stratify=labels[tmp_ids],
    )

    train = df_long.loc[train_ids].copy()
    val = df_long.loc[val_ids].copy()
    test = df_long.loc[test_ids].copy()

    # Impute static columns (median from train, per-column first occurrence)
    train_first = train.groupby(level=0)[static_cols].first()
    imp = SimpleImputer(strategy="median")
    imp.fit(train_first)

    for split in [train, val, test]:
        imputed = imp.transform(split[static_cols])
        split[static_cols] = imputed

    return train, val, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {RDATA_PATH} ...")
    import pyreadr
    result = pyreadr.read_r(RDATA_PATH)
    df = result["dat.def"].set_index("patientID")
    print(f"  Loaded {len(df)} patients, {len(df.columns)} columns")

    # --- Derived / binarized features ---
    df["ece_bin"] = (df["ece"].fillna(0) >= 1).astype(float)
    df["svi_bin"] = (df["svi"].fillna(0) >= 1).astype(float)
    df["lni_bin"] = (df["lni"].fillna(0) >= 1).astype(float)
    df["pathgg_group"] = df.apply(get_gleason_group, axis=1)
    df["percposnodes"] = df["positive_nodes"] / df["total_nodes"]

    # Force numeric where possible
    for col in df.select_dtypes(include="object").columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    # --- PSA long format ---
    print("Building PSA long format ...")
    psa_long = build_psa_long(df)
    print(f"  {len(psa_long)} PSA observations across {psa_long['id'].nunique()} patients")

    # --- PSA-derived static features ---
    print("Computing PSA-derived static features ...")
    psa_feats = compute_psa_features(psa_long)
    df = df.join(psa_feats, how="left")
    print(f"  Added: {list(psa_feats.columns)}")

    # --- Complication severity (Clavien-Dindo proxy) ---
    drs_cols = [f"DRS_{i}" for i in range(1, 6)]
    avail_drs = [c for c in drs_cols if c in df.columns]
    if avail_drs:
        df["drs_max"] = df[avail_drs].fillna(0).max(axis=1)
    else:
        df["drs_max"] = 0.0

    # --- Pre-operative Quality of Life ---
    if "QoL_pre" not in df.columns:
        df["QoL_pre"] = np.nan

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- EF dataset (IIEF>=26: full recovery, harder threshold, longer tte) ---
    # Restrict to patients with >= MIN_PSA_OBS PSA observations so the
    # transformer has actual longitudinal signal to attend to.
    psa_counts = psa_long.groupby("id").size()
    ef_psa_long = psa_long[psa_long["id"].isin(psa_counts[psa_counts >= MIN_PSA_OBS].index)]
    print(f"\nBuilding EF dataset (IIEF EF domain >= 26, PSA obs >= {MIN_PSA_OBS}) ...")
    print(f"  Patients with >= {MIN_PSA_OBS} PSA obs: {ef_psa_long['id'].nunique()} / {psa_long['id'].nunique()}")
    ef_long = assemble_outcome(
        df, ef_psa_long,
        outcome_col="IIEF_26", time_col="ttIIEF_26",
        static_cols=EF_STATIC,
    )
    print(f"  Patients: {ef_long.index.nunique()}, observations: {len(ef_long)}")
    print(f"  Events: {ef_long.groupby(level=0)['label'].first().sum():.0f} / {ef_long.index.nunique()}")
    ef_train, ef_val, ef_test = split_and_impute(ef_long, EF_STATIC)
    for name, split in [("ef_train", ef_train), ("ef_val", ef_val), ("ef_test", ef_test)]:
        path = os.path.join(OUT_DIR, f"{name}.csv")
        split.reset_index().to_csv(path, index=False)
        n_pat = split.index.nunique()
        n_ev = split.groupby(level=0)["label"].first().sum()
        print(f"  Saved {path}: {n_pat} patients, {len(split)} rows, {n_ev:.0f} events")

    # --- UC dataset ---
    print("\nBuilding UC dataset (ICIQ = 1, continent) ...")
    uc_long = assemble_outcome(
        df, psa_long,
        outcome_col="ICIQ", time_col="ttICIQ",
        static_cols=UC_STATIC,
    )
    print(f"  Patients: {uc_long.index.nunique()}, observations: {len(uc_long)}")
    print(f"  Events: {uc_long.groupby(level=0)['label'].first().sum():.0f} / {uc_long.index.nunique()}")
    uc_train, uc_val, uc_test = split_and_impute(uc_long, UC_STATIC)
    for name, split in [("uc_train", uc_train), ("uc_val", uc_val), ("uc_test", uc_test)]:
        path = os.path.join(OUT_DIR, f"{name}.csv")
        split.reset_index().to_csv(path, index=False)
        n_pat = split.index.nunique()
        n_ev = split.groupby(level=0)["label"].first().sum()
        print(f"  Saved {path}: {n_pat} patients, {len(split)} rows, {n_ev:.0f} events")

    print("\nDone.")


if __name__ == "__main__":
    main()
