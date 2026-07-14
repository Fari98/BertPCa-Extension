#!/usr/bin/env python3
"""
Prepare EF and UC datasets using URI longitudinal IIEF/ICIQ data as the
dynamic feature instead of PSA.

Data sources:
  - Static features + outcome labels (IIEF_17/ttIIEF_17, ICIQ/ttICIQ):
      data/Master_Prostate_Milan_2025-09-22.RData  (same as prepare_dataset.py)
  - Dynamic feature time series:
      functional_outcomes/data/uri_ef_long.csv   (IIEF_EF per visit)
      functional_outcomes/data/uri_uc_long.csv   (iciq_sf_tot per visit)

Matching key: patientID == P_1_id in both sources.

Output format is identical to prepare_dataset.py, just with
  iief_ef  (or iciq_sf_tot)  in place of  psa
as the per-row dynamic feature column.

Run from repo root:
  python functional_outcomes/scripts/prepare_dataset_uri.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

RDATA_PATH = os.path.join("data", "Master_Prostate_Milan_2025-09-22.RData")
URI_EF_LONG = os.path.join("functional_outcomes", "data", "uri_ef_long.csv")
URI_UC_LONG = os.path.join("functional_outcomes", "data", "uri_uc_long.csv")
OUT_DIR     = os.path.join("functional_outcomes", "data")

T_MAX_UC     = 365.0   # 12 months: UC recovery peaks in first year
T_MAX_EF     = 730.0   # 24 months: EF recovery median ~20 months, need wider window
T_MAX        = T_MAX_UC  # legacy alias used by load_uri_long default
RANDOM_STATE = 42

# ttIIEF_17 and ttICIQ in the Milan RData are in MONTHS;
# observation times in the URI CSVs are in DAYS (visit_month × 30.4375).
# Both get divided by t_max in load_and_preprocess_data, so they must share units.
DAYS_PER_MONTH = 30.44  # matches URI CSV convention (visit months × 30.44 = days)


# ---------------------------------------------------------------------------
# Static feature lists (mirror prepare_dataset.py exactly)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers (shared with prepare_dataset.py)
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
    """Build PSA long format (for PSA-derived static features only)."""
    psa_cols  = [f"psa_{i}"      for i in range(1, 69)]
    date_cols = [f"date_psa_{i}" for i in range(1, 69)]
    dos = pd.to_datetime(df["dos"], errors="coerce")
    df_dates = df[date_cols].apply(lambda col: pd.to_datetime(col, errors="coerce"))
    df_days  = df_dates.subtract(dos, axis=0).apply(
        lambda col: col.dt.days if hasattr(col, "dt")
        else col.map(lambda x: x.days if pd.notna(x) else np.nan)
    )
    df_psa = df[psa_cols].copy()
    neg_mask = df_days < 0
    df_days[neg_mask] = np.nan
    df_psa[neg_mask]  = np.nan
    records = []
    for pid in df.index:
        days_row = df_days.loc[pid].values
        psa_row  = df_psa.loc[pid].values
        valid = ~(np.isnan(days_row) | np.isnan(psa_row))
        t, p = days_row[valid], psa_row[valid]
        if len(t) == 0:
            continue
        order = np.argsort(t)
        t, p = t[order], p[order]
        keep = np.ones(len(t), dtype=bool)
        for i in range(1, len(t)):
            if t[i] <= t[i - 1]:
                keep[i] = False
        t, p = t[keep], p[keep]
        cap = t <= T_MAX
        t, p = t[cap], p[cap]
        for ti, pi in zip(t, p):
            records.append({"id": pid, "times": ti, "psa": pi})
    return pd.DataFrame(records)


def compute_psa_features(psa_long):
    records = []
    for pid, grp in psa_long.groupby("id"):
        t, p = grp["times"].values, grp["psa"].values
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


# ---------------------------------------------------------------------------
# Longitudinal questionnaire series
# ---------------------------------------------------------------------------

def load_uri_long(csv_path: str, value_col: str, t_max: float = T_MAX) -> pd.DataFrame:
    """
    Load uri_ef_long.csv or uri_uc_long.csv and keep only observations
    within [0, t_max] days post-RRP.

    Returns a DataFrame with columns [id, times, <value_col>].
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"patientID": "id", value_col: value_col})
    df = df[["id", "times", value_col]].copy()
    df["id"]    = df["id"].astype(int)
    df["times"] = pd.to_numeric(df["times"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["id", "times", value_col])
    df = df[df["times"] >= 0]
    df = df[df["times"] <= t_max]
    # Sort and enforce strictly increasing timestamps per patient
    df = df.sort_values(["id", "times"]).reset_index(drop=True)
    keep_rows = []
    for pid, grp in df.groupby("id"):
        t_vals = grp["times"].values
        keep = np.ones(len(t_vals), dtype=bool)
        for i in range(1, len(t_vals)):
            if t_vals[i] <= t_vals[i - 1]:
                keep[i] = False
        keep_rows.append(grp[keep])
    df = pd.concat(keep_rows).reset_index(drop=True) if keep_rows else df.iloc[:0]
    return df


# ---------------------------------------------------------------------------
# Assemble long-format outcome dataset (like assemble_outcome in prepare_dataset.py)
# ---------------------------------------------------------------------------

def assemble_outcome_uri(df_static, dyn_long, value_col,
                          outcome_col, time_col, static_cols,
                          t_max=T_MAX_UC, tte_scale=1.0):
    """
    Build per-observation DataFrame for one outcome.
    dyn_long has columns [id, times, <value_col>].

    tte_scale converts time_col units to days (e.g. DAYS_PER_MONTH when
    the RData stores times in months but URI CSVs use days).
    Events that fall after t_max are recoded as censored at t_max.
    """
    has_outcome = df_static[outcome_col].notna() & df_static[time_col].notna()
    dyn_ids = set(dyn_long["id"].unique())
    valid_ids = df_static[has_outcome].index.intersection(list(dyn_ids))

    df_s = df_static.loc[valid_ids, static_cols + [outcome_col, time_col]].copy()
    df_s["label"] = df_s[outcome_col].astype(int)
    tte_days = df_s[time_col] * tte_scale
    # Events beyond the study window become censored at t_max
    df_s.loc[tte_days > t_max, "label"] = 0
    df_s["tte"] = tte_days.clip(upper=t_max)
    df_s = df_s[df_s["tte"] > 0]

    dyn_sub = dyn_long[dyn_long["id"].isin(df_s.index)].copy()
    dyn_sub = dyn_sub.set_index("id")

    merged_rows = []
    for pid, grp in dyn_sub.groupby(level=0):
        if pid not in df_s.index:
            continue
        row = df_s.loc[pid]
        # Keep only observations up to tte (mirrors BCR truncation at event time).
        # This ensures t_last <= tte so the Weibull loss interval is non-negative.
        # Post-recovery observations are informative but would make t_last > tte
        # and collapse t_new = max(tte - t_last, ε) to ε for almost every patient.
        grp = grp[grp["times"] <= row["tte"]]
        if grp.empty:
            continue  # no pre-event observations for this patient — skip
        for _, dyn_row in grp.iterrows():
            rec = {
                "id":       pid,
                "tte":      row["tte"],
                "label":    row["label"],
                "times":    dyn_row["times"],
                value_col:  dyn_row[value_col],
            }
            for col in static_cols:
                rec[col] = row[col]
            merged_rows.append(rec)

    out = pd.DataFrame(merged_rows).set_index("id")
    return out


# ---------------------------------------------------------------------------
# Split + impute (identical to prepare_dataset.py)
# ---------------------------------------------------------------------------

def split_and_impute(df_long, static_cols, random_state=RANDOM_STATE):
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
    val   = df_long.loc[val_ids].copy()
    test  = df_long.loc[test_ids].copy()

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
    # ── Milan RData (static features + outcome labels) ──────────────────────
    print(f"Loading {RDATA_PATH} ...")
    import pyreadr
    result = pyreadr.read_r(RDATA_PATH)
    raw = result["dat.def"].copy()
    raw["patientID"] = pd.to_numeric(raw["patientID"], errors="coerce")
    raw = raw.dropna(subset=["patientID"])
    raw["patientID"] = raw["patientID"].astype(int)
    df = raw.set_index("patientID")
    print(f"  {len(df)} patients, {len(df.columns)} columns")

    df["ece_bin"]      = (df["ece"].fillna(0) >= 1).astype(float)
    df["svi_bin"]      = (df["svi"].fillna(0) >= 1).astype(float)
    df["lni_bin"]      = (df["lni"].fillna(0) >= 1).astype(float)
    df["pathgg_group"] = df.apply(get_gleason_group, axis=1)
    df["percposnodes"] = df["positive_nodes"] / df["total_nodes"]

    for col in df.select_dtypes(include="object").columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    drs_cols  = [f"DRS_{i}" for i in range(1, 6)]
    avail_drs = [c for c in drs_cols if c in df.columns]
    df["drs_max"] = df[avail_drs].fillna(0).max(axis=1) if avail_drs else 0.0
    if "QoL_pre" not in df.columns:
        df["QoL_pre"] = np.nan

    # PSA-derived static features (still useful predictors)
    print("Building PSA long format (for static features) ...")
    psa_long  = build_psa_long(df)
    psa_feats = compute_psa_features(psa_long)
    df = df.join(psa_feats, how="left")
    print(f"  PSA-derived features added: {list(psa_feats.columns)}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── EF dataset: IIEF_EF as dynamic feature ───────────────────────────────
    print(f"\nLoading URI IIEF time series: {URI_EF_LONG}")
    ef_dyn = load_uri_long(URI_EF_LONG, "iief_ef", T_MAX_EF)
    print(f"  {len(ef_dyn)} observations across {ef_dyn['id'].nunique()} patients "
          f"(within {T_MAX_EF:.0f} days / 24 months)")

    print("Building EF dataset (IIEF_EF as dynamic feature, IIEF>=17 outcome) ...")
    ef_long = assemble_outcome_uri(
        df, ef_dyn, "iief_ef",
        outcome_col="IIEF_17", time_col="ttIIEF_17",
        static_cols=EF_STATIC,
        t_max=T_MAX_EF, tte_scale=DAYS_PER_MONTH,
    )
    print(f"  Patients: {ef_long.index.nunique()}, observations: {len(ef_long)}")
    n_events = ef_long.groupby(level=0)["label"].first().sum()
    print(f"  Events (IIEF>=17): {n_events:.0f} / {ef_long.index.nunique()}")

    ef_train, ef_val, ef_test = split_and_impute(ef_long, EF_STATIC)
    for name, split in [("uri_ef_train", ef_train), ("uri_ef_val", ef_val), ("uri_ef_test", ef_test)]:
        path = os.path.join(OUT_DIR, f"{name}.csv")
        split.reset_index().to_csv(path, index=False)
        n_pat = split.index.nunique()
        n_ev  = split.groupby(level=0)["label"].first().sum()
        print(f"  Saved {path}: {n_pat} patients, {len(split)} rows, {n_ev:.0f} events")

    # ── UC dataset: iciq_sf_tot as dynamic feature ────────────────────────────
    print(f"\nLoading URI ICIQ time series: {URI_UC_LONG}")
    uc_dyn = load_uri_long(URI_UC_LONG, "iciq_sf_tot", T_MAX_UC)
    print(f"  {len(uc_dyn)} observations across {uc_dyn['id'].nunique()} patients "
          f"(within {T_MAX_UC:.0f} days / 12 months)")

    print("Building UC dataset (iciq_sf_tot as dynamic feature, ICIQ=0 outcome) ...")
    uc_long = assemble_outcome_uri(
        df, uc_dyn, "iciq_sf_tot",
        outcome_col="ICIQ", time_col="ttICIQ",
        static_cols=UC_STATIC,
        t_max=T_MAX_UC, tte_scale=DAYS_PER_MONTH,
    )
    print(f"  Patients: {uc_long.index.nunique()}, observations: {len(uc_long)}")
    n_events_uc = uc_long.groupby(level=0)["label"].first().sum()
    print(f"  Events (ICIQ=0): {n_events_uc:.0f} / {uc_long.index.nunique()}")

    uc_train, uc_val, uc_test = split_and_impute(uc_long, UC_STATIC)
    for name, split in [("uri_uc_train", uc_train), ("uri_uc_val", uc_val), ("uri_uc_test", uc_test)]:
        path = os.path.join(OUT_DIR, f"{name}.csv")
        split.reset_index().to_csv(path, index=False)
        n_pat = split.index.nunique()
        n_ev  = split.groupby(level=0)["label"].first().sum()
        print(f"  Saved {path}: {n_pat} patients, {len(split)} rows, {n_ev:.0f} events")

    print("\nDone.")


if __name__ == "__main__":
    main()
