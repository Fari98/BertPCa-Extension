#!/usr/bin/env python3
"""
Prepare Milan dataset (Master_Prostate_Milan RData) for BertPCa training
using the STKLM0 feature schema.

Produces long-format CSVs (one row per PSA observation per patient) with
STKLM0-compatible column names for both BCR and CSM outcomes.

Features not available in Milan are automatically dropped — they are NOT
imputed or replaced with zeros.

Run from repo root:
  python stklm0/scripts/prepare_milan.py --outcome both
  python stklm0/scripts/prepare_milan.py --outcome bcr
  python stklm0/scripts/prepare_milan.py --outcome csm
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "functional_outcomes", "scripts"))

RDATA_PATH = os.path.join(_REPO_ROOT, "data", "Master_Prostate_Milan_2025-09-22.RData")
OUT_DIR    = os.path.join(_REPO_ROOT, "stklm0", "data")
T_MAX      = 3650.0   # 10 years — shared cap with STKLM0 for comparability
BCR_THRESH = 0.2      # ng/mL
RANDOM_STATE = 42

# Encoding maps — must match STKLM0 schema
T_CLEAN_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 9: np.nan}

# Clinical T stage candidates (sorted by priority — first match wins)
CLINSTAGE_CANDIDATES = ["clinstage", "cT", "clinical_t", "tstage", "cstage"]
# CSM column candidates
CSM_COL_CANDIDATES   = ["crmort", "csm", "cancer_death", "csm_event", "death_cancer"]
CSM_TIME_CANDIDATES  = ["ttcsm", "os_csm", "csm_time", "t_csm", "time_csm"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _search_columns(df, candidates, keywords=None):
    """Return first candidate column found in df, or search by keyword."""
    for col in candidates:
        if col in df.columns:
            return col
    if keywords:
        for kw in keywords:
            matches = [c for c in df.columns if kw.lower() in c.lower()]
            if matches:
                return matches[0]
    return None


def _get_gleason_group(row, primary_col="pathgg_primary", secondary_col="pathgg_secondary"):
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


def build_psa_long_milan(df, t_max=T_MAX):
    """
    Convert Milan wide PSA columns (psa_1...psa_68 + date_psa_1...date_psa_68)
    to long format with (id, times, psa). Caps at t_max, enforces monotonicity.
    """
    psa_cols  = [f"psa_{i}" for i in range(1, 69)]
    date_cols = [f"date_psa_{i}" for i in range(1, 69)]

    available_psa  = [c for c in psa_cols  if c in df.columns]
    available_date = [c for c in date_cols if c in df.columns]

    dos = pd.to_datetime(df["dos"], errors="coerce")
    df_dates = df[available_date].apply(lambda col: pd.to_datetime(col, errors="coerce"))
    df_days  = df_dates.subtract(dos, axis=0).apply(
        lambda col: col.dt.days if hasattr(col, "dt") else col.map(
            lambda x: x.days if pd.notna(x) else np.nan
        )
    )
    df_psa = df[available_psa].copy()

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


def encode_milan_features(df):
    """
    Map Milan dat.def columns to STKLM0 feature schema.
    Features without a Milan equivalent are silently dropped.
    Returns a DataFrame with only the successfully extracted features.
    """
    out = {}

    def _try(name, series_fn):
        try:
            s = series_fn()
            if s is None or (hasattr(s, "isna") and s.isna().all()):
                print(f"  [skip] '{name}' — not found or all-NaN in Milan")
                return
            out[name] = s
        except (KeyError, AttributeError, TypeError) as exc:
            print(f"  [skip] '{name}' — error: {exc}")

    _try("d_diaage",    lambda: df["age"])
    _try("d_spsa",      lambda: df["tpsa"])

    # ISUP at biopsy — biopsy Gleason grade groups
    bxgg_col = _search_columns(df, ["bxgg_group", "gleason_biopsy_group", "bx_gg"])
    if bxgg_col:
        _try("isup_gealson", lambda: df[bxgg_col])
    else:
        print("  [skip] 'isup_gealson' — no biopsy Gleason group column found")

    # Clinical T stage
    clinstage_col = _search_columns(df, CLINSTAGE_CANDIDATES, keywords=["clinstage", "cT", "cstage"])
    if clinstage_col:
        _try("t_clean_ord", lambda: df[clinstage_col].map(T_CLEAN_MAP))
    else:
        print("  [skip] 't_clean_ord' — no clinical T stage column found")

    # ISUP at RP — pathological Gleason grade groups
    if "pathgg_group" in df.columns:
        _try("isup_RP", lambda: df["pathgg_group"])
    elif "pathgg_primary" in df.columns and "pathgg_secondary" in df.columns:
        df["_pathgg_group"] = df.apply(_get_gleason_group, axis=1)
        _try("isup_RP", lambda: df["_pathgg_group"])
    else:
        print("  [skip] 'isup_RP' — no pathological Gleason group column found")

    # pT ordinal — derive from ece_bin / svi_bin if pstage not directly available
    pstage_col = _search_columns(df, ["pstage", "pT", "pathT", "path_t"], keywords=["pstage", "patht"])
    if pstage_col and df[pstage_col].nunique() > 2:
        pT_map = {0: np.nan, 2: 0, 5: 1, 3: 1, 6: 2, 4: 3, 9: np.nan}
        _try("pT_ord", lambda: df[pstage_col].map(pT_map))
    elif "ece_bin" in df.columns or "svi_bin" in df.columns:
        ece = df.get("ece_bin", pd.Series(0, index=df.index)).fillna(0)
        svi = df.get("svi_bin", pd.Series(0, index=df.index)).fillna(0)
        pT_ord = np.where(svi >= 1, 2, np.where(ece >= 1, 1, 0)).astype(float)
        _try("pT_ord", lambda: pd.Series(pT_ord, index=df.index))
    elif "ece" in df.columns or "svi" in df.columns:
        ece = (df.get("ece", pd.Series(0, index=df.index)).fillna(0) >= 1).astype(float)
        svi = (df.get("svi", pd.Series(0, index=df.index)).fillna(0) >= 1).astype(float)
        pT_ord = np.where(svi >= 1, 2, np.where(ece >= 1, 1, 0)).astype(float)
        _try("pT_ord", lambda: pd.Series(pT_ord, index=df.index))
    else:
        print("  [skip] 'pT_ord' — no pT stage or ECE/SVI columns found")

    # Positive surgical margins
    _try("pR_bin", lambda: df["psm"].map({0: 0, 1: 1}))

    # Margin length in mm
    prl_col = _search_columns(df, ["pRlenght", "margin_length", "psm_length", "pRlength"])
    if prl_col:
        _try("pRlenght", lambda: df[prl_col])
    else:
        print("  [skip] 'pRlenght' — no margin length column found")

    # Pathological N stage
    lni_col = _search_columns(df, ["lni_bin", "lni", "pN_bin", "pN"])
    if lni_col:
        _try("pN_bin", lambda: (df[lni_col].fillna(0) >= 1).astype(float))
    else:
        print("  [skip] 'pN_bin' — no LNI/pN column found")

    result = pd.DataFrame(out, index=df.index)
    print(f"  Extracted {len(result.columns)} features: {list(result.columns)}")
    return result


def derive_bcr_outcome(df_static, psa_long, static_cols, t_max=T_MAX):
    """
    Derive BCR outcome from PSA trajectory (PSA >= 0.2 ng/mL).
    PSA is truncated at BCR event time (no post-BCR readings).
    """
    psa_ids = set(psa_long["id"].unique())
    valid_ids = df_static.index.intersection(list(psa_ids))
    df_s = df_static.loc[valid_ids, static_cols].copy()

    records = []
    for pid in valid_ids:
        psa_sub = psa_long[psa_long["id"] == pid].sort_values("times")
        times = psa_sub["times"].values
        psas  = psa_sub["psa"].values

        # Find first BCR
        bcr_idx = np.where(psas >= BCR_THRESH)[0]
        if len(bcr_idx) > 0:
            bcr_i   = bcr_idx[0]
            label   = 1
            tte     = min(times[bcr_i], t_max)
            # Truncate PSA at BCR (include up to and including BCR reading)
            times_k = times[:bcr_i + 1]
            psas_k  = psas[:bcr_i + 1]
        else:
            label   = 0
            tte     = min(times[-1] if len(times) > 0 else 1.0, t_max)
            times_k = times
            psas_k  = psas

        tte = max(tte, 1.0)
        times_k = times_k[times_k <= tte]
        psas_k  = psas_k[:len(times_k)]

        for ti, pi in zip(times_k, psas_k):
            rec = {"id": pid, "tte": tte, "label": label, "times": ti, "psa": pi}
            for col in static_cols:
                rec[col] = df_s.loc[pid, col]
            records.append(rec)

    out = pd.DataFrame(records).set_index("id")
    return out


def derive_csm_outcome(df, df_static, psa_long, static_cols, csm_col, csm_time_col, t_max=T_MAX):
    """
    Derive CSM outcome from mortality column.
    PSA is NOT truncated — full trajectory within t_max provides context.
    """
    has_csm = df[csm_col].notna() & df[csm_time_col].notna()
    psa_ids = set(psa_long["id"].unique())
    valid_ids = df_static[has_csm].index.intersection(list(psa_ids))

    df_s = df_static.loc[valid_ids, static_cols].copy()
    df_s["label"] = (df.loc[valid_ids, csm_col] == 1).astype(int)
    df_s["tte"]   = df.loc[valid_ids, csm_time_col].clip(lower=1.0, upper=t_max)

    psa_sub = psa_long[psa_long["id"].isin(valid_ids)].copy().set_index("id")

    records = []
    for pid, grp in psa_sub.groupby(level=0):
        if pid not in df_s.index:
            continue
        row = df_s.loc[pid]
        for _, psa_row in grp.iterrows():
            rec = {"id": pid, "tte": row["tte"], "label": row["label"],
                   "times": psa_row["times"], "psa": psa_row["psa"]}
            for col in static_cols:
                rec[col] = row[col]
            records.append(rec)

    out = pd.DataFrame(records).set_index("id")
    return out


def split_and_impute(df_long, static_cols, random_state=RANDOM_STATE):
    """80/10/10 patient-level split stratified on label; median imputation."""
    unique_ids = df_long.index.unique()
    labels     = df_long.groupby(level=0)["label"].first()

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
    parser = argparse.ArgumentParser(description="Prepare Milan data with STKLM0 feature schema")
    parser.add_argument("--outcome", choices=["bcr", "csm", "both"], default="both")
    parser.add_argument("--rdata",   default=RDATA_PATH)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    import pyreadr
    print(f"Loading {args.rdata} ...")
    result = pyreadr.read_r(args.rdata)
    df = result["dat.def"].set_index("patientID")
    print(f"  {len(df)} patients, {len(df.columns)} columns")

    # Print potentially relevant columns to help identify CSM/T-stage columns
    for kw in ["mort", "death", "csm", "stage", "status", "crmort", "vital", "last"]:
        matches = [c for c in df.columns if kw.lower() in c.lower()]
        if matches:
            print(f"  Columns containing '{kw}': {matches[:10]}")

    # Pre-compute derived features for Milan
    if "pathgg_group" not in df.columns and "pathgg_primary" in df.columns:
        df["pathgg_group"] = df.apply(_get_gleason_group, axis=1)
    if "ece_bin" not in df.columns and "ece" in df.columns:
        df["ece_bin"] = (df["ece"].fillna(0) >= 1).astype(float)
    if "svi_bin" not in df.columns and "svi" in df.columns:
        df["svi_bin"] = (df["svi"].fillna(0) >= 1).astype(float)
    if "lni_bin" not in df.columns and "lni" in df.columns:
        df["lni_bin"] = (df["lni"].fillna(0) >= 1).astype(float)

    print("\nEncoding Milan features to STKLM0 schema ...")
    df_static = encode_milan_features(df)
    static_cols = list(df_static.columns)

    print(f"\nBuilding PSA long format (cap={T_MAX} days) ...")
    psa_long = build_psa_long_milan(df, t_max=T_MAX)
    print(f"  {len(psa_long)} observations across {psa_long['id'].nunique()} patients")

    os.makedirs(args.out_dir, exist_ok=True)
    outcomes = ["bcr", "csm"] if args.outcome == "both" else [args.outcome]

    for outcome in outcomes:
        print(f"\n--- {outcome.upper()} outcome ---")

        if outcome == "bcr":
            df_long = derive_bcr_outcome(df_static, psa_long, static_cols, t_max=T_MAX)

        else:  # csm
            csm_col  = _search_columns(df, CSM_COL_CANDIDATES,  keywords=["mort", "death", "csm"])
            csm_time = _search_columns(df, CSM_TIME_CANDIDATES,  keywords=["ttcsm", "csm_time", "os"])
            if csm_col is None or csm_time is None:
                print(f"  ERROR: could not identify CSM column (found: {csm_col}) or "
                      f"CSM time column (found: {csm_time}). "
                      "Please check the column list printed above and update CSM_COL_CANDIDATES "
                      "/ CSM_TIME_CANDIDATES in this script.")
                continue
            print(f"  CSM column: '{csm_col}', time column: '{csm_time}'")
            df_long = derive_csm_outcome(df, df_static, psa_long, static_cols,
                                         csm_col=csm_col, csm_time_col=csm_time, t_max=T_MAX)

        n_pat = df_long.index.nunique()
        n_ev  = df_long.groupby(level=0)["label"].first().sum()
        print(f"  Patients: {n_pat} | Events: {n_ev:.0f} ({100*n_ev/n_pat:.1f}%)")
        print(f"  Observations: {len(df_long)}")

        train, val, test = split_and_impute(df_long, static_cols)

        for split_name, split in [("train", train), ("val", val), ("test", test)]:
            path = os.path.join(args.out_dir, f"milan_{outcome}_{split_name}.csv")
            split.reset_index().to_csv(path, index=False)
            np = split.index.nunique()
            ne = split.groupby(level=0)["label"].first().sum()
            print(f"  Saved {path}: {np} patients, {len(split)} rows, {ne:.0f} events")

    print("\nDone.")


if __name__ == "__main__":
    main()
