#!/usr/bin/env python3
"""
Parse the URI SPSS export (TXT + SPS) into long-format CSVs for BertPCa.

Produces two files in functional_outcomes/data/:
  uri_ef_long.csv   — IIEF EF domain time series per patient
  uri_uc_long.csv   — ICIQ-SF total time series per patient

Each file has columns:
  patientID | times (days since RRP) | iief_ef | iciq_sf_tot | [source visit label]

Run from repo root:
  python functional_outcomes/scripts/parse_uri_export.py
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SPS_PATH = os.path.join("..","..","data", "20260707133325_Davide.SPS")
TXT_PATH = os.path.join("..","..","data", "20260707133325_Davide.TXT")
OUT_DIR  = os.path.join("functional_outcomes", "data")

# Approximate days per month used for monthFromRrp conversion
DAYS_PER_MONTH = 30.44

# Maximum number of FUP_N_ blocks to scan
MAX_FUP = 814

# ── Parse column names from SPS ───────────────────────────────────────────────

def parse_sps_columns(sps_path: str) -> list[str]:
    """
    Extract ordered variable names from the /VARIABLES= block of the SPS file.
    Lines have the form:  VAR_NAME  TYPE
    """
    cols = []
    in_vars = False
    with open(sps_path, "r", encoding="latin-1", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.upper().startswith("/VARIABLES="):
                in_vars = True
                rest = stripped[len("/VARIABLES="):].strip()
                if rest:
                    parts = rest.split()
                    if parts:
                        cols.append(parts[0])
                continue
            if in_vars:
                if not stripped or stripped.startswith("/") or stripped.startswith("."):
                    break
                parts = stripped.split()
                if parts:
                    cols.append(parts[0])
    return cols


def build_usecols(all_cols: list[str]) -> list[int]:
    """
    Return column indices for the subset we actually need:
      - INF_1_id
      - INF_1_iief17Recovery, INF_1_iief17RecoveryTime
      - INF_1_iciqRecovery,   INF_1_iciqRecoveryTime
      - FUP40_1_date, FUP40_1_IIEF_EF, FUP40_1_iciqSfTot (if present)
      - FUP_N_questDate, FUP_N_monthFromRrp, FUP_N_IIEF_EF, FUP_N_iciqSfTot
        for N = 1 .. MAX_FUP
    """
    wanted = {
        "INF_1_id",
        "INF_1_iief17Recovery", "INF_1_iief17RecoveryTime",
        "INF_1_iciqRecovery",   "INF_1_iciqRecoveryTime",
        "FUP40_1_date", "FUP40_1_IIEF_EF", "FUP40_1_iciqSfTot",
    }
    fup_pattern = re.compile(
        r"^FUP_(\d+)_(questDate|monthFromRrp|IIEF_EF|iciqSfTot)$"
    )
    for col in all_cols:
        m = fup_pattern.match(col)
        if m and int(m.group(1)) <= MAX_FUP:
            wanted.add(col)
    col_set = set(all_cols)
    wanted = {c for c in wanted if c in col_set}
    idx_map = {c: i for i, c in enumerate(all_cols)}
    return sorted(wanted, key=lambda c: idx_map[c])


# ── Load TXT ──────────────────────────────────────────────────────────────────

def load_txt(txt_path: str, all_cols: list[str], usecols: list[str],
             n_vars: int = 17144) -> pd.DataFrame:
    """
    The SPS uses /DELCASE=VARIABLES 17144, meaning each patient record spans
    multiple physical lines. We tokenise the entire file with the csv module
    (space-delimited, double-quote qualifier) and group every n_vars tokens
    into one case — exactly what SPSS does internally.
    """
    import csv

    n_cols = len(all_cols)
    want_idx = {c: all_cols.index(c) for c in usecols}

    print(f"  Tokenising {txt_path} (multi-line records, {n_vars} vars/case) ...")

    # Accumulate all tokens from the file
    tokens: list[str] = []
    with open(txt_path, "r", encoding="latin-1", errors="replace") as fh:
        reader = csv.reader(fh, delimiter=" ", quotechar='"', skipinitialspace=True)
        for line_tokens in reader:
            tokens.extend(line_tokens)

    total = len(tokens)
    n_cases = total // n_vars
    remainder = total % n_vars
    print(f"  Total tokens: {total:,}  →  {n_cases} cases"
          + (f"  ({remainder} trailing tokens ignored)" if remainder else ""))

    # Build rows — only keep usecols by index
    rows: list[dict] = []
    for case_i in range(n_cases):
        base = case_i * n_vars
        row = {col: tokens[base + idx] for col, idx in want_idx.items()}
        rows.append(row)

    df = pd.DataFrame(rows, columns=usecols)
    print(f"  Loaded {len(df)} rows × {len(usecols)} columns")
    return df


# ── Wide → long ───────────────────────────────────────────────────────────────

def to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        series = series.str.strip()
    return pd.to_numeric(series, errors="coerce")


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt the wide FUP_N columns into a long DataFrame with one row per
    (patient, visit) and columns: patientID, times, iief_ef, iciq_sf_tot, visit.
    """
    df = df.copy()
    df["patientID"] = to_numeric(df["INF_1_id"])

    records = []

    # --- 40-day first visit (fixed time ≈ 40 days = 1.31 months) ---
    if "FUP40_1_IIEF_EF" in df.columns:
        fup40_iief = to_numeric(df["FUP40_1_IIEF_EF"])
        fup40_iciq = to_numeric(df.get("FUP40_1_iciqSfTot", pd.Series(np.nan, index=df.index)))
        mask40 = df["patientID"].notna() & (fup40_iief.notna() | fup40_iciq.notna())
        for i in df.index[mask40]:
            pid  = df.loc[i, "patientID"]
            iief = fup40_iief.loc[i]
            iciq = fup40_iciq.loc[i]
            if pd.notna(pid):
                records.append({
                    "patientID":   int(pid),
                    "times":       40.0,
                    "iief_ef":     float(iief) if pd.notna(iief) else np.nan,
                    "iciq_sf_tot": float(iciq) if pd.notna(iciq) else np.nan,
                    "visit":       "FUP40",
                })

    # --- Regular FUP_N visits ---
    # Find which N values exist in columns
    fup_ns = sorted({
        int(m.group(1))
        for col in df.columns
        for m in [re.match(r"^FUP_(\d+)_monthFromRrp$", col)]
        if m
    })
    print(f"  Found FUP blocks: 1 .. {max(fup_ns) if fup_ns else 0} "
          f"({len(fup_ns)} with monthFromRrp)")

    for n in fup_ns:
        col_month = f"FUP_{n}_monthFromRrp"
        col_iief  = f"FUP_{n}_IIEF_EF"
        col_iciq  = f"FUP_{n}_iciqSfTot"

        months = to_numeric(df[col_month]) if col_month in df.columns else pd.Series([np.nan]*len(df))
        iief   = to_numeric(df[col_iief])  if col_iief  in df.columns else pd.Series([np.nan]*len(df))
        iciq   = to_numeric(df[col_iciq])  if col_iciq  in df.columns else pd.Series([np.nan]*len(df))
        pids   = to_numeric(df["INF_1_id"])

        mask = pids.notna() & (iief.notna() | iciq.notna()) & months.notna()
        for i in df.index[mask]:
            records.append({
                "patientID":   int(pids.loc[i]),
                "times":       float(months.loc[i]) * DAYS_PER_MONTH,
                "iief_ef":     float(iief.loc[i]) if pd.notna(iief.loc[i]) else np.nan,
                "iciq_sf_tot": float(iciq.loc[i]) if pd.notna(iciq.loc[i]) else np.nan,
                "visit":       f"FUP_{n}",
            })

    long_df = pd.DataFrame(records)
    if long_df.empty:
        print("  WARNING: no records assembled — check column names")
        return long_df

    long_df = long_df.sort_values(["patientID", "times"]).reset_index(drop=True)
    return long_df


# ── Outcome columns ───────────────────────────────────────────────────────────

def add_outcomes(long_df: pd.DataFrame, df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Attach INF_1 recovery columns so downstream scripts have tte/label.
    """
    out = df_wide[["INF_1_id",
                   "INF_1_iief17Recovery",  "INF_1_iief17RecoveryTime",
                   "INF_1_iciqRecovery",    "INF_1_iciqRecoveryTime"]].copy()
    out["patientID"]              = to_numeric(out["INF_1_id"])
    out["iief17_event"]           = (out["INF_1_iief17Recovery"].str.strip() == "1").astype(float)
    out["iief17_tte"]             = to_numeric(out["INF_1_iief17RecoveryTime"])
    out["iciq_event"]             = (out["INF_1_iciqRecovery"].str.strip() == "1").astype(float)
    out["iciq_tte"]               = to_numeric(out["INF_1_iciqRecoveryTime"])
    out = out[["patientID", "iief17_event", "iief17_tte", "iciq_event", "iciq_tte"]]
    return long_df.merge(out.dropna(subset=["patientID"]), on="patientID", how="left")


# ── Summary stats ─────────────────────────────────────────────────────────────

def summarise(long_df: pd.DataFrame, label: str) -> None:
    n_pts   = long_df["patientID"].nunique()
    n_obs   = len(long_df)
    obs_per = long_df.groupby("patientID").size()
    print(f"\n  {label}:")
    print(f"    Patients : {n_pts}")
    print(f"    Obs total: {n_obs}")
    print(f"    Obs / pt : median={obs_per.median():.1f}  max={obs_per.max()}")
    if "times" in long_df.columns:
        print(f"    Times    : min={long_df['times'].min():.0f}d  "
              f"median={long_df['times'].median():.0f}d  "
              f"max={long_df['times'].max():.0f}d")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Parsing SPS: {SPS_PATH}")
    all_cols = parse_sps_columns(SPS_PATH)
    print(f"  Total columns in SPS: {len(all_cols)}")

    usecols = build_usecols(all_cols)
    print(f"  Columns to load: {len(usecols)}")

    print(f"\nLoading TXT: {TXT_PATH}")
    df_wide = load_txt(TXT_PATH, all_cols, usecols)

    print("\nMelting to long format ...")
    long_df = wide_to_long(df_wide)
    long_df = add_outcomes(long_df, df_wide)

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- EF long (rows with IIEF_EF measured) ---
    ef_long = long_df[long_df["iief_ef"].notna()].copy()
    ef_path = os.path.join(OUT_DIR, "uri_ef_long.csv")
    ef_long.to_csv(ef_path, index=False)
    summarise(ef_long, "EF (IIEF_EF observations)")
    print(f"  Saved: {ef_path}")

    # --- UC long (rows with iciq_sf_tot measured) ---
    uc_long = long_df[long_df["iciq_sf_tot"].notna()].copy()
    uc_path = os.path.join(OUT_DIR, "uri_uc_long.csv")
    uc_long.to_csv(uc_path, index=False)
    summarise(uc_long, "UC (ICIQ-SF observations)")
    print(f"  Saved: {uc_path}")

    # --- Full long (any observation) ---
    full_path = os.path.join(OUT_DIR, "uri_full_long.csv")
    long_df.to_csv(full_path, index=False)
    print(f"\n  Full long saved: {full_path}")

    # --- Coverage vs Milan RData ---
    print("\n  Patient ID range in URI export:")
    ids = long_df["patientID"].dropna().unique()
    print(f"    min={int(ids.min())}  max={int(ids.max())}  n={len(ids)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
