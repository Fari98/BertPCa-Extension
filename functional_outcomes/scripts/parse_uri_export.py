#!/usr/bin/env python3
"""
Parse the URI SPSS export (TXT + SPS) into long-format CSVs for BertPCa.

Findings from reverse-engineering the file format:
  - Each physical line = one patient record (20,013 patients)
  - Each line has 17,145 tokens (SPS declares 17,144 — last token is always empty, ignored)
  - Token[N] maps exactly to SPS variable N (no alignment shift when reading line-by-line)
  - Prostate patients WITH questionnaire data have INF_1_id non-empty
    (patients with P_1_prostateProtocolType='2' are newer enrolments with no data yet)
  - Italian locale: decimal comma in numeric fields (e.g. '1,00' → 1.0)

Produces:
  functional_outcomes/data/uri_ef_long.csv   — IIEF EF domain time series
  functional_outcomes/data/uri_uc_long.csv   — ICIQ-SF total time series
  functional_outcomes/data/uri_full_long.csv — combined

Run from repo root:
  python functional_outcomes/scripts/parse_uri_export.py
"""

import os
import re
import csv
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SPS_PATH = os.path.join("..","..","data", "20260707133325_Davide.SPS")
TXT_PATH = os.path.join("..","..","data", "20260707133325_Davide.TXT")
OUT_DIR  = os.path.join("functional_outcomes", "data")

DAYS_PER_MONTH = 30.44
MAX_FUP        = 814      # max FUP_N_ block index in the SPS


# ── Parse SPS column names ────────────────────────────────────────────────────

def parse_sps_columns(sps_path: str) -> list[str]:
    cols: list[str] = []
    in_vars = False
    with open(sps_path, "r", encoding="latin-1", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.upper().startswith("/VARIABLES="):
                in_vars = True
                rest = s[len("/VARIABLES="):].strip()
                if rest:
                    cols.append(rest.split()[0])
                continue
            if in_vars:
                if not s or s.startswith("/") or s.startswith("."):
                    break
                parts = s.split()
                if parts:
                    cols.append(parts[0])
    return cols


# ── Identify which columns to keep ───────────────────────────────────────────

def wanted_cols(all_cols: list[str]) -> list[str]:
    """
    Return the subset of SPS column names we actually need, in SPS order.
    """
    want = {
        "P_1_id", "P_1_idCentral",
        "INF_1_id",
        "INF_1_iief17Recovery", "INF_1_iief17RecoveryTime",
        "INF_1_iciqRecovery",   "INF_1_iciqRecoveryTime",
        "FUP40_1_questIsCompiled", "FUP40_1_IIEF_EF", "FUP40_1_iciqSfTot",
        "FUP40_1_monthFromRrp",
    }
    fup_pat = re.compile(r"^FUP_(\d+)_(questIsCompiled|monthFromRrp|IIEF_EF|iciqSfTot)$")
    for col in all_cols:
        m = fup_pat.match(col)
        if m and int(m.group(1)) <= MAX_FUP:
            want.add(col)
    col_set = set(all_cols)
    want = {c for c in want if c in col_set}
    idx_map = {c: i for i, c in enumerate(all_cols)}
    return sorted(want, key=lambda c: idx_map[c])


# ── Load TXT (line-by-line) ────────────────────────────────────────────────────

def load_txt(txt_path: str, all_cols: list[str], keep_cols: list[str]) -> pd.DataFrame:
    """
    Read the TXT file line-by-line (each physical line = one patient record).
    Each line has 17,145 tokens; SPS declares 17,144 — the last token is always
    an empty trailing field and is ignored.  Token[N] maps directly to all_cols[N].
    """
    n_cols   = len(all_cols)                               # 17,144 (SPS declaration)
    keep_idx = {col: all_cols.index(col) for col in keep_cols}

    rows: list[dict] = []
    n_lines = 0
    with open(txt_path, "r", encoding="latin-1", errors="replace") as fh:
        reader = csv.reader(fh, delimiter=" ", quotechar='"', skipinitialspace=True)
        for line_tokens in reader:
            n_lines += 1
            # line_tokens has 17,145 entries; slice to n_cols to drop the trailing empty
            row = {col: (line_tokens[ti] if ti < len(line_tokens) else "")
                   for col, ti in keep_idx.items()}
            rows.append(row)

    df = pd.DataFrame(rows, columns=keep_cols)
    print(f"  Loaded {n_lines} patients × {len(keep_cols)} selected columns")
    return df


# ── Numeric conversion (handles Italian decimal comma) ───────────────────────

def to_num(series: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        series = series.str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


# ── Wide → long ────────────────────────────────────────────────────────────────

def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only patients with INF_1_id (the old URI prostate protocol cohort),
    then melt FUP visits into long format.
    """
    df = df.copy()

    # INF_1_id is filled for patients enrolled in the old prostate follow-up
    # program (the cohort with IIEF / ICIQ questionnaire data).
    inf_id_num = to_num(df["INF_1_id"])
    prostate_mask = inf_id_num.notna()
    print(f"  Patients with INF_1_id (old prostate cohort): "
          f"{prostate_mask.sum()} / {len(df)} total")
    df = df[prostate_mask].copy()

    # Primary patient key: P_1_id (URI sequential patient number)
    df["patientID"] = to_num(df["P_1_id"])

    records: list[dict] = []

    # --- FUP40 visit (~40 days post-RRP) ---
    if "FUP40_1_IIEF_EF" in df.columns:
        iief40 = to_num(df["FUP40_1_IIEF_EF"])
        iciq40 = to_num(df.get("FUP40_1_iciqSfTot",
                                pd.Series(np.nan, index=df.index)))
        mon40  = to_num(df.get("FUP40_1_monthFromRrp",
                                pd.Series(np.nan, index=df.index)))
        mask40 = df["patientID"].notna() & (iief40.notna() | iciq40.notna())
        for i in df.index[mask40]:
            months = mon40.loc[i] if pd.notna(mon40.loc[i]) else 40.0 / DAYS_PER_MONTH
            records.append({
                "patientID":   int(df.loc[i, "patientID"]),
                "times":       float(months) * DAYS_PER_MONTH,
                "iief_ef":     float(iief40.loc[i]) if pd.notna(iief40.loc[i]) else np.nan,
                "iciq_sf_tot": float(iciq40.loc[i]) if pd.notna(iciq40.loc[i]) else np.nan,
                "visit":       "FUP40",
            })

    # --- Regular FUP_N visits ---
    fup_ns = sorted({
        int(m.group(1))
        for col in df.columns
        for m in [re.match(r"^FUP_(\d+)_monthFromRrp$", col)]
        if m
    })
    if fup_ns:
        print(f"  FUP blocks: 1 .. {max(fup_ns)} ({len(fup_ns)} with monthFromRrp)")

    for n in fup_ns:
        col_month = f"FUP_{n}_monthFromRrp"
        col_iief  = f"FUP_{n}_IIEF_EF"
        col_iciq  = f"FUP_{n}_iciqSfTot"

        months = to_num(df[col_month]) if col_month in df.columns else pd.Series(np.nan, index=df.index)
        iief   = to_num(df[col_iief])  if col_iief  in df.columns else pd.Series(np.nan, index=df.index)
        iciq   = to_num(df[col_iciq])  if col_iciq  in df.columns else pd.Series(np.nan, index=df.index)
        pids   = df["patientID"]

        mask = pids.notna() & (iief.notna() | iciq.notna()) & months.notna()
        for i in df.index[mask]:
            records.append({
                "patientID":   int(pids.loc[i]),
                "times":       float(months.loc[i]) * DAYS_PER_MONTH,
                "iief_ef":     float(iief.loc[i]) if pd.notna(iief.loc[i]) else np.nan,
                "iciq_sf_tot": float(iciq.loc[i]) if pd.notna(iciq.loc[i]) else np.nan,
                "visit":       f"FUP_{n}",
            })

    if not records:
        print("  WARNING: no records — check column names and filters")
        return pd.DataFrame(columns=["patientID", "times", "iief_ef", "iciq_sf_tot", "visit"])

    long_df = pd.DataFrame(records).sort_values(["patientID", "times"]).reset_index(drop=True)
    return long_df


# ── Attach outcome labels (tte / event) ──────────────────────────────────────

def add_outcomes(long_df: pd.DataFrame, df_wide: pd.DataFrame) -> pd.DataFrame:
    inf_id_num = to_num(df_wide["INF_1_id"])
    df_w = df_wide[inf_id_num.notna()].copy()
    df_w["patientID"]    = to_num(df_w["P_1_id"])
    df_w["iief17_event"] = (df_w["INF_1_iief17Recovery"].str.strip() == "1").astype(float)
    df_w["iief17_tte"]   = to_num(df_w["INF_1_iief17RecoveryTime"])
    df_w["iciq_event"]   = (df_w["INF_1_iciqRecovery"].str.strip() == "1").astype(float)
    df_w["iciq_tte"]     = to_num(df_w["INF_1_iciqRecoveryTime"])

    out = df_w[["patientID", "iief17_event", "iief17_tte",
                "iciq_event", "iciq_tte"]].dropna(subset=["patientID"])
    return long_df.merge(out, on="patientID", how="left")


# ── Summary ───────────────────────────────────────────────────────────────────

def summarise(df: pd.DataFrame, label: str) -> None:
    n_pts = df["patientID"].nunique()
    n_obs = len(df)
    obs_pp = df.groupby("patientID").size()
    print(f"\n  {label}:")
    print(f"    Patients : {n_pts}")
    print(f"    Obs total: {n_obs}")
    if n_pts:
        print(f"    Obs / pt : median={obs_pp.median():.1f}  max={obs_pp.max()}")
        print(f"    Times    : min={df['times'].min():.0f}d  "
              f"median={df['times'].median():.0f}d  "
              f"max={df['times'].max():.0f}d")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Parsing SPS: {SPS_PATH}")
    all_cols = parse_sps_columns(SPS_PATH)
    print(f"  Total SPS columns: {len(all_cols)}")

    keep = wanted_cols(all_cols)
    print(f"  Columns to load: {len(keep)}")

    print(f"\nLoading TXT (line-by-line): {TXT_PATH}")
    df_wide = load_txt(TXT_PATH, all_cols, keep)

    print("\nMelting to long format ...")
    long_df = wide_to_long(df_wide)
    long_df = add_outcomes(long_df, df_wide)

    os.makedirs(OUT_DIR, exist_ok=True)

    ef_long = long_df[long_df["iief_ef"].notna()].copy()
    ef_path = os.path.join(OUT_DIR, "uri_ef_long.csv")
    ef_long.to_csv(ef_path, index=False)
    summarise(ef_long, "EF (IIEF_EF observations)")
    print(f"  Saved: {ef_path}")

    uc_long = long_df[long_df["iciq_sf_tot"].notna()].copy()
    uc_path = os.path.join(OUT_DIR, "uri_uc_long.csv")
    uc_long.to_csv(uc_path, index=False)
    summarise(uc_long, "UC (ICIQ-SF observations)")
    print(f"  Saved: {uc_path}")

    full_path = os.path.join(OUT_DIR, "uri_full_long.csv")
    long_df.to_csv(full_path, index=False)
    print(f"\n  Full long saved: {full_path}")

    # Cross-check patient IDs with Milan RData
    ids = long_df["patientID"].dropna().astype(int).unique()
    print(f"\n  URI prostate patients: n={len(ids)}  "
          f"min={ids.min() if len(ids) else 'N/A'}  max={ids.max() if len(ids) else 'N/A'}")

    rdata_path = os.path.join("data", "Master_Prostate_Milan_2025-09-22.RData")
    if os.path.exists(rdata_path):
        try:
            import pyreadr
            milan_ids = set(
                pyreadr.read_r(rdata_path)["dat.def"]["patientID"].dropna().astype(int)
            )
            uri_ids = set(ids)
            overlap = milan_ids & uri_ids
            print(f"\n  Milan RData patients   : {len(milan_ids)}")
            print(f"  URI prostate patients  : {len(uri_ids)}")
            print(f"  Overlap (matched IDs)  : {len(overlap)}")
            if overlap:
                sample = sorted(overlap)[:5]
                print(f"  Sample matching IDs    : {sample}")
            print(f"  In Milan but not URI   : {len(milan_ids - uri_ids)}")
            print(f"  In URI but not Milan   : {len(uri_ids - milan_ids)}")
            # Also try P_1_idCentral
            uri_central = to_num(df_wide[to_num(df_wide["INF_1_id"]).notna()]["P_1_idCentral"])
            uri_central_set = set(uri_central.dropna().astype(int))
            overlap_central = milan_ids & uri_central_set
            print(f"\n  Overlap via P_1_idCentral: {len(overlap_central)}")
        except Exception as e:
            print(f"  (RData cross-check failed: {e})")

    print("\nDone.")


if __name__ == "__main__":
    main()
