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

# Encoding maps — must match the STKLM0 codebook (numeric codes only)
T_CLEAN_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 9: np.nan}
PT_MAP      = {0: np.nan, 2: 0, 3: 1, 4: 3, 5: 1, 6: 2, 9: np.nan}
PR_MAP      = {1: 0, 2: 1, 3: np.nan, 98: np.nan}
PN_MAP      = {0: 0, 1: 1, 9: np.nan, 98: np.nan}

STATIC_COLS = [
    "d_diaage", "d_spsa", "isup_gealson", "t_clean_ord",
    "isup_RP",  "pT_ord", "pR_bin",       "pRlenght",   "pN_bin",
]

# ---------------------------------------------------------------------------
# Robust scalar parsers — handle numeric codes AND common string formats
# ---------------------------------------------------------------------------

_UNKNOWN_STRINGS = {"", "nan", "none", "na", "n/a", "unknown", "unk", "missing", ".", "-"}


def _to_float(val) -> float:
    """Convert a scalar to float, handling European comma decimals and </>  prefixes."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip()
    if s.lower() in _UNKNOWN_STRINGS:
        return np.nan
    # Strip leading comparison operators: "<0.01" → "0.01", ">200" → "200"
    s = s.lstrip("<>≤≥~").strip()
    # European decimal comma: "3,5" → "3.5"  (only when no other comma present)
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_pt_stage(val) -> float:
    """
    Parse pathological T-stage to ordinal {0=pT2, 1=pT3a, 2=pT3b/c, 3=pT4}.

    Accepts:
      - Numeric STKLM0 codes: 0, 2, 3, 4, 5, 6, 9
      - Strings: 'pT2', 'pT2a', 'pT2b', 'pT2c', 'T2', 'pT3', 'pT3a', 'pT3b',
                 'pT3c', 'T3a', 'T3b', 'pT4', 'T4', '2', '3a', '3b' …
    """
    # Try numeric codebook first
    num = _to_float(val)
    if not np.isnan(num):
        code = int(num)
        if code in PT_MAP:
            return PT_MAP[code]
        # Codes not in the map are unexpected → NaN
        return np.nan
    # String parsing
    s = str(val).strip().upper().replace(" ", "").replace("-", "").replace("_", "")
    if s in _UNKNOWN_STRINGS or s in {"0", "9", "98"}:
        return np.nan
    # Strip leading "PT" or "T"
    for prefix in ("PT", "T"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    # Match stage
    if s.startswith("2"):          # pT2, pT2a, pT2b, pT2c → 0
        return 0.0
    if s in ("3", "3A", "3C"):     # pT3 / pT3a / pT3c (no consensus → treat as 3a) → 1
        return 1.0
    if s in ("3B",):               # pT3b → 2
        return 2.0
    if s.startswith("4"):          # pT4 → 3
        return 3.0
    return np.nan


def _parse_t_clean(val) -> float:
    """
    Parse clinical T-stage to ordinal {0=T1, 1=T2a, 2=T2b/c, 3=T3+}.

    Accepts numeric codes {0,1,2,3,9} or strings 'T1', 'T1c', 'T2a', 'T2b',
    'T2c', 'T2', 'T3', 'T3a', 'T3b', '1', '2a', '3' …
    """
    num = _to_float(val)
    if not np.isnan(num):
        code = int(num)
        return T_CLEAN_MAP.get(code, np.nan)
    s = str(val).strip().upper().replace(" ", "").replace("-", "")
    if s in _UNKNOWN_STRINGS or s == "9":
        return np.nan
    for prefix in ("CT", "T"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.startswith("1"):   return 0.0   # T1, T1a, T1b, T1c
    if s in ("2A",):        return 1.0
    if s in ("2B", "2C", "2"):  return 2.0
    if s.startswith("3") or s.startswith("4"):  return 3.0
    return np.nan


def _parse_margin(val) -> float:
    """
    Parse surgical margin status to binary {0=negative, 1=positive}.

    Accepts numeric codes {1,2,3,98} or strings 'R0', 'R1', 'neg', 'pos',
    'negative', 'positive', 'free', '0', '1' …
    """
    num = _to_float(val)
    if not np.isnan(num):
        code = int(num)
        return PR_MAP.get(code, np.nan)
    s = str(val).strip().upper().replace(" ", "")
    if s in _UNKNOWN_STRINGS or s in {"3", "98"}:
        return np.nan
    if s in ("R0", "0", "NEG", "NEGATIVE", "FREE", "CLEAR", "N"):
        return 0.0
    if s in ("R1", "R2", "1", "2", "POS", "POSITIVE", "INVOLVED", "Y"):
        return 1.0
    return np.nan


def _parse_node(val) -> float:
    """
    Parse lymph node status to binary {0=N0, 1=N1}.

    Accepts numeric codes {0,1,9,98} or strings 'N0', 'N1', 'neg', 'pos' …
    """
    num = _to_float(val)
    if not np.isnan(num):
        code = int(num)
        return PN_MAP.get(code, np.nan)
    s = str(val).strip().upper().replace(" ", "")
    if s in _UNKNOWN_STRINGS or s in {"9", "98"}:
        return np.nan
    if s in ("N0", "0", "NEG", "NEGATIVE", "N"):
        return 0.0
    if s in ("N1", "N2", "N3", "1", "2", "3", "POS", "POSITIVE", "Y"):
        return 1.0
    return np.nan


def _parse_isup(val) -> float:
    """Parse ISUP grade to int {1-5}; handles 'Grade 3', '3+4', '7' (sum → grade)."""
    num = _to_float(val)
    if not np.isnan(num):
        v = int(num)
        # Gleason sum (6-10) → ISUP grade
        if 6 <= v <= 10:
            return float({6: 1, 7: 2, 8: 3, 9: 4, 10: 5}.get(v, np.nan))
        if 1 <= v <= 5:
            return float(v)
        return np.nan
    s = str(val).strip().upper().replace(" ", "")
    # "3+4", "4+3" → sum → ISUP
    if "+" in s:
        parts = s.replace("GRADE", "").split("+")
        try:
            total = sum(int(p) for p in parts)
            return _parse_isup(total)
        except ValueError:
            pass
    # "GRADE3" / "GG3"
    for prefix in ("GRADE", "GG", "G"):
        if s.startswith(prefix):
            return _parse_isup(s[len(prefix):])
    return np.nan


def _apply_parser(series: pd.Series, fn) -> pd.Series:
    """Apply a scalar parser element-wise, returning a float Series."""
    return series.apply(fn).astype(float)


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def encode_stklm0_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply STKLM0 codebook encodings and produce a clean static feature DataFrame.
    Input df must have one row per patient, indexed by patient ID.

    Handles both numeric codebook values AND common string formats
    (e.g. 'pT3a', 'T2b', 'R0', 'N1', 'Grade 3', '<0.01', '3,5').

    Codebook-mapped unknowns become NaN and are later filled by split_and_impute
    (training) or fallback batch-median imputation (inference).
    """
    out = pd.DataFrame(index=df.index)

    # --- Continuous features ------------------------------------------------
    age = _apply_parser(df["d_diaage"], _to_float)
    out["d_diaage"] = age.where(age > 0).clip(20, 100)   # 0 → NaN → imputed

    out["d_spsa"]       = _apply_parser(df["d_spsa"],       _to_float).clip(0, 2000)
    out["isup_gealson"] = _apply_parser(df["isup_gealson"], _parse_isup).clip(1, 5)
    out["isup_RP"]      = _apply_parser(df["isup_RP"],      _parse_isup).clip(1, 5)

    # --- Staged / categorical features --------------------------------------
    out["t_clean_ord"] = _apply_parser(df["t_clean"], _parse_t_clean)
    out["pT_ord"]      = _apply_parser(df["pT"],      _parse_pt_stage)
    out["pR_bin"]      = _apply_parser(df["pR"],      _parse_margin)
    out["pN_bin"]      = _apply_parser(df["pN"],      _parse_node)

    # --- Margin length ------------------------------------------------------
    # If pR is available in the source column, try to use it for pRlenght logic
    pRlenght_raw = df.get("pRlenght", pd.Series(np.nan, index=df.index))
    pRlenght = _apply_parser(pRlenght_raw, _to_float).clip(0, 100)
    # Negative / unknown margins → margin length = 0 (not applicable)
    out["pRlenght"] = np.where(out["pR_bin"] == 0, 0.0, pRlenght)

    return out


# ---------------------------------------------------------------------------
# PSA wide → long conversion
# ---------------------------------------------------------------------------

def _parse_date_flexible(series: pd.Series) -> pd.Series:
    """
    Parse dates tolerating multiple formats:
    YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, DD.MM.YYYY, DD-MM-YYYY.
    Returns a datetime Series (NaT on failure).
    """
    # Try ISO first (most common, unambiguous)
    parsed = pd.to_datetime(series, errors="coerce", format="%Y-%m-%d")
    failed = parsed.isna() & series.notna() & (series.astype(str).str.strip() != "")
    if failed.any():
        # dayfirst=True covers DD/MM/YYYY, DD.MM.YYYY, DD-MM-YYYY
        fallback = pd.to_datetime(series[failed], errors="coerce", dayfirst=True)
        parsed = parsed.copy()
        parsed[failed] = fallback
    return parsed


def build_psa_long_stklm0(df: pd.DataFrame, t_max: float = T_MAX, n_psa: int = N_PSA) -> pd.DataFrame:
    """
    Convert PSA1...PSA{n_psa} + psadate1...psadate{n_psa} to long format.
    Surgery reference date: exp_date.
    Caps at t_max, enforces strictly monotone timestamps.

    Robust to:
    - String PSA values: '<0.01', '3,5' (European decimal), 'undetectable'
    - Mixed date formats: YYYY-MM-DD, DD/MM/YYYY, DD.MM.YYYY
    - Mismatched PSA / date column counts
    """
    # Only keep indices where BOTH PSA{i} and psadate{i} exist — prevents shape mismatch
    valid_idx = [i for i in range(1, n_psa + 1)
                 if f"PSA{i}" in df.columns and f"psadate{i}" in df.columns]
    psa_cols  = [f"PSA{i}"     for i in valid_idx]
    date_cols = [f"psadate{i}" for i in valid_idx]

    dos      = _parse_date_flexible(df["exp_date"])
    df_dates = df[date_cols].apply(_parse_date_flexible)
    df_days  = df_dates.subtract(dos, axis=0).apply(
        lambda col: col.dt.days if hasattr(col, "dt") else col.map(
            lambda x: x.days if pd.notna(x) else np.nan
        )
    ).astype("float64")   # nullable Int64 → plain float64 so np.isnan works

    # Parse PSA values robustly (handles '<0.01', '3,5', 'undetectable', etc.)
    df_psa = df[psa_cols].copy().apply(
        lambda col: col.apply(_to_float)
    ).astype("float64")

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
    df_s = df_s[~df_s.index.duplicated(keep="first")]  # guard against duplicate IDs
    df_s["label"] = df_outcome.loc[df_s.index, "label"]
    df_s["tte"]   = df_outcome.loc[df_s.index, "tte"]
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
    # Use plain Python lists to avoid PyArrow-backed Index / Series in sklearn.
    # Any pandas fancy-indexing with a numpy array on a PyArrow-backed structure
    # raises "only integer scalar arrays can be converted to a scalar index".
    unique_ids = list(df_long.index.unique())
    label_map  = df_long.groupby(level=0)["label"].first().to_dict()
    strat_all  = [int(label_map[uid]) for uid in unique_ids]
    total_test = val_frac + test_frac

    train_ids, tmp_ids = train_test_split(
        unique_ids, test_size=total_test, random_state=random_state,
        stratify=strat_all,
    )
    strat_tmp  = [int(label_map[uid]) for uid in tmp_ids]
    val_ids, test_ids = train_test_split(
        tmp_ids, test_size=test_frac / total_test, random_state=random_state,
        stratify=strat_tmp,
    )

    train = df_long.loc[train_ids].copy()
    val   = df_long.loc[val_ids].copy()
    test  = df_long.loc[test_ids].copy()

    train_first = train.groupby(level=0)[static_cols].first()
    # keep_empty_features=True prevents sklearn from dropping all-NaN columns during
    # transform (added in sklearn 1.1).  Older sklearn silently drops them, which would
    # return (n, 8) when static_cols has 9 entries, causing "index 8 out of bounds".
    try:
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        imp = SimpleImputer(strategy="median")
    imp.fit(train_first)

    # Which static_cols had a computable median (non-NaN statistic)?
    valid_stat_mask = ~np.isnan(imp.statistics_)

    for split in [train, val, test]:
        imputed = imp.transform(split[static_cols])  # may be (n, < len(static_cols))
        # Guard: if sklearn dropped all-NaN columns, expand back to full width with 0.
        if imputed.shape[1] < len(static_cols):
            full = np.zeros((imputed.shape[0], len(static_cols)), dtype=float)
            full[:, valid_stat_mask] = imputed
            imputed = full
        # Assign column-by-column to avoid pandas "Columns must be the same length as key"
        # error that fires when assigning a 2-D array or DataFrame to a multi-column loc slice.
        for i, col in enumerate(static_cols):
            split[col] = imputed[:, i]

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
