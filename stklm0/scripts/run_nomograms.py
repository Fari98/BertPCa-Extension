#!/usr/bin/env python3
"""
Compute CAPRA-S and MSKCC nomogram scores on STKLM0 data and evaluate
discrimination for CSM using IPCW time-dependent C-index.

Run from repo root:
    python stklm0/scripts/run_nomograms.py [--data stklm0/data/test_patients.csv]
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in [
    os.path.join(_REPO_ROOT, "bertpca", "src"),
    os.path.join(_REPO_ROOT, "bertpca"),
    os.path.join(_REPO_ROOT, "stklm0", "scripts"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prepare_stklm0 import (
    encode_stklm0_features, build_psa_long_stklm0,
    assemble_long_format, split_and_impute, STATIC_COLS,
    _parse_date_flexible,
)

T_MAX  = 3650.0
E_TIMES = [1825, 3650]   # 5y, 10y (1y rarely has events for CSM)


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------

def capras_score(df_s: pd.DataFrame) -> np.ndarray:
    """CAPRA-S from STKLM0 features. pT_ord proxies ECE (>=1) and SVI (>=2)."""
    psa  = pd.to_numeric(df_s.get("d_spsa",      pd.Series(np.nan, index=df_s.index)), errors="coerce").values
    isup = pd.to_numeric(df_s.get("isup_gealson", pd.Series(np.nan, index=df_s.index)), errors="coerce").values
    psm  = pd.to_numeric(df_s.get("pR_bin",       pd.Series(0.0,   index=df_s.index)), errors="coerce").fillna(0).values
    lni  = pd.to_numeric(df_s.get("pN_bin",       pd.Series(0.0,   index=df_s.index)), errors="coerce").fillna(0).values
    pT   = pd.to_numeric(df_s.get("pT_ord",       pd.Series(0.0,   index=df_s.index)), errors="coerce").fillna(0).values

    scores = (
        np.where(psa < 6, 0, np.where(psa <= 10, 1, 2))
        + np.where(isup <= 1, 0, np.where(isup == 2, 1, np.where(isup == 3, 2, 3)))
        + np.where(psm >= 1, 2, 0)
        + np.where(pT >= 1, 1, 0)   # ECE proxy
        + np.where(pT >= 2, 2, 0)   # SVI proxy
        + np.where(lni >= 1, 4, 0)
    ).astype(float)
    scores[np.isnan(psa) | np.isnan(isup)] = np.nan
    return scores


def mskcc_score(df_s: pd.DataFrame) -> np.ndarray:
    """MSKCC BCR nomogram (Stephenson 2005) approximated from STKLM0 features."""
    psa  = pd.to_numeric(df_s.get("d_spsa",      pd.Series(np.nan, index=df_s.index)), errors="coerce").values
    isup = pd.to_numeric(df_s.get("isup_gealson", pd.Series(np.nan, index=df_s.index)), errors="coerce").values
    psm  = pd.to_numeric(df_s.get("pR_bin",       pd.Series(0.0,   index=df_s.index)), errors="coerce").fillna(0).values
    pT   = pd.to_numeric(df_s.get("pT_ord",       pd.Series(0.0,   index=df_s.index)), errors="coerce").fillna(0).values

    gp = np.where(isup >= 3, 4.0, 3.0)
    gs = np.where(isup == 1, 3.0, np.where(isup == 2, 4.0, np.where(isup == 3, 3.0, 4.0)))
    gp[np.isnan(isup)] = np.nan
    gs[np.isnan(isup)] = np.nan
    ece = (pT >= 1).astype(float)
    svi = (pT >= 2).astype(float)
    pt4 = (pT >= 3).astype(float)

    lp = (0.508 * np.log(np.clip(psa, 1e-3, None) + 0.1)
          + 0.396 * (gp == 4) + 0.781 * (gp == 5)
          + 0.360 * (gs == 4) + 0.886 * (gs == 5)
          + 0.540 * np.where((ece >= 1) & (svi < 1), 1.0, 0.0)
          + 0.831 * svi + 1.021 * pt4
          + 0.386 * (psm >= 1)).astype(float)
    lp[np.isnan(psa) | np.isnan(gp)] = np.nan
    return 1.0 - 0.92 ** np.exp(lp)


def nomogram_c_index(scores, train_last, test_last):
    from bertpca.metrics import weighted_c_index as _wci
    T_tr = train_last["tte"].values
    Y_tr = train_last["label"].values.astype(float)
    T_te = test_last["tte"].values
    Y_te = test_last["label"].values.astype(float)
    valid = ~np.isnan(scores)
    results = {}
    for e in E_TIMES:
        if valid.sum() < 5:
            results[e] = np.nan
        else:
            results[e] = _wci(T_tr, Y_tr, scores[valid], T_te[valid], Y_te[valid], float(e))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(_REPO_ROOT, "stklm0", "data", "test_patients.csv"))
    args = parser.parse_args()

    print(f"Loading: {args.data}")
    df_raw = pd.read_csv(args.data)
    if "id" in df_raw.columns:
        df_raw = df_raw.set_index("id")

    print(f"  {len(df_raw):,} patients")

    # Encode and assemble long format (same pipeline as compare_app.py)
    df_work = df_raw.copy()
    df_work["label"] = (pd.to_numeric(df_work.get("crmort", 0), errors="coerce") == 1).astype(int)
    exp_date = _parse_date_flexible(df_work["exp_date"])
    t_end    = _parse_date_flexible(df_work.get("t_end", pd.Series(pd.NaT, index=df_work.index)))
    df_work["tte"] = (t_end - exp_date).dt.days.clip(lower=1, upper=T_MAX)

    df_static = encode_stklm0_features(df_work)
    psa_long  = build_psa_long_stklm0(df_work, t_max=T_MAX)
    df_long   = assemble_long_format(df_static, df_work[["label", "tte"]], psa_long, STATIC_COLS)
    train_df, val_df, test_df, _ = split_and_impute(df_long, STATIC_COLS)

    n_train = train_df.index.nunique()
    n_test  = test_df.index.nunique()
    n_ev_tr = int(train_df.groupby(level=0)["label"].first().sum())
    n_ev_te = int(test_df.groupby(level=0)["label"].first().sum())
    print(f"  Train: {n_train} patients ({n_ev_tr} CSM events)")
    print(f"  Test:  {n_test} patients ({n_ev_te} CSM events)")

    train_last = train_df.groupby(level=0).first()
    test_last  = test_df.groupby(level=0).first()

    # Score distributions
    df_s = test_last[STATIC_COLS]
    capras = capras_score(df_s)
    mskcc  = mskcc_score(df_s)

    print(f"\nCAPRA-S  — valid: {(~np.isnan(capras)).sum()}/{len(capras)} | "
          f"mean={np.nanmean(capras):.2f}, range=[{np.nanmin(capras):.0f}, {np.nanmax(capras):.0f}]")
    print(f"MSKCC    — valid: {(~np.isnan(mskcc)).sum()}/{len(mskcc)} | "
          f"mean={np.nanmean(mskcc):.3f}, range=[{np.nanmin(mskcc):.3f}, {np.nanmax(mskcc):.3f}]")

    # C-index
    capras_ci = nomogram_c_index(capras, train_last, test_last)
    mskcc_ci  = nomogram_c_index(mskcc,  train_last, test_last)

    print("\n─── IPCW C-index (test split) ───────────────────────")
    print(f"{'Model':<12}  {'e=5y':>8}  {'e=10y':>8}  {'Mean':>8}")
    print("─" * 44)
    for name, ci in [("CAPRA-S", capras_ci), ("MSKCC", mskcc_ci)]:
        c5  = ci.get(1825, np.nan)
        c10 = ci.get(3650, np.nan)
        mean = float(np.nanmean([v for v in ci.values() if v != -1]))
        def fmt(v):
            return f"{v:.4f}" if not np.isnan(v) and v != -1 else "   —  "
        print(f"{name:<12}  {fmt(c5):>8}  {fmt(c10):>8}  {fmt(mean):>8}")
    print("─" * 44)

    # Save
    out_dir = os.path.join(_REPO_ROOT, "stklm0", "outputs", "results")
    os.makedirs(out_dir, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for name, ci in [("CAPRA-S", capras_ci), ("MSKCC", mskcc_ci)]:
        c5  = ci.get(1825, np.nan)
        c10 = ci.get(3650, np.nan)
        mean = float(np.nanmean([v for v in ci.values() if not np.isnan(v) and v != -1]))
        rows.append({"Method": name, "Type": "Nomogram",
                     "Mean C-index": round(mean, 4),
                     "e=1y": np.nan, "e=5y": round(c5, 4), "e=10y": round(c10, 4)})
    out_path = os.path.join(out_dir, f"nomogram_comparison_{ts}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
