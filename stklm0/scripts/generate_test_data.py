#!/usr/bin/env python3
"""
Generate a fake STKLM0-schema CSV for testing the Streamlit app.

Schema notes
------------
- Has an 'id' column (app uses it as patient index).
- Raw codebook values for t_clean / pT / pR / pN — the app's
  encode_stklm0_features() maps these to ordinal/binary encodings.
- ~8% CSM events (crmort=1), ~7% other deaths (crmort=2, treated as censored).
- ~4% of patients have age=0 to exercise the imputation path.
- Some unknown codebook values (pT=9, pN=9, pR=98) to test NaN handling.
- Variable number of PSA observations (3–18 per patient, stored in PSA1…PSAn
  + psadate1…psdaten; missing slots are left blank / NaN in the CSV).

Run from repo root:
  python stklm0/scripts/generate_test_data.py           # 200 patients
  python stklm0/scripts/generate_test_data.py --n 50   # smaller batch
  python stklm0/scripts/generate_test_data.py --n 50 --out stklm0/data/mini_test.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import date, timedelta

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Raw codebook values (before encode_stklm0_features maps them)
T_CLEAN_VALUES = [0, 1, 2, 3, 9]       # 9 = unknown -> NaN after encoding
T_CLEAN_PROBS  = [0.05, 0.22, 0.30, 0.38, 0.05]

PT_VALUES      = [2, 3, 5, 6, 4, 9]    # 9 = unknown -> NaN
PT_PROBS       = [0.18, 0.12, 0.28, 0.20, 0.07, 0.15]

PR_VALUES      = [1, 2, 3, 98]         # 3/98 = unknown -> NaN
PR_PROBS       = [0.63, 0.27, 0.05, 0.05]

PN_VALUES      = [0, 1, 9, 98]         # 9/98 = unknown -> NaN
PN_PROBS       = [0.76, 0.10, 0.07, 0.07]


def generate(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(n):
        # Surgery date: 2008–2018
        year  = int(rng.integers(2008, 2019))
        month = int(rng.integers(1, 13))
        day   = int(rng.integers(1, 29))
        dos   = date(year, month, day)

        # Follow-up: 2–10 years (t_max=3650 in the app)
        fu_days = int(rng.integers(730, 3651))
        t_end   = dos + timedelta(days=fu_days)

        # Outcome: 85% censored, 8% CSM, 7% other death
        crmort = int(rng.choice([0, 1, 2], p=[0.85, 0.08, 0.07]))

        # Age: mostly 52–80; ~4% have age=0 (missing -> median imputed by app)
        if rng.random() < 0.04:
            d_diaage = 0
        else:
            d_diaage = int(np.clip(rng.normal(65, 7), 45, 85))

        # Pre-op PSA (log-normal, realistic)
        d_spsa = round(float(np.clip(rng.lognormal(1.6, 0.9), 0.5, 200)), 2)

        # Gleason groups (biopsy + RP)
        isup_gealson = int(rng.choice([1, 2, 3, 4, 5], p=[0.10, 0.28, 0.36, 0.16, 0.10]))
        isup_RP      = int(rng.choice([1, 2, 3, 4, 5], p=[0.08, 0.26, 0.38, 0.18, 0.10]))

        t_clean = int(rng.choice(T_CLEAN_VALUES, p=T_CLEAN_PROBS))
        pT_val  = int(rng.choice(PT_VALUES,      p=PT_PROBS))
        pR_val  = int(rng.choice(PR_VALUES,      p=PR_PROBS))
        pN_val  = int(rng.choice(PN_VALUES,      p=PN_PROBS))

        # Margin length: 0 if negative/unknown margins, else 1–20 mm
        pRlenght = round(float(rng.uniform(1, 20)), 1) if pR_val == 2 else 0.0

        # PSA observations (variable count, ~monthly)
        n_psa   = int(rng.integers(3, 19))
        max_day = max(31, min(fu_days - 1, 3640))
        if max_day > n_psa * 30:
            obs_days = sorted(rng.choice(range(30, max_day), size=n_psa, replace=False).tolist())
        else:
            obs_days = sorted(range(30, max(31, max_day), max(1, max_day // n_psa)))[:n_psa]

        # PSA trajectory: stable near zero unless BCR/CSM
        is_bcr  = (crmort == 1) or (rng.random() < 0.15)
        baseline = float(rng.uniform(0.01, 0.12))
        psa_vals = []
        for day in obs_days:
            if is_bcr:
                doubling = float(rng.uniform(540, 900))   # ~18–30 months
                psa = baseline * (2.0 ** (day / doubling))
            else:
                psa = baseline + float(rng.normal(0, 0.015))
            psa_vals.append(round(max(float(psa), 0.001), 3))

        row = {
            "id":           f"PT{i + 1:04d}",
            "exp_date":     dos.isoformat(),
            "t_end":        t_end.isoformat(),
            "crmort":       crmort,
            "d_diaage":     d_diaage,
            "d_spsa":       d_spsa,
            "isup_gealson": isup_gealson,
            "t_clean":      t_clean,
            "isup_RP":      isup_RP,
            "pT":           pT_val,
            "pR":           pR_val,
            "pRlenght":     pRlenght,
            "pN":           pN_val,
        }

        # Fill only the columns that actually have observations
        for j, (day, psa) in enumerate(zip(obs_days, psa_vals), start=1):
            row[f"PSA{j}"]     = psa
            row[f"psadate{j}"] = (dos + timedelta(days=int(day))).isoformat()

        rows.append(row)

    # Build DataFrame — missing PSA slots stay NaN automatically
    static_cols = [
        "id", "exp_date", "t_end", "crmort",
        "d_diaage", "d_spsa", "isup_gealson", "t_clean",
        "isup_RP", "pT", "pR", "pRlenght", "pN",
    ]
    df = pd.DataFrame(rows)
    psa_cols = sorted(
        [c for c in df.columns if c.startswith("PSA") and not c.startswith("psadate")],
        key=lambda x: int(x[3:])
    )
    date_cols = [f"psadate{c[3:]}" for c in psa_cols]
    return df[static_cols + psa_cols + date_cols]


def main():
    parser = argparse.ArgumentParser(description="Generate fake STKLM0 test data")
    parser.add_argument("--n",    type=int, default=200, help="Number of patients (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  default=os.path.join(_REPO_ROOT, "stklm0", "data", "test_patients.csv"))
    args = parser.parse_args()

    df = generate(n=args.n, seed=args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    n_csm   = int((df["crmort"] == 1).sum())
    n_other = int((df["crmort"] == 2).sum())
    n_age0  = int((df["d_diaage"] == 0).sum())
    n_unkn  = int(df["pT"].isin([9]).sum() + df["pN"].isin([9, 98]).sum() + df["pR"].isin([3, 98]).sum())
    n_psa_c = sum(1 for c in df.columns if c.startswith("PSA") and not c.startswith("psadate"))

    print(f"Written {len(df)} patients to {args.out}")
    print(f"  Columns      : {len(df.columns)}  (up to {n_psa_c} PSA observations each)")
    print(f"  Events (CSM) : {n_csm} / {len(df)} ({100*n_csm/len(df):.1f}%)")
    print(f"  Other deaths : {n_other}")
    print(f"  Age = 0      : {n_age0}  (-> median imputed by app)")
    print(f"  Unknown codes: {n_unkn} patient-features (pT/pR/pN = 9/98 -> NaN -> imputed)")

    preview = ["id", "exp_date", "d_diaage", "d_spsa", "isup_gealson",
               "t_clean", "isup_RP", "pT", "pR", "pN", "crmort", "PSA1", "psadate1"]
    print(df[[c for c in preview if c in df.columns]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
