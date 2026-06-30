#!/usr/bin/env python3
"""
Generate a fake STKLM0-schema CSV for testing the Streamlit app.

Run from repo root:
  python stklm0/scripts/generate_test_data.py
  python stklm0/scripts/generate_test_data.py --n 50 --out stklm0/data/test_patients.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import date, timedelta

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def generate(n: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    surgery_dates = [
        date(int(rng.integers(2012, 2021)), int(rng.integers(1, 13)), int(rng.integers(1, 28)))
        for _ in range(n)
    ]

    rows = []
    for i, dos in enumerate(surgery_dates):
        row = {
            "id":           f"PT{i + 1:03d}",
            "exp_date":     dos.isoformat(),
            "d_diaage":     int(rng.integers(52, 78)),
            "d_spsa":       round(float(rng.uniform(1.5, 25.0)), 2),
            "isup_gealson": int(rng.integers(1, 6)),
            "t_clean":      int(rng.choice([0, 1, 2, 3])),
            "isup_RP":      int(rng.integers(1, 6)),
            "pT":           int(rng.choice([2, 3, 4, 5, 6])),
            "pR":           int(rng.choice([1, 1, 2])),    # mostly negative margins
            "pRlenght":     round(float(rng.uniform(0, 8)), 1),
            "pN":           int(rng.choice([0, 0, 0, 1])), # mostly node-negative
            "crmort":       int(rng.choice([0, 0, 0, 0, 1])),
            "t_end":        (dos + timedelta(days=int(rng.integers(365, 3650)))).isoformat(),
        }
        if row["pR"] == 1:
            row["pRlenght"] = 0.0

        # PSA time series: 3-12 observations at ~monthly intervals
        n_psa = int(rng.integers(3, 13))
        psa_val = float(rng.uniform(0.02, 0.5))
        for j in range(1, 136):
            if j <= n_psa:
                psa_date = dos + timedelta(days=int(j * 30 + rng.integers(-7, 8)))
                psa_val  = max(0.01, psa_val + float(rng.normal(0.02, 0.05)))
                row[f"PSA{j}"]     = round(psa_val, 3)
                row[f"psadate{j}"] = psa_date.isoformat()
            else:
                row[f"PSA{j}"]     = None
                row[f"psadate{j}"] = None
        rows.append(row)

    static_cols = [
        "id", "exp_date", "t_end", "crmort",
        "d_diaage", "d_spsa", "isup_gealson", "t_clean",
        "isup_RP", "pT", "pR", "pRlenght", "pN",
    ]
    psa_cols = [f"PSA{j}" for j in range(1, 136)] + [f"psadate{j}" for j in range(1, 136)]
    return pd.DataFrame(rows)[static_cols + psa_cols]


def main():
    parser = argparse.ArgumentParser(description="Generate fake STKLM0 test data")
    parser.add_argument("--n",    type=int, default=20, help="Number of patients (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  default=os.path.join(_REPO_ROOT, "stklm0", "data", "test_patients.csv"))
    args = parser.parse_args()

    df = generate(n=args.n, seed=args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    static_preview = ["id", "exp_date", "d_diaage", "d_spsa", "isup_gealson",
                      "t_clean", "isup_RP", "pT", "pR", "pRlenght", "pN", "crmort"]
    print(f"Written {len(df)} patients to {args.out}")
    print(df[static_preview].to_string(index=False))


if __name__ == "__main__":
    main()
