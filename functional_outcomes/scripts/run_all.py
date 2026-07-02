#!/usr/bin/env python3
"""
End-to-end functional outcomes pipeline for EF (IIEF >= 26).

Steps
-----
  1. Dataset preparation  (prepare_dataset.py)
  2. Baseline evaluation  (CAPRA-S, MSKCC, CoxPH, RSF, DDH — no BertPCa yet)
  3. BertPCa pipeline     (Boruta feature selection → HPT → full training)
  4. Comparison table     (merge baseline + BertPCa results, print & save)

Each step checks for its output and skips if already done unless a --force
flag overrides. Run from the repo root:

  python functional_outcomes/scripts/run_all.py
  python functional_outcomes/scripts/run_all.py --n-trials 100
  python functional_outcomes/scripts/run_all.py --force
  python functional_outcomes/scripts/run_all.py --skip-baselines --force-train
"""

import os
import sys
import csv
import argparse
import subprocess
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PYTHON    = sys.executable
_SCRIPTS   = os.path.join(_REPO_ROOT, "functional_outcomes", "scripts")
_DATA_DIR  = os.path.join(_REPO_ROOT, "functional_outcomes", "data")
_OUT_DIR   = os.path.join(_REPO_ROOT, "functional_outcomes", "outputs")

SEP  = "=" * 65
SEP2 = "-" * 65

EF_P_TIMES = [14, 30, 60]
EF_E_TIMES = [60, 120, 180]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hdr(msg: str) -> None:
    print(f"\n{SEP}\n  {msg}\n{SEP}")


def _run(label: str, args: list, stop_on_failure: bool = True) -> bool:
    _hdr(label)
    rc = subprocess.run([_PYTHON] + args, cwd=_REPO_ROOT).returncode
    if rc != 0:
        print(f"\n[FAILED] {label} (exit {rc})")
        if stop_on_failure:
            sys.exit(rc)
        return False
    print(f"\n[OK] {label}")
    return True


def _baseline_csv() -> str:
    return os.path.join(_OUT_DIR, "baseline_results_ef.csv")


def _pipeline_cindex_csv() -> str:
    return os.path.join(_OUT_DIR, "results", "ef", "pipeline", "c_index_table.csv")


def _pipeline_model() -> str:
    return os.path.join(_OUT_DIR, "models", "pipeline_model_ef.keras")


# ---------------------------------------------------------------------------
# Step 1 — Dataset preparation
# ---------------------------------------------------------------------------

def step_prepare(force: bool) -> None:
    ef_csv = os.path.join(_DATA_DIR, "ef_train.csv")
    if not force and os.path.exists(ef_csv):
        print("\n[SKIP] Dataset preparation — CSVs already exist "
              "(use --force-prepare to regenerate)")
        return
    _run(
        "Step 1 — Prepare EF dataset from Milan RData (IIEF>=26, PSA>=3 obs)",
        [os.path.join(_SCRIPTS, "prepare_dataset.py")],
    )


# ---------------------------------------------------------------------------
# Step 2 — Baselines (no BertPCa yet)
# ---------------------------------------------------------------------------

def step_baselines(force: bool) -> None:
    out = _baseline_csv()
    if not force and os.path.exists(out):
        print(f"\n[SKIP] Baselines — {os.path.basename(out)} exists "
              "(use --force-baselines to re-run)")
        return
    _run(
        "Step 2 — EF baselines (CAPRA-S, MSKCC, CoxPH, RSF, DDH)",
        [
            os.path.join(_SCRIPTS, "run_baselines.py"),
            "--outcome", "ef",
        ],
        stop_on_failure=False,
    )


# ---------------------------------------------------------------------------
# Step 3 — BertPCa pipeline (Boruta → HPT → train → eval)
# ---------------------------------------------------------------------------

def step_bertpca_pipeline(n_trials: int, storage: str,
                           force_boruta: bool, force_hpt: bool,
                           force_train: bool) -> None:
    model_path = _pipeline_model()
    cindex_path = _pipeline_cindex_csv()

    all_done = os.path.exists(model_path) and os.path.exists(cindex_path)
    if all_done and not any([force_boruta, force_hpt, force_train]):
        print(f"\n[SKIP] BertPCa pipeline — model and c-index table exist "
              "(use --force-train to retrain)")
        return

    pipeline_args = [
        os.path.join(_SCRIPTS, "run_pipeline.py"),
        "--outcome", "ef",
        "--n-trials", str(n_trials),
    ]
    if storage:
        pipeline_args += ["--storage", storage]
    if force_boruta:
        pipeline_args.append("--force-boruta")
    if force_hpt:
        pipeline_args.append("--force-hpt")
    if force_train:
        pipeline_args.append("--force-train")

    _run(
        f"Step 3 — BertPCa pipeline: Boruta + HPT ({n_trials} trials) + train",
        pipeline_args,
    )


# ---------------------------------------------------------------------------
# Step 4 — Merge results and print comparison table
# ---------------------------------------------------------------------------

def _read_pipeline_cindex() -> np.ndarray | None:
    """Read pipeline c-index CSV → (len(p_times), len(e_times)) array."""
    path = _pipeline_cindex_csv()
    if not os.path.exists(path):
        print(f"  [WARNING] Pipeline c-index not found: {path}")
        return None
    df = pd.read_csv(path)
    e_cols = [f"e_time_{int(e)}" for e in EF_E_TIMES]
    available = [c for c in e_cols if c in df.columns]
    if not available:
        # Report what columns are actually present to aid debugging
        print(f"  [WARNING] Expected {e_cols}, found: {list(df.columns)}")
        return None
    if len(available) < len(e_cols):
        print(f"  [WARNING] Only {len(available)}/{len(e_cols)} e_time columns found")
    matrix = df[available].values.astype(float)
    return matrix


def _bertpca_row(c_matrix: np.ndarray) -> dict:
    """Convert c-index matrix to a flat row matching the baseline CSV format."""
    row = {"method": "BertPCa (pipeline)"}
    idx = 0
    vals = []
    for p in EF_P_TIMES:
        for e in EF_E_TIMES:
            col = f"p{int(p)}_e{int(e)}"
            v = float(c_matrix[EF_P_TIMES.index(p), EF_E_TIMES.index(e)])
            row[col] = round(v, 6)
            vals.append(v)
            idx += 1
    row["mean"] = round(float(np.nanmean(vals)), 6)
    return row


def step_compare() -> None:
    _hdr("Step 4 — Comparison table (baselines + BertPCa)")

    # Load baseline results
    bl_path = _baseline_csv()
    if not os.path.exists(bl_path):
        print(f"  [WARNING] Baseline CSV not found: {bl_path}")
        df_bl = pd.DataFrame()
    else:
        df_bl = pd.read_csv(bl_path)

    # Load BertPCa pipeline c-index
    c_matrix = _read_pipeline_cindex()

    if c_matrix is not None:
        bp_row = _bertpca_row(c_matrix)
        # Remove any previous BertPCa rows and append the new one
        if not df_bl.empty:
            df_bl = df_bl[~df_bl["method"].str.contains("BertPCa", na=False)]
        df_bl = pd.concat([df_bl, pd.DataFrame([bp_row])], ignore_index=True)
    else:
        print("  BertPCa results unavailable — showing baselines only.")

    if df_bl.empty:
        print("  No results to show.")
        return

    # Pretty-print
    print(f"\n  EF (IIEF>=26) — time-dependent C-index on test set")
    print(f"  Prediction times: {EF_P_TIMES}d | Evaluation times: {EF_E_TIMES}d\n")
    print(df_bl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Save merged table
    merged_path = os.path.join(_OUT_DIR, "full_results_ef.csv")
    df_bl.to_csv(merged_path, index=False)
    print(f"\n  Full results saved to {merged_path}")

    # Ranked summary
    if "mean" in df_bl.columns:
        ranked = df_bl[["method", "mean"]].sort_values("mean", ascending=False)
        print(f"\n  {SEP2}")
        print(f"  Ranked by mean C-index:")
        for _, r in ranked.iterrows():
            marker = " <-- BertPCa" if "BertPCa" in str(r["method"]) else ""
            print(f"    {r['method']:<28}  {r['mean']:.3f}{marker}")
        print(f"  {SEP2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full EF functional outcomes pipeline: data → baselines → BertPCa → comparison"
    )
    parser.add_argument("--n-trials",         type=int, default=50,
                        help="Optuna HPT trials for BertPCa (default: 50)")
    parser.add_argument("--storage",          type=str, default=None,
                        help="Optuna storage URL for resumable HPT "
                             "(e.g. sqlite:///functional_outcomes/outputs/hpt.db)")

    # Skip flags
    parser.add_argument("--skip-baselines",   action="store_true",
                        help="Skip baseline evaluation (useful if already run)")
    parser.add_argument("--skip-pipeline",    action="store_true",
                        help="Skip BertPCa pipeline (Boruta+HPT+train)")

    # Force flags — individual stages
    parser.add_argument("--force",            action="store_true",
                        help="Re-run all stages from scratch")
    parser.add_argument("--force-prepare",    action="store_true")
    parser.add_argument("--force-baselines",  action="store_true")
    parser.add_argument("--force-boruta",     action="store_true")
    parser.add_argument("--force-hpt",        action="store_true")
    parser.add_argument("--force-train",      action="store_true")

    args = parser.parse_args()
    fa = args.force

    print(f"\n{SEP}")
    print(f"  BertPCa — Erectile Function (IIEF>=26) full pipeline")
    print(f"  n_trials={args.n_trials}  storage={args.storage or 'in-memory'}")
    print(SEP)

    step_prepare(force=args.force_prepare or fa)

    if not args.skip_baselines:
        step_baselines(force=args.force_baselines or fa)
    else:
        print("\n[SKIP] Baselines (--skip-baselines)")

    if not args.skip_pipeline:
        step_bertpca_pipeline(
            n_trials=args.n_trials,
            storage=args.storage,
            force_boruta=args.force_boruta or fa,
            force_hpt=args.force_hpt    or fa,
            force_train=args.force_train or fa,
        )
    else:
        print("\n[SKIP] BertPCa pipeline (--skip-pipeline)")

    step_compare()

    print(f"\n{SEP}\n  Done.\n{SEP}\n")


if __name__ == "__main__":
    main()
