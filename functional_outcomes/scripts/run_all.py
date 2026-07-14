#!/usr/bin/env python3
"""
End-to-end functional outcomes pipeline for EF (IIEF >= 17) and UC (ICIQ = 0).

Steps
-----
  1. Dataset preparation  (prepare_dataset_uri.py)
  2. Baseline evaluation  (CAPRA-S, MSKCC, CoxPH, RSF, DDH)
  3. BertPCa pipeline     (Boruta feature selection → training with YAML params)
  4. Comparison table     (merge baseline + BertPCa results, print & save)

Run from the repo root:

  python functional_outcomes/scripts/run_all.py
  python functional_outcomes/scripts/run_all.py --skip-baselines --force-train
"""

import os
import sys
import csv
import zipfile
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PYTHON    = sys.executable
_SCRIPTS   = os.path.join(_REPO_ROOT, "functional_outcomes", "scripts")
_DATA_DIR  = os.path.join(_REPO_ROOT, "functional_outcomes", "data")
_OUT_DIR   = os.path.join(_REPO_ROOT, "functional_outcomes", "outputs")

SEP  = "=" * 65
SEP2 = "-" * 65

EF_P_TIMES = [45, 90, 180]
EF_E_TIMES = [90, 180, 365]
UC_P_TIMES = [45, 90, 180]
UC_E_TIMES = [90, 180, 365]


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


def _baseline_csv(outcome: str) -> str:
    return os.path.join(_OUT_DIR, f"baseline_results_{outcome}.csv")


def _pipeline_cindex_csv(outcome: str) -> str:
    return os.path.join(_OUT_DIR, "results", f"{outcome}_uri", "pipeline", "c_index_table.csv")


def _pipeline_model(outcome: str) -> str:
    return os.path.join(_OUT_DIR, "models", f"pipeline_model_{outcome}.keras")


# ---------------------------------------------------------------------------
# Step 1 — Dataset preparation
# ---------------------------------------------------------------------------

def step_prepare(force: bool) -> None:
    ef_csv = os.path.join(_DATA_DIR, "uri_ef_train.csv")
    if not force and os.path.exists(ef_csv):
        print("\n[SKIP] Dataset preparation — CSVs already exist "
              "(use --force-prepare to regenerate)")
        return
    _run(
        "Step 1 — Prepare EF dataset from URI longitudinal IIEF data",
        [os.path.join(_SCRIPTS, "prepare_dataset_uri.py")],
    )


# ---------------------------------------------------------------------------
# Step 2 — Baselines (no BertPCa yet)
# ---------------------------------------------------------------------------

def step_baselines(force: bool) -> None:
    out_ef = _baseline_csv("ef")
    out_uc = _baseline_csv("uc")
    if not force and os.path.exists(out_ef) and os.path.exists(out_uc):
        print(f"\n[SKIP] Baselines — CSVs exist (use --force-baselines to re-run)")
        return
    _run(
        "Step 2 — EF + UC baselines (CAPRA-S, MSKCC, CoxPH, RSF, DDH)",
        [
            os.path.join(_SCRIPTS, "run_baselines.py"),
            "--outcome", "all",
        ],
        stop_on_failure=False,
    )


# ---------------------------------------------------------------------------
# Step 3 — BertPCa pipeline (Boruta → HPT → train → eval)
# ---------------------------------------------------------------------------

def step_bertpca_pipeline() -> None:
    pipeline_args = [
        os.path.join(_SCRIPTS, "run_pipeline.py"),
        "--outcome", "all",
    ]
    _run(
        "Step 3 — BertPCa pipeline: Boruta + train",
        pipeline_args,
    )


# ---------------------------------------------------------------------------
# Step 4 — Merge results and print comparison table
# ---------------------------------------------------------------------------

_OUTCOME_TIMES = {
    "ef": {"p": EF_P_TIMES, "e": EF_E_TIMES, "label": "EF (IIEF>=17)"},
    "uc": {"p": UC_P_TIMES, "e": UC_E_TIMES, "label": "UC (ICIQ=0)"},
}


def _read_pipeline_cindex(outcome: str) -> np.ndarray | None:
    """Read pipeline c-index CSV → (len(p_times), len(e_times)) array."""
    path = _pipeline_cindex_csv(outcome)
    e_times = _OUTCOME_TIMES[outcome]["e"]
    if not os.path.exists(path):
        print(f"  [WARNING] Pipeline c-index not found: {path}")
        return None
    df = pd.read_csv(path)
    e_cols = [f"e_time_{int(e)}" for e in e_times]
    available = [c for c in e_cols if c in df.columns]
    if not available:
        print(f"  [WARNING] Expected {e_cols}, found: {list(df.columns)}")
        return None
    if len(available) < len(e_cols):
        print(f"  [WARNING] Only {len(available)}/{len(e_cols)} e_time columns found")
    return df[available].values.astype(float)


def _bertpca_row(c_matrix: np.ndarray, outcome: str) -> dict:
    """Convert c-index matrix to a flat row matching the baseline CSV format."""
    p_times = _OUTCOME_TIMES[outcome]["p"]
    e_times = _OUTCOME_TIMES[outcome]["e"]
    row = {"method": "BertPCa (pipeline)"}
    vals = []
    for pi, p in enumerate(p_times):
        for ei, e in enumerate(e_times):
            col = f"p{int(p)}_e{int(e)}"
            v = float(c_matrix[pi, ei])
            row[col] = round(v, 6)
            if v != -1.0:  # exclude -1.0 sentinel (IPCW failure / e<=p cells)
                vals.append(v)
    row["mean"] = round(float(np.nanmean(vals)) if vals else np.nan, 6)
    return row


def _compare_outcome(outcome: str) -> None:
    times = _OUTCOME_TIMES[outcome]

    bl_path = _baseline_csv(outcome)
    if not os.path.exists(bl_path):
        print(f"  [WARNING] Baseline CSV not found: {bl_path}")
        df_bl = pd.DataFrame()
    else:
        df_bl = pd.read_csv(bl_path)

    c_matrix = _read_pipeline_cindex(outcome)
    if c_matrix is not None:
        bp_row = _bertpca_row(c_matrix, outcome)
        if not df_bl.empty:
            df_bl = df_bl[~df_bl["method"].str.contains("BertPCa", na=False)]
        df_bl = pd.concat([df_bl, pd.DataFrame([bp_row])], ignore_index=True)
    else:
        print(f"  BertPCa {outcome.upper()} results unavailable — showing baselines only.")

    if df_bl.empty:
        print(f"  No results to show for {outcome.upper()}.")
        return

    print(f"\n  {times['label']} — time-dependent C-index on test set")
    print(f"  Prediction times: {times['p']}d | Evaluation times: {times['e']}d\n")
    print(df_bl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    merged_path = os.path.join(_OUT_DIR, f"full_results_{outcome}.csv")
    df_bl.to_csv(merged_path, index=False)
    print(f"\n  Full results saved to {merged_path}")

    if "mean" in df_bl.columns:
        ranked = df_bl[["method", "mean"]].sort_values("mean", ascending=False)
        print(f"\n  {SEP2}")
        print(f"  Ranked by mean C-index ({outcome.upper()}):")
        for _, r in ranked.iterrows():
            marker = " <-- BertPCa" if "BertPCa" in str(r["method"]) else ""
            print(f"    {r['method']:<28}  {r['mean']:.3f}{marker}")
        print(f"  {SEP2}")


def step_compare() -> None:
    _hdr("Step 4 — Comparison tables (baselines + BertPCa)")
    for outcome in ("ef", "uc"):
        _compare_outcome(outcome)


# ---------------------------------------------------------------------------
# Step 5 — Bundle all results into a single zip
# ---------------------------------------------------------------------------

_BUNDLE_FILES = [
    # EF comparison tables
    ("outputs/full_results_ef.csv",                                  "full_results_ef.csv"),
    ("outputs/baseline_results_ef.csv",                              "baseline_results_ef.csv"),
    # EF pipeline internals
    ("outputs/boruta_ef_features.json",                              "boruta_ef_features.json"),
    ("outputs/hpt_best_ef.json",                                     "hpt_best_ef.json"),
    # EF BertPCa evaluation
    ("outputs/results/ef_uri/pipeline/c_index_table.csv",            "bertpca_ef_c_index_table.csv"),
    ("outputs/results/ef_uri/training_log.txt",                      "bertpca_ef_training_log.txt"),
    # UC comparison tables
    ("outputs/full_results_uc.csv",                                  "full_results_uc.csv"),
    ("outputs/baseline_results_uc.csv",                              "baseline_results_uc.csv"),
    # UC pipeline internals
    ("outputs/boruta_uc_features.json",                              "boruta_uc_features.json"),
    ("outputs/hpt_best_uc.json",                                     "hpt_best_uc.json"),
    # UC BertPCa evaluation
    ("outputs/results/uc_uri/pipeline/c_index_table.csv",            "bertpca_uc_c_index_table.csv"),
    ("outputs/results/uc_uri/training_log.txt",                      "bertpca_uc_training_log.txt"),
]


def step_bundle() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name  = f"ef_results_{timestamp}.zip"
    zip_path  = os.path.join(_OUT_DIR, zip_name)

    _hdr("Step 5 — Bundle results")

    added, missing = [], []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel, arcname in _BUNDLE_FILES:
            src = os.path.join(_REPO_ROOT, "functional_outcomes", rel)
            if os.path.exists(src):
                zf.write(src, arcname)
                added.append(arcname)
            else:
                missing.append(arcname)

    if added:
        print(f"\n  Bundled {len(added)} file(s) → {zip_path}")
        for f in added:
            print(f"    {f}")
    if missing:
        print(f"\n  Skipped (not found): {missing}")

    return zip_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full EF functional outcomes pipeline: data → baselines → BertPCa → comparison"
    )
    # Opt-out flags — use these to skip individual stages
    parser.add_argument("--skip-baselines",  action="store_true",
                        help="Skip baseline evaluation")
    parser.add_argument("--skip-pipeline",   action="store_true",
                        help="Skip BertPCa pipeline (Boruta+train)")
    parser.add_argument("--skip-prepare",    action="store_true",
                        help="Skip dataset preparation (reuse existing CSVs)")

    args = parser.parse_args()

    print(f"\n{SEP}")
    print(f"  BertPCa — EF (IIEF>=17) + UC (ICIQ=0) full pipeline")
    print(SEP)

    step_prepare(force=not args.skip_prepare)

    if not args.skip_baselines:
        step_baselines(force=True)
    else:
        print("\n[SKIP] Baselines (--skip-baselines)")

    if not args.skip_pipeline:
        step_bertpca_pipeline()
    else:
        print("\n[SKIP] BertPCa pipeline (--skip-pipeline)")

    step_compare()

    zip_path = step_bundle()

    print(f"\n{SEP}\n  Done (EF + UC).  Bundle: {zip_path}\n{SEP}\n")


if __name__ == "__main__":
    main()
