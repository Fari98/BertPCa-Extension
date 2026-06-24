#!/usr/bin/env python3
"""
Full functional outcomes pipeline — runs all steps in order:
  1. Prepare EF and UC datasets from Milan RData
  2. Train BertPCa on EF
  3. Train BertPCa on UC
  4. Run all baselines (CAPRA-S, MSKCC, CoxPH, RSF, DDH) with BertPCa comparison

Run from repo root:
  python functional_outcomes/scripts/run_all.py
"""

import os
import sys
import subprocess
import argparse

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PYTHON    = sys.executable

STEP_SEP = "=" * 60


def run_step(label: str, args: list, stop_on_failure: bool = True) -> bool:
    print(f"\n{STEP_SEP}")
    print(f"  {label}")
    print(STEP_SEP)
    result = subprocess.run([_PYTHON] + args, cwd=_REPO_ROOT)
    if result.returncode != 0:
        print(f"\n[FAILED] {label} (exit code {result.returncode})")
        if stop_on_failure:
            sys.exit(result.returncode)
        return False
    print(f"\n[OK] {label}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full functional outcomes pipeline")
    parser.add_argument("--skip-prep",      action="store_true",
                        help="Skip dataset preparation (if CSVs already exist)")
    parser.add_argument("--skip-train",     action="store_true",
                        help="Skip BertPCa training (if models already saved)")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline evaluation")
    parser.add_argument("--outcome",        choices=["ef", "uc", "all"], default="all",
                        help="Which outcome(s) to train/evaluate (default: all)")
    args = parser.parse_args()

    outcomes = ["ef", "uc"] if args.outcome == "all" else [args.outcome]

    # ---- Step 1: Prepare datasets ------------------------------------------------
    if not args.skip_prep:
        run_step(
            "Step 1/4 — Prepare EF and UC datasets from Milan RData",
            ["functional_outcomes/scripts/prepare_dataset.py"],
        )
    else:
        print("\n[SKIP] Dataset preparation")

    # ---- Step 2 & 3: Train BertPCa -----------------------------------------------
    if not args.skip_train:
        for i, outcome in enumerate(outcomes, start=2):
            ef_model_path = f"functional_outcomes/outputs/models/best_model_{outcome}.keras"
            run_step(
                f"Step {i}/{2 + len(outcomes)} — Train BertPCa on {outcome.upper()}",
                [
                    "functional_outcomes/scripts/train_functional.py",
                    "--outcome", outcome,
                    "--output", ef_model_path,
                ],
            )
    else:
        print("\n[SKIP] BertPCa training")

    # ---- Step 4: Baselines -------------------------------------------------------
    if not args.skip_baselines:
        baseline_args = [
            "functional_outcomes/scripts/run_baselines.py",
            "--outcome", args.outcome,
        ]
        for outcome in outcomes:
            model_path = os.path.join(
                _REPO_ROOT, "functional_outcomes", "outputs", "models",
                f"best_model_{outcome}.keras",
            )
            if os.path.exists(model_path):
                baseline_args += ["--bertpca-model", model_path]
                break  # run_baselines.py handles outcome-specific model lookup internally

        run_step(
            f"Step {2 + len(outcomes) + 1}/{2 + len(outcomes) + 1} — Run baselines "
            f"({args.outcome.upper()})",
            baseline_args,
            stop_on_failure=False,  # baselines are optional — don't abort on failure
        )
    else:
        print("\n[SKIP] Baseline evaluation")

    print(f"\n{STEP_SEP}")
    print("  Pipeline complete.")
    print(STEP_SEP)


if __name__ == "__main__":
    main()
