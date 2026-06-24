#!/usr/bin/env python3
"""
Full STKLM0 pipeline — runs all steps in order:
  1. Prepare Milan data with STKLM0-compatible features (BCR + CSM)
  2. Prepare STKLM0 data from CSV
  3. Step 1a: Train BertPCa on Milan BCR  (saves model + scaling params)
  4. Step 1b: Train BertPCa on Milan CSM  (saves model + scaling params)
  5. Step 2a: Evaluate Milan BCR model on STKLM0 test set
  6. Step 2b: Evaluate Milan CSM model on STKLM0 test set
  7. Step 3:  Train and evaluate BertPCa on STKLM0 only (within-dataset)

Steps 3/4 (training) and 5/6 (evaluation) are deliberately separate scripts
so they can be run at different times on different datasets.

Run from repo root:
  python stklm0/scripts/run_all.py --stklm0-input data/stklm0.csv

Skip already-completed steps:
  python stklm0/scripts/run_all.py --stklm0-input data/stklm0.csv \\
      --skip-prep --skip-train-milan
"""

import os
import sys
import subprocess
import argparse

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PYTHON    = sys.executable

STEP_SEP = "=" * 60
_TOTAL   = 7


def run_step(step: int, label: str, args: list, stop_on_failure: bool = True) -> bool:
    print(f"\n{STEP_SEP}")
    print(f"  Step {step}/{_TOTAL} — {label}")
    print(STEP_SEP)
    result = subprocess.run([_PYTHON] + args, cwd=_REPO_ROOT)
    if result.returncode != 0:
        print(f"\n[FAILED] {label} (exit code {result.returncode})")
        if stop_on_failure:
            sys.exit(result.returncode)
        return False
    print(f"\n[OK] {label}")
    return True


def skip_step(step: int, label: str, reason: str = "skipped"):
    print(f"\n[SKIP] Step {step}/{_TOTAL} — {label} ({reason})")


def main():
    parser = argparse.ArgumentParser(description="Run full STKLM0 BertPCa pipeline")
    parser.add_argument("--stklm0-input",      default=None,
                        help="Path to STKLM0 CSV file (required unless --skip-prep-stklm0)")
    parser.add_argument("--skip-prep-milan",   action="store_true",
                        help="Skip Milan data preparation")
    parser.add_argument("--skip-prep-stklm0",  action="store_true",
                        help="Skip STKLM0 data preparation")
    parser.add_argument("--skip-train-milan",  action="store_true",
                        help="Skip training on Milan (Steps 3 & 4)")
    parser.add_argument("--skip-eval-stklm0",  action="store_true",
                        help="Skip STKLM0 evaluation of Milan models (Steps 5 & 6)")
    parser.add_argument("--skip-stklm0-train", action="store_true",
                        help="Skip Step 7: train+evaluate on STKLM0 only")
    parser.add_argument("--milan-outcome",     choices=["bcr", "csm", "both"], default="both",
                        help="Which Milan outcome(s) to use (default: both)")
    args = parser.parse_args()

    if not args.skip_prep_stklm0 and args.stklm0_input is None:
        parser.error("--stklm0-input is required unless --skip-prep-stklm0 is set")

    outcomes = ["bcr", "csm"] if args.milan_outcome == "both" else [args.milan_outcome]

    step = 1

    # ---- Step 1: Prepare Milan data ----------------------------------------------
    if not args.skip_prep_milan:
        run_step(step, f"Prepare Milan data (outcome={args.milan_outcome})",
                 ["stklm0/scripts/prepare_milan.py", "--outcome", args.milan_outcome])
    else:
        skip_step(step, "Prepare Milan data")
    step += 1

    # ---- Step 2: Prepare STKLM0 data --------------------------------------------
    if not args.skip_prep_stklm0:
        run_step(step, f"Prepare STKLM0 data from {args.stklm0_input}",
                 ["stklm0/scripts/prepare_stklm0.py", "--input", args.stklm0_input])
    else:
        skip_step(step, "Prepare STKLM0 data")
    step += 1

    # ---- Steps 3 & 4: Train on Milan --------------------------------------------
    for outcome in outcomes:
        if not args.skip_train_milan:
            run_step(step, f"Train BertPCa on Milan {outcome.upper()}",
                     ["stklm0/scripts/train_milan.py", "--outcome", outcome])
        else:
            skip_step(step, f"Train BertPCa on Milan {outcome.upper()}")
        step += 1

    # Pad step counter if only one outcome selected
    if len(outcomes) == 1:
        step += 1

    # ---- Steps 5 & 6: Evaluate Milan models on STKLM0 ---------------------------
    for outcome in outcomes:
        if not args.skip_eval_stklm0:
            run_step(step, f"Evaluate Milan {outcome.upper()} model on STKLM0",
                     ["stklm0/scripts/eval_stklm0.py", "--outcome", outcome])
        else:
            skip_step(step, f"Evaluate Milan {outcome.upper()} on STKLM0")
        step += 1

    if len(outcomes) == 1:
        step += 1

    # ---- Step 7: Train + evaluate on STKLM0 ------------------------------------
    if not args.skip_stklm0_train:
        run_step(step, "Train and evaluate BertPCa on STKLM0 (within-dataset)",
                 ["stklm0/scripts/train_eval_stklm0.py"])
    else:
        skip_step(step, "Train+evaluate on STKLM0")

    print(f"\n{STEP_SEP}")
    print("  Pipeline complete.")
    print(STEP_SEP)


if __name__ == "__main__":
    main()
