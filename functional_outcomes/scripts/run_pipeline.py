#!/usr/bin/env python3
"""
End-to-end BertPCa pipeline for functional outcomes (EF and UC).

  Stage 1: Feature-expanded dataset preparation (runs prepare_dataset_uri.py)
  Stage 2: Boruta feature selection (always re-run — no cache)
  Stage 3: Full training + C-index evaluation (uses YAML hyperparameters)

Usage (from repo root):
  python functional_outcomes/scripts/run_pipeline.py
  python functional_outcomes/scripts/run_pipeline.py --outcome uc
  python functional_outcomes/scripts/run_pipeline.py --outcome ef --force-train
  python functional_outcomes/scripts/run_pipeline.py --force   # re-run everything
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bertpca", "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bertpca"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "functional_outcomes", "scripts"))

from config.load_config import load_yaml_config
from bertpca import load_and_preprocess_data, set_seeds
from train_functional import train_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_DIR = os.path.join(_REPO_ROOT, "functional_outcomes", "config")
_CONFIG_MAP = {
    "ef": os.path.join(_CONFIG_DIR, "config_ef_uri.yaml"),
    "uc": os.path.join(_CONFIG_DIR, "config_uc_uri.yaml"),
}
_DATA_DIR   = os.path.join(_REPO_ROOT, "functional_outcomes", "data")
_OUTPUT_DIR = os.path.join(_REPO_ROOT, "functional_outcomes", "outputs")

_PSA_DERIVED = ["psa_nadir", "time_to_nadir", "psa_at_last_obs", "psa_slope", "n_psa_obs"]
_EXTRA_STATIC = ["QoL_pre", "drs_max"]

_EF_CANDIDATES = [
    "nerve_sparing", "IIEF_EFdomain_pre", "age", "tpsa", "bmi",
    "pathgg_group", "ece_bin", "svi_bin", "psm", "lni_bin",
    "neo_adjHT", "pstage",
] + _EXTRA_STATIC + _PSA_DERIVED

_UC_CANDIDATES = [
    "nerve_sparing", "IPSS_pre", "age", "tpsa", "bmi",
    "pathgg_group", "ece_bin", "svi_bin", "psm",
    "prostate_vol", "operative_time",
] + _EXTRA_STATIC + _PSA_DERIVED

_CANDIDATES = {"ef": _EF_CANDIDATES, "uc": _UC_CANDIDATES}

SEP = "=" * 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(outcome: str, static_features: list):
    """Load YAML config and patch with pipeline-specific feature list."""
    config = load_yaml_config(_CONFIG_MAP[outcome])
    config.TRAIN_PATH  = os.path.join(_REPO_ROOT, config.TRAIN_PATH)
    config.VAL_PATH    = os.path.join(_REPO_ROOT, config.VAL_PATH)
    config.TEST_PATH   = os.path.join(_REPO_ROOT, config.TEST_PATH)
    config.RESULTS_DIR = os.path.join(_REPO_ROOT, config.RESULTS_DIR)
    config.MODEL_DIR   = os.path.join(_REPO_ROOT, config.MODEL_DIR)
    config.STATIC_FEATURES = list(static_features)
    return config


# ---------------------------------------------------------------------------
# Stage 1 — Dataset preparation
# ---------------------------------------------------------------------------

def stage_1_prepare(force: bool) -> None:
    expected = [
        os.path.join(_DATA_DIR, f)
        for f in ("uri_ef_train.csv", "uri_ef_val.csv", "uri_ef_test.csv",
                  "uri_uc_train.csv", "uri_uc_val.csv", "uri_uc_test.csv")
    ]
    if not force and all(os.path.exists(p) for p in expected):
        print("[Stage 1] CSVs already exist — skipping (use --force-prepare to regenerate).")
        return

    print("[Stage 1] Running prepare_dataset_uri.py ...")
    script = os.path.join(_REPO_ROOT, "functional_outcomes", "scripts", "prepare_dataset_uri.py")
    result = subprocess.run([sys.executable, script], cwd=_REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"prepare_dataset_uri.py exited with code {result.returncode}")
    print("[Stage 1] Done.")


# ---------------------------------------------------------------------------
# Stage 2 — Boruta feature selection (always re-run)
# ---------------------------------------------------------------------------

def _boruta_select(train_csv: str, candidates: list,
                   random_state: int, max_iter: int,
                   include_tentative: bool) -> tuple:
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer

    df = pd.read_csv(train_csv).set_index("id")

    available = [f for f in candidates if f in df.columns]
    missing = set(candidates) - set(available)
    if missing:
        print(f"  WARNING: features not found in CSV (will skip): {sorted(missing)}")

    pt = df.groupby(level=0)[available + ["label"]].first().dropna(subset=["label"])
    X = pt[available].values.astype(float)
    y = pt["label"].values.astype(int)

    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)

    rf = RandomForestClassifier(
        n_jobs=-1,
        class_weight="balanced",
        max_depth=5,
        random_state=random_state,
    )
    selector = BorutaPy(
        rf,
        n_estimators="auto",
        max_iter=max_iter,
        random_state=random_state,
        verbose=1,
    )
    selector.fit(X, y)

    confirmed = [f for f, s in zip(available, selector.support_)      if s]
    tentative = [f for f, s in zip(available, selector.support_weak_) if s]

    if len(confirmed) < 4:
        print(f"  WARNING: Boruta confirmed only {len(confirmed)} feature(s) — "
              "falling back to all candidates (need ≥4 for transformer).")
        confirmed = list(available)
        tentative = []

    selected = confirmed + [f for f in tentative if f not in confirmed] \
        if include_tentative else confirmed
    return confirmed, tentative, selected


def stage_2_boruta(outcome: str, random_state: int, max_iter: int,
                   include_tentative: bool) -> list:
    out_path = os.path.join(_OUTPUT_DIR, f"boruta_{outcome}_features.json")
    train_csv  = os.path.join(_DATA_DIR, f"uri_{outcome}_train.csv")
    candidates = _CANDIDATES[outcome]
    print(f"[Stage 2/{outcome.upper()}] Running Boruta on {len(candidates)} candidates ...")

    confirmed, tentative, selected = _boruta_select(
        train_csv, candidates, random_state, max_iter, include_tentative
    )
    print(f"  Confirmed  ({len(confirmed)}): {confirmed}")
    print(f"  Tentative  ({len(tentative)}): {tentative}")
    print(f"  Selected   ({len(selected)}): {selected}")

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "outcome":           outcome,
            "candidates":        candidates,
            "confirmed":         confirmed,
            "tentative":         tentative,
            "include_tentative": include_tentative,
            "selected":          selected,
        }, f, indent=2)
    print(f"  Saved to {out_path}")
    return selected


# ---------------------------------------------------------------------------
# Stage 3 — Full training and evaluation
# ---------------------------------------------------------------------------

def stage_3_train(outcome: str, selected_features: list) -> None:
    os.makedirs(os.path.join(_OUTPUT_DIR, "models"), exist_ok=True)
    model_path = os.path.join(_OUTPUT_DIR, "models", f"pipeline_model_{outcome}.keras")

    config = _make_config(outcome, selected_features)

    config.RESULTS_DIR = os.path.join(config.RESULTS_DIR, "pipeline")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR,   exist_ok=True)

    print(f"[Stage 3/{outcome.upper()}] Training with {len(selected_features)} features ...")
    print(f"  Static:    {selected_features}")
    print(f"  Model cfg: {config.MODEL_CONFIG}")
    print(f"  Batch sz:  {config.BATCH_SIZE}")

    train_model(config, output_path=model_path)
    print(f"[Stage 3/{outcome.upper()}] Model saved to {model_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BertPCa pipeline: dataset → Boruta → training (no Optuna)"
    )
    parser.add_argument("--outcome", choices=["ef", "uc", "all"], default="ef")
    parser.add_argument("--force",            action="store_true", help="Re-run all stages")
    parser.add_argument("--force-prepare",    action="store_true")
    parser.add_argument("--include-tentative", action="store_true",
                        help="Include Boruta tentative features alongside confirmed ones")
    parser.add_argument("--boruta-max-iter",  type=int, default=100)
    parser.add_argument("--random-state",     type=int, default=42)
    args = parser.parse_args()

    force_prepare = args.force_prepare or args.force
    outcomes = ["ef", "uc"] if args.outcome == "all" else [args.outcome]

    print(f"\n{SEP}\n  STAGE 1 — Dataset Preparation\n{SEP}")
    stage_1_prepare(force=force_prepare)

    for outcome in outcomes:
        print(f"\n{SEP}\n  OUTCOME: {outcome.upper()}\n{SEP}")

        print(f"\n--- Stage 2: Boruta Feature Selection ({outcome.upper()}) ---")
        selected = stage_2_boruta(
            outcome,
            random_state=args.random_state,
            max_iter=args.boruta_max_iter,
            include_tentative=args.include_tentative,
        )

        print(f"\n--- Stage 3: Full Training ({outcome.upper()}) ---")
        stage_3_train(outcome, selected)

    print(f"\n{SEP}")
    print(f"  Pipeline complete: {', '.join(o.upper() for o in outcomes)}")
    print(SEP)


if __name__ == "__main__":
    main()
