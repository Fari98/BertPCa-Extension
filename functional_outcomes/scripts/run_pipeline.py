#!/usr/bin/env python3
"""
End-to-end BertPCa pipeline for functional outcomes (EF and UC).

  Stage 1: Feature-expanded dataset preparation (runs prepare_dataset.py)
  Stage 2: Boruta feature selection (patient-level, on training split)
  Stage 3: Hyperparameter tuning (Optuna, 50 trials with Boruta-selected features)
  Stage 4: Full training + C-index evaluation

Each stage caches its result and is skipped on re-runs unless a --force flag is given.

Usage (from repo root):
  python functional_outcomes/scripts/run_pipeline.py
  python functional_outcomes/scripts/run_pipeline.py --outcome uc --n-trials 100
  python functional_outcomes/scripts/run_pipeline.py --storage sqlite:///functional_outcomes/outputs/pipeline.db
  python functional_outcomes/scripts/run_pipeline.py --outcome ef --force-hpt --force-train
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
from tune_functional import _run_trial
from train_functional import train_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_DIR = os.path.join(_REPO_ROOT, "functional_outcomes", "config")
_CONFIG_MAP = {
    "ef": os.path.join(_CONFIG_DIR, "config_ef.yaml"),
    "uc": os.path.join(_CONFIG_DIR, "config_uc.yaml"),
}
_DATA_DIR   = os.path.join(_REPO_ROOT, "functional_outcomes", "data")
_OUTPUT_DIR = os.path.join(_REPO_ROOT, "functional_outcomes", "outputs")

# Mirror EF_STATIC / UC_STATIC from prepare_dataset.py (kept in sync manually).
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

# HPT params that live in MODEL_CONFIG (everything except batch_size)
_HPT_MODEL_KEYS = {
    "learning_rate", "dropout", "gamma",
    "num_encoder_layers", "intermediate_dim", "num_heads",
    "num_conv_blocks", "filters", "kernel_size",
    "num_dense_layers", "dense_units",
}

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
        for f in ("ef_train.csv", "ef_val.csv", "ef_test.csv",
                  "uc_train.csv", "uc_val.csv", "uc_test.csv")
    ]
    if not force and all(os.path.exists(p) for p in expected):
        print("[Stage 1] CSVs already exist — skipping (use --force-prepare to regenerate).")
        return

    print("[Stage 1] Running prepare_dataset.py ...")
    script = os.path.join(_REPO_ROOT, "functional_outcomes", "scripts", "prepare_dataset.py")
    result = subprocess.run([sys.executable, script], cwd=_REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"prepare_dataset.py exited with code {result.returncode}")
    print("[Stage 1] Done.")


# ---------------------------------------------------------------------------
# Stage 2 — Boruta feature selection
# ---------------------------------------------------------------------------

def _boruta_select(train_csv: str, candidates: list,
                   random_state: int, max_iter: int,
                   include_tentative: bool) -> tuple:
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer

    df = pd.read_csv(train_csv).set_index("id")

    # Keep only candidates actually present in the CSV
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

    if not confirmed:
        print("  WARNING: Boruta confirmed 0 features — falling back to all candidates.")
        confirmed = list(available)
        tentative = []

    selected = confirmed + [f for f in tentative if f not in confirmed] \
        if include_tentative else confirmed
    return confirmed, tentative, selected


def stage_2_boruta(outcome: str, random_state: int, max_iter: int,
                   include_tentative: bool, force: bool) -> list:
    out_path = os.path.join(_OUTPUT_DIR, f"boruta_{outcome}_features.json")

    if not force and os.path.exists(out_path):
        print(f"[Stage 2/{outcome.upper()}] Loading cached Boruta result from {out_path}")
        with open(out_path) as f:
            data = json.load(f)
        selected = data["selected"]
        print(f"  Selected ({len(selected)}): {selected}")
        return selected

    train_csv  = os.path.join(_DATA_DIR, f"{outcome}_train.csv")
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
# Stage 3 — Hyperparameter tuning
# ---------------------------------------------------------------------------

def stage_3_hpt(outcome: str, selected_features: list,
                n_trials: int, storage: str, force: bool) -> dict:
    import optuna

    out_path = os.path.join(_OUTPUT_DIR, f"hpt_best_{outcome}.json")

    if not force and os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
        cached_feats   = set(data.get("static_features", []))
        current_feats  = set(selected_features)
        if cached_feats != current_feats:
            print(f"[Stage 3/{outcome.upper()}] WARNING: cached HPT used different features.")
            print(f"  Cached:  {sorted(cached_feats)}")
            print(f"  Current: {sorted(current_feats)}")
            print("  Use --force-hpt to re-run with current features.")
        else:
            print(f"[Stage 3/{outcome.upper()}] Loading cached HPT result from {out_path}")
        return data["params"]

    config = _make_config(outcome, selected_features)
    set_seeds(config.SEED)

    print(f"[Stage 3/{outcome.upper()}] Loading data ({len(selected_features)} features) ...")
    train_ds, val_ds, _, _, _, _ = load_and_preprocess_data(
        config.TRAIN_PATH, config.VAL_PATH, config.TEST_PATH,
        config.STATIC_FEATURES, config.DYNAMIC_FEATURES,
        config.SEQ_LENGTH, config.BATCH_SIZE,
        config.T_MAX, config.AUGMENT_DATA, config.SCALE_FEATURES,
    )
    n_features = len(selected_features) + len(config.DYNAMIC_FEATURES)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study  = optuna.create_study(
        direction="minimize",
        study_name=f"bertpca_{outcome}_pipeline_hpt",
        storage=storage,
        load_if_exists=True,
        pruner=pruner,
    )

    print(f"[Stage 3/{outcome.upper()}] Starting {n_trials} Optuna trials ...")
    study.optimize(
        lambda trial: _run_trial(trial, train_ds, val_ds, config, n_features),
        n_trials=n_trials,
        catch=(Exception,),
    )

    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        print(f"[Stage 3/{outcome.upper()}] WARNING: all trials failed/pruned — using YAML defaults.")
        best_params   = {}
        best_val_nll  = float("inf")
    else:
        best_params   = study.best_trial.params
        best_val_nll  = study.best_trial.value

    print(f"  Best val NLL: {best_val_nll:.6f}")
    print(f"  Best params:  {best_params}")

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "outcome":         outcome,
            "static_features": selected_features,
            "n_trials":        n_trials,
            "best_val_nll":    best_val_nll,
            "params":          best_params,
        }, f, indent=2)
    print(f"  Saved to {out_path}")
    return best_params


# ---------------------------------------------------------------------------
# Stage 4 — Full training and evaluation
# ---------------------------------------------------------------------------

def stage_4_train(outcome: str, selected_features: list,
                  best_params: dict, force: bool) -> None:
    os.makedirs(os.path.join(_OUTPUT_DIR, "models"), exist_ok=True)
    model_path = os.path.join(_OUTPUT_DIR, "models", f"pipeline_model_{outcome}.keras")

    if not force and os.path.exists(model_path):
        print(f"[Stage 4/{outcome.upper()}] Model already exists — skipping (use --force-train).")
        return

    config = _make_config(outcome, selected_features)

    # Apply HPT-tuned model params
    for key in _HPT_MODEL_KEYS:
        if key in best_params:
            config.MODEL_CONFIG[key] = best_params[key]

    if "batch_size" in best_params:
        config.BATCH_SIZE = best_params["batch_size"]
        config.TRAINING_CONFIG["batch_size"] = best_params["batch_size"]

    # Redirect outputs to pipeline-specific sub-directory
    config.RESULTS_DIR = os.path.join(config.RESULTS_DIR, "pipeline")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR,   exist_ok=True)

    print(f"[Stage 4/{outcome.upper()}] Training with {len(selected_features)} features ...")
    print(f"  Static:    {selected_features}")
    print(f"  Model cfg: {config.MODEL_CONFIG}")
    print(f"  Batch sz:  {config.BATCH_SIZE}")

    train_model(config, output_path=model_path)
    print(f"[Stage 4/{outcome.upper()}] Model saved to {model_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BertPCa pipeline: feature expansion → Boruta → HPT → training"
    )
    parser.add_argument("--outcome", choices=["ef", "uc", "all"], default="ef")
    parser.add_argument("--n-trials",         type=int,  default=50)
    parser.add_argument("--force",            action="store_true", help="Re-run all stages")
    parser.add_argument("--force-prepare",    action="store_true")
    parser.add_argument("--force-boruta",     action="store_true")
    parser.add_argument("--force-hpt",        action="store_true")
    parser.add_argument("--force-train",      action="store_true")
    parser.add_argument("--include-tentative",action="store_true",
                        help="Include Boruta tentative features alongside confirmed ones")
    parser.add_argument("--boruta-max-iter",  type=int, default=100)
    parser.add_argument("--storage",          type=str, default=None,
                        help="Optuna storage URL, e.g. sqlite:///pipeline.db")
    parser.add_argument("--random-state",     type=int, default=42)
    args = parser.parse_args()

    fa = args.force
    force = {
        "prepare": args.force_prepare or fa,
        "boruta":  args.force_boruta  or fa,
        "hpt":     args.force_hpt     or fa,
        "train":   args.force_train   or fa,
    }
    outcomes = ["ef", "uc"] if args.outcome == "all" else [args.outcome]

    # ── Stage 1 (shared) ──────────────────────────────────────────
    print(f"\n{SEP}\n  STAGE 1 — Dataset Preparation\n{SEP}")
    stage_1_prepare(force=force["prepare"])

    # ── Stages 2-4 (per outcome) ──────────────────────────────────
    for outcome in outcomes:
        print(f"\n{SEP}\n  OUTCOME: {outcome.upper()}\n{SEP}")

        print(f"\n--- Stage 2: Boruta Feature Selection ({outcome.upper()}) ---")
        selected = stage_2_boruta(
            outcome,
            random_state=args.random_state,
            max_iter=args.boruta_max_iter,
            include_tentative=args.include_tentative,
            force=force["boruta"],
        )

        print(f"\n--- Stage 3: HPT ({outcome.upper()}, {args.n_trials} trials) ---")
        best_params = stage_3_hpt(
            outcome, selected,
            n_trials=args.n_trials,
            storage=args.storage,
            force=force["hpt"],
        )

        print(f"\n--- Stage 4: Full Training ({outcome.upper()}) ---")
        stage_4_train(outcome, selected, best_params, force=force["train"])

    print(f"\n{SEP}")
    print(f"  Pipeline complete: {', '.join(o.upper() for o in outcomes)}")
    print(SEP)


if __name__ == "__main__":
    main()
