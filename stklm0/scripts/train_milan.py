#!/usr/bin/env python3
"""
Train BertPCa on the Milan dataset using the STKLM0 feature schema.
Saves the model and scaling parameters for later evaluation on STKLM0.

Prerequisites:
  python stklm0/scripts/prepare_milan.py --outcome bcr   # or csm or both

Run from repo root:
  python stklm0/scripts/train_milan.py --outcome bcr
  python stklm0/scripts/train_milan.py --outcome csm
"""

import csv
import json
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bertpca", "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bertpca"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "stklm0"))

from bertpca import build_bert_pca, training_loop, set_seeds
from bertpca.data import preprocess_data, augment_dataframe
from bertpca.train import TRAINING_LOG_FILENAME
from config.load_config import load_yaml_config

_CONFIG_PATH = os.path.join(_REPO_ROOT, "stklm0", "config", "config_stklm0.yaml")
_DATA_DIR    = os.path.join(_REPO_ROOT, "stklm0", "data")


class Tee:
    def __init__(self, stream, path):
        self._stream = stream
        self._file   = open(path, "w", encoding="utf-8")

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _load_csv(path):
    df = pd.read_csv(path)
    df.set_index("id", inplace=True)
    return df.astype(float)


def _build_structured_labels(df_long):
    dt   = np.dtype([("Status", "?"), ("Survival_in_days", "<f8")])
    last = df_long.groupby(level=0).last()
    return np.array(list(zip(last["label"].astype(bool), last["tte"])), dtype=dt)


def _scale_df(df, features_to_scale, train_max, train_min, t_max):
    denom = (train_max - train_min).replace(0, 1)
    df = df.copy()
    df[features_to_scale] = (df[features_to_scale] - train_min) / denom
    df[["tte", "times"]]  = df[["tte", "times"]] / t_max
    return df


def run(outcome: str, output_path: str = None):
    config = load_yaml_config(_CONFIG_PATH)
    config.MODEL_DIR   = os.path.join(_REPO_ROOT, config.MODEL_DIR)
    config.RESULTS_DIR = os.path.join(_REPO_ROOT, "stklm0", "outputs", "results",
                                       f"milan_{outcome}_train")
    os.makedirs(config.MODEL_DIR,   exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    set_seeds(config.SEED)

    original_stdout = sys.stdout
    tee = Tee(sys.stdout, os.path.join(config.RESULTS_DIR, TRAINING_LOG_FILENAME))
    sys.stdout = tee

    try:
        train_path = os.path.join(_DATA_DIR, f"milan_{outcome}_train.csv")
        val_path   = os.path.join(_DATA_DIR, f"milan_{outcome}_val.csv")

        for p in [train_path, val_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Missing: {p}\n"
                    f"Run: python stklm0/scripts/prepare_milan.py --outcome {outcome}"
                )

        print(f"Loading Milan {outcome.upper()} train/val ...")
        train_raw = _load_csv(train_path)
        val_raw   = _load_csv(val_path)

        # Shared features: only those present in the Milan CSV (some may have been dropped)
        all_static = config.STATIC_FEATURES
        available_static = [f for f in all_static if f in train_raw.columns]
        dropped = set(all_static) - set(available_static)
        if dropped:
            print(f"Features not in Milan — dropped: {sorted(dropped)}")
        print(f"Training features ({len(available_static)}): {available_static}")

        dynamic_features  = config.DYNAMIC_FEATURES
        t_max             = config.T_MAX
        features_to_scale = [f for f in available_static + dynamic_features if f != "times"]

        # Compute and persist scaling parameters from Milan train
        train_max = train_raw[features_to_scale].max()
        train_min = train_raw[features_to_scale].min()

        train_scaled = _scale_df(train_raw, features_to_scale, train_max, train_min, t_max)
        val_scaled   = _scale_df(val_raw,   features_to_scale, train_max, train_min, t_max)

        y_train_struct = _build_structured_labels(train_scaled)
        y_val_struct   = _build_structured_labels(val_scaled)

        train_aug = augment_dataframe(train_scaled)
        print(f"  Train: {train_scaled.index.nunique()} patients "
              f"({train_aug.index.nunique()} after augmentation)")
        print(f"  Val:   {val_scaled.index.nunique()} patients")

        train_ds, _ = preprocess_data(train_aug, available_static, dynamic_features,
                                       "label", config.SEQ_LENGTH, config.BATCH_SIZE)
        val_ds,   _ = preprocess_data(val_scaled, available_static, dynamic_features,
                                       "label", config.SEQ_LENGTH, config.BATCH_SIZE)

        X_train = np.array(train_ds["features"])
        y_train = np.array(train_ds["labels_surv"])
        X_val   = np.array(val_ds["features"])
        y_val   = np.array(val_ds["labels_surv"])

        batch_size = config.TRAINING_CONFIG["batch_size"]
        train_tf   = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                      .shuffle(1024).batch(batch_size))
        val_tf     = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

        n_features = len(available_static) + len(dynamic_features)
        print(f"\nBuilding model ({n_features} features) ...")
        keras.backend.clear_session()
        model = build_bert_pca(n_features=n_features, seq_length=config.SEQ_LENGTH,
                                **config.MODEL_CONFIG)

        print(f"Training BertPCa on Milan {outcome.upper()} ...")
        model, _ = training_loop(
            model, train_tf, val_tf,
            y_train=y_train_struct, y_val=y_val_struct,
            training_config=config.TRAINING_CONFIG,
            evaluation_config=config.EVALUATION_CONFIG,
            c_index_interval=5,
        )
        print("Training complete.")

        # ---- Save model ----------------------------------------------------------
        if output_path is None:
            output_path = os.path.join(config.MODEL_DIR, f"best_model_milan_{outcome}.keras")
        model.save(output_path)
        print(f"Model saved to {output_path}")

        # ---- Save scaling parameters (needed by eval_stklm0.py) -----------------
        scaling_path = os.path.join(config.MODEL_DIR, f"milan_{outcome}_scaling.json")
        scaling_params = {
            "available_static":  available_static,
            "dynamic_features":  dynamic_features,
            "features_to_scale": features_to_scale,
            "train_max":         train_max.to_dict(),
            "train_min":         train_min.to_dict(),
            "t_max":             t_max,
            "model_path":        output_path,
        }
        with open(scaling_path, "w") as f:
            json.dump(scaling_params, f, indent=2)
        print(f"Scaling params saved to {scaling_path}")

        # ---- Save training-set structured labels (needed for IPCW in eval) ------
        y_train_path = os.path.join(config.MODEL_DIR, f"milan_{outcome}_y_train.npz")
        np.savez(y_train_path,
                 Status=y_train_struct["Status"],
                 Survival_in_days=y_train_struct["Survival_in_days"])
        print(f"Training labels saved to {y_train_path}")

        return model

    finally:
        sys.stdout = original_stdout
        tee.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train BertPCa on Milan data (STKLM0 feature schema)"
    )
    parser.add_argument("--outcome", choices=["bcr", "csm", "both"], default="both",
                        help="Milan training outcome (default: both)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the trained model (.keras) — only used when --outcome is bcr or csm")
    args = parser.parse_args()

    outcomes = ["bcr", "csm"] if args.outcome == "both" else [args.outcome]

    for outcome in outcomes:
        output = args.output
        if output and len(outcomes) == 1 and not os.path.isabs(output):
            output = os.path.join(_REPO_ROOT, output)
        elif len(outcomes) > 1:
            output = None  # use default per-outcome path

        run(outcome, output_path=output)
        print("\nDone.")


if __name__ == "__main__":
    main()
