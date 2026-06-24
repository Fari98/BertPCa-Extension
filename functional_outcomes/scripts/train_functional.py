#!/usr/bin/env python3
"""
Train BertPCa on a functional outcome (EF or UC).

Usage:
  python functional_outcomes/scripts/train_functional.py --outcome ef
  python functional_outcomes/scripts/train_functional.py --outcome uc
  python functional_outcomes/scripts/train_functional.py --outcome ef --output functional_outcomes/outputs/models/ef_model.keras
"""

import csv
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Resolve paths relative to repo root (two levels up from this file)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bertpca", "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bertpca"))

from bertpca import (
    build_bert_pca,
    load_and_preprocess_data,
    calculate_time_dependent_c_index,
    training_loop,
    set_seeds,
)
from bertpca.train import TRAINING_LOG_FILENAME
from config.load_config import load_yaml_config

_CONFIG_DIR = os.path.join(_REPO_ROOT, "functional_outcomes", "config")
_CONFIG_MAP = {
    "ef": os.path.join(_CONFIG_DIR, "config_ef.yaml"),
    "uc": os.path.join(_CONFIG_DIR, "config_uc.yaml"),
}


class Tee:
    """Write to both stdout and a file."""

    def __init__(self, stream, path):
        self._stream = stream
        self._file = open(path, "w", encoding="utf-8")

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


def train_model(config, output_path=None):
    seed = config.SEED
    set_seeds(seed)

    results_dir = config.RESULTS_DIR
    original_stdout = sys.stdout
    tee = None
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        tee = Tee(sys.stdout, os.path.join(results_dir, TRAINING_LOG_FILENAME))
        sys.stdout = tee

    try:
        print("Loading and preprocessing data ...")
        train_ds, val_ds, test_ds, y_train_struct, y_val_struct, y_test_struct = (
            load_and_preprocess_data(
                config.TRAIN_PATH,
                config.VAL_PATH,
                config.TEST_PATH,
                config.STATIC_FEATURES,
                config.DYNAMIC_FEATURES,
                config.SEQ_LENGTH,
                config.BATCH_SIZE,
                config.T_MAX,
                config.AUGMENT_DATA,
                config.SCALE_FEATURES,
            )
        )

        n_features = len(config.STATIC_FEATURES) + len(config.DYNAMIC_FEATURES)

        print("Building model ...")
        keras.backend.clear_session()
        model = build_bert_pca(n_features=n_features, seq_length=config.SEQ_LENGTH, **config.MODEL_CONFIG)

        X_train = np.array(train_ds["features"])
        y_train = np.array(train_ds["labels_surv"])
        X_val = np.array(val_ds["features"])
        y_val = np.array(val_ds["labels_surv"])

        batch_size = config.TRAINING_CONFIG["batch_size"]
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(1024).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

        print("Training ...")
        model, history = training_loop(
            model, train_dataset, val_dataset,
            y_train=y_train_struct, y_val=y_val_struct,
            training_config=config.TRAINING_CONFIG,
            evaluation_config=config.EVALUATION_CONFIG,
            c_index_interval=5,
        )
        print("Training complete.")

        print("Evaluating on test set ...")
        p_times = np.array(config.EVALUATION_CONFIG["p_times"])
        e_times = np.array(config.EVALUATION_CONFIG["e_times"])
        test_results = calculate_time_dependent_c_index(
            np.array(test_ds["features"]), y_train_struct, y_test_struct,
            model, p_times=p_times, e_times=e_times,
            t_max=config.EVALUATION_CONFIG["t_max"], return_mean=False,
        )

        print("\nTest C-Index Results:")
        print(test_results)
        mean_c = float(np.mean(test_results))
        print(f"\nMean C-Index: {mean_c:.6f}")

        if results_dir:
            with open(os.path.join(results_dir, "mean_c_index.txt"), "w") as f:
                f.write(f"{mean_c:.6f}\n")
            table_path = os.path.join(results_dir, "c_index_table.csv")
            with open(table_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["p_time"] + [f"e_time_{int(e)}" for e in e_times])
                for i, p in enumerate(p_times):
                    writer.writerow([int(p)] + [f"{test_results[i, j]:.6f}" for j in range(len(e_times))])
            print(f"C-index table saved to {table_path}")

        if output_path is None:
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            output_path = os.path.join(config.MODEL_DIR, "best_model.keras")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        model.save(output_path)
        print(f"Model saved to {output_path}")

        return model, history, test_results

    finally:
        if tee is not None:
            sys.stdout = original_stdout
            tee.close()


def main():
    parser = argparse.ArgumentParser(description="Train BertPCa on functional outcomes")
    parser.add_argument("--outcome", choices=["ef", "uc"], required=True,
                        help="Outcome to train on: 'ef' (erectile function) or 'uc' (urinary continence)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the trained model (.keras)")
    args = parser.parse_args()

    config_path = _CONFIG_MAP[args.outcome]
    config = load_yaml_config(config_path)

    # Override results_dir to be outcome-specific if needed
    results_dir = os.path.join(_REPO_ROOT, config.RESULTS_DIR)
    model_dir = os.path.join(_REPO_ROOT, config.MODEL_DIR)
    config.RESULTS_DIR = results_dir
    config.MODEL_DIR = model_dir
    config.TRAIN_PATH = os.path.join(_REPO_ROOT, config.TRAIN_PATH)
    config.VAL_PATH = os.path.join(_REPO_ROOT, config.VAL_PATH)
    config.TEST_PATH = os.path.join(_REPO_ROOT, config.TEST_PATH)

    output_path = args.output
    if output_path:
        output_path = os.path.join(_REPO_ROOT, output_path)
    else:
        os.makedirs(model_dir, exist_ok=True)
        output_path = os.path.join(model_dir, f"best_model_{args.outcome}.keras")

    train_model(config, output_path=output_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
