#!/usr/bin/env python3
"""
Step 2: Train and evaluate BertPCa on the STKLM0 dataset.

Uses a standard 80/10/10 train/val/test split (produced by prepare_stklm0.py)
and reports the time-dependent C-index with IPCW on the held-out test set.

Prerequisites:
  python stklm0/scripts/prepare_stklm0.py --input data/stklm0.csv

Run from repo root:
  python stklm0/scripts/train_eval_stklm0.py
  python stklm0/scripts/train_eval_stklm0.py --output stklm0/outputs/models/my_model.keras
"""

import csv
import os
import sys
import argparse
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bertpca", "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bertpca"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "stklm0"))

from bertpca import (
    build_bert_pca,
    load_and_preprocess_data,
    calculate_time_dependent_c_index,
    training_loop,
    set_seeds,
)
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


def run(output_path: str = None):
    config = load_yaml_config(_CONFIG_PATH)

    # Absolute paths
    config.TRAIN_PATH   = os.path.join(_DATA_DIR, "stklm0_train.csv")
    config.VAL_PATH     = os.path.join(_DATA_DIR, "stklm0_val.csv")
    config.TEST_PATH    = os.path.join(_DATA_DIR, "stklm0_test.csv")
    config.MODEL_DIR    = os.path.join(_REPO_ROOT, config.MODEL_DIR)
    config.RESULTS_DIR  = os.path.join(_REPO_ROOT, config.RESULTS_DIR)

    for p in [config.TRAIN_PATH, config.VAL_PATH, config.TEST_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing: {p}\n"
                "Run prepare_stklm0.py first:\n"
                "  python stklm0/scripts/prepare_stklm0.py --input data/stklm0.csv"
            )

    os.makedirs(config.MODEL_DIR,   exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    set_seeds(config.SEED)

    original_stdout = sys.stdout
    tee = Tee(sys.stdout, os.path.join(config.RESULTS_DIR, TRAINING_LOG_FILENAME))
    sys.stdout = tee

    try:
        print("Loading and preprocessing STKLM0 data ...")
        train_ds, val_ds, test_ds, y_train_struct, y_val_struct, y_test_struct = (
            load_and_preprocess_data(
                config.TRAIN_PATH, config.VAL_PATH, config.TEST_PATH,
                config.STATIC_FEATURES, config.DYNAMIC_FEATURES,
                config.SEQ_LENGTH, config.BATCH_SIZE,
                config.T_MAX, config.AUGMENT_DATA, config.SCALE_FEATURES,
            )
        )

        n_features = len(config.STATIC_FEATURES) + len(config.DYNAMIC_FEATURES)
        print(f"  Features: {n_features} "
              f"({len(config.STATIC_FEATURES)} static + {len(config.DYNAMIC_FEATURES)} dynamic)")

        print("\nBuilding model ...")
        keras.backend.clear_session()
        model = build_bert_pca(n_features=n_features, seq_length=config.SEQ_LENGTH,
                                **config.MODEL_CONFIG)

        X_train = np.array(train_ds["features"])
        y_train = np.array(train_ds["labels_surv"])
        X_val   = np.array(val_ds["features"])
        y_val   = np.array(val_ds["labels_surv"])

        batch_size = config.TRAINING_CONFIG["batch_size"]
        train_tf   = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                      .shuffle(1024).batch(batch_size))
        val_tf     = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

        print("Training on STKLM0 ...")
        model, history = training_loop(
            model, train_tf, val_tf,
            y_train=y_train_struct, y_val=y_val_struct,
            training_config=config.TRAINING_CONFIG,
            evaluation_config=config.EVALUATION_CONFIG,
            c_index_interval=5,
        )
        print("Training complete.")

        print("\nEvaluating on STKLM0 test set ...")
        p_times = np.array(config.EVALUATION_CONFIG["p_times"])
        e_times = np.array(config.EVALUATION_CONFIG["e_times"])

        c_matrix = calculate_time_dependent_c_index(
            np.array(test_ds["features"]), y_train_struct, y_test_struct,
            model, p_times=p_times, e_times=e_times,
            t_max=config.EVALUATION_CONFIG["t_max"], return_mean=False,
        )

        print("\nTest C-Index Results (STKLM0):")
        print(c_matrix)
        mean_c = float(np.nanmean(c_matrix))
        print(f"\nMean C-Index: {mean_c:.6f}")

        with open(os.path.join(config.RESULTS_DIR, "mean_c_index.txt"), "w") as f:
            f.write(f"{mean_c:.6f}\n")

        table_path = os.path.join(config.RESULTS_DIR, "c_index_table.csv")
        with open(table_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["p_time"] + [f"e_time_{int(e)}" for e in e_times])
            for i, p in enumerate(p_times):
                writer.writerow([int(p)] + [f"{c_matrix[i, j]:.6f}" for j in range(len(e_times))])
        print(f"C-index table saved to {table_path}")

        if output_path is None:
            output_path = os.path.join(config.MODEL_DIR, "best_model_stklm0.keras")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        model.save(output_path)
        print(f"Model saved to {output_path}")

        return model, history, c_matrix

    finally:
        sys.stdout = original_stdout
        tee.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate BertPCa on STKLM0 dataset"
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the trained model (.keras)")
    args = parser.parse_args()

    output = args.output
    if output:
        output = os.path.join(_REPO_ROOT, output)

    run(output_path=output)
    print("\nDone.")


if __name__ == "__main__":
    main()
