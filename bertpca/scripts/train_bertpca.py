#!/usr/bin/env python3
"""
Training script for BertPCa survival analysis model.
"""

import csv
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, project_root)

from bertpca import (
    build_bert_pca,
    load_and_preprocess_data,
    calculate_time_dependent_c_index,
    training_loop,
    set_seeds,
)
from bertpca.train import TRAINING_LOG_FILENAME
from config.load_config import load_yaml_config

config = load_yaml_config()


class Tee:
    """Write to both stdout and a file."""

    def __init__(self, stream, path):
        self._stream = stream
        self._path = path
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


def train_model(
    data_config: dict,
    model_config: dict,
    training_config: dict,
    evaluation_config: dict,
    output_path: str = None,
    model_dir: str = None,
    results_dir: str = None,
):
    """
    Train the BertPCa model.

    Parameters
    ----------
    data_config : dict
        Data and preprocessing configuration. Expected keys:
        train_path, val_path, test_path, static_features, dynamic_features,
        seq_length, batch_size, t_max, augment_data, scale_features, seed
    model_config : dict
        Model architecture configuration (passed to build_bert_pca).
    training_config : dict
        Training configuration (epochs, batch_size, early_stopping_patience, etc.).
    evaluation_config : dict
        Evaluation configuration (p_times, e_times, t_max).
    output_path : str, optional
        Path to save the trained model. If None, uses model_dir/best_model.keras.
    model_dir : str, optional
        Directory for saving the model when output_path is None.
    results_dir : str, optional
        Directory for saving test results.

    Returns
    -------
    tuple
        (model, history, test_results)
    """
    seed = data_config.get("seed", 42)
    set_seeds(seed)

    original_stdout = sys.stdout
    tee = None
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        log_path = os.path.join(results_dir, TRAINING_LOG_FILENAME)
        tee = Tee(sys.stdout, log_path)
        sys.stdout = tee

    try:
        print("Loading and preprocessing data...")
        train_ds, val_ds, test_ds, y_train_struct, y_val_struct, y_test_struct = (
            load_and_preprocess_data(
                data_config["train_path"],
                data_config["val_path"],
                data_config["test_path"],
                data_config["static_features"],
                data_config["dynamic_features"],
                data_config["seq_length"],
                data_config["batch_size"],
                data_config["t_max"],
                data_config.get("augment_data", True),
                data_config.get("scale_features", True),
            )
        )

        n_features = len(data_config["static_features"]) + len(
            data_config["dynamic_features"]
        )

        print("Building model...")
        keras.backend.clear_session()
        model = build_bert_pca(
            n_features=n_features,
            seq_length=data_config["seq_length"],
            **model_config,
        )

        X_train = np.array(train_ds["features"])
        y_train = np.array(train_ds["labels_surv"])
        X_val = np.array(val_ds["features"])
        y_val = np.array(val_ds["labels_surv"])

        batch_size = training_config["batch_size"]
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = (
            train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        )

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size)

        print("Training model...")
        model, history = training_loop(
            model,
            train_dataset,
            val_dataset,
            y_train=y_train_struct,
            y_val=y_val_struct,
            training_config=training_config,
            evaluation_config=evaluation_config,
            c_index_interval=5,
        )
        print("Training completed!")

        print("Evaluating on test set...")
        p_times = np.array(evaluation_config["p_times"])
        e_times = np.array(evaluation_config["e_times"])

        test_results = calculate_time_dependent_c_index(
            np.array(test_ds["features"]),
            y_train_struct,
            y_test_struct,
            model,
            p_times=p_times,
            e_times=e_times,
            t_max=evaluation_config["t_max"],
            return_mean=False,
        )

        print("\nTest Set C-Index Results:")
        print(test_results)
        mean_c_index = float(np.mean(test_results))
        print(f"\nMean C-Index: {mean_c_index:.6f}")

        if results_dir:
            os.makedirs(results_dir, exist_ok=True)

            # Mean time-dependent C-index
            mean_c_index_path = os.path.join(results_dir, "mean_c_index.txt")
            with open(mean_c_index_path, "w") as f:
                f.write(f"{mean_c_index:.6f}\n")
            print(f"Mean C-index saved to {mean_c_index_path}")

            # Table: rows = p_times, columns = e_times
            p_times_arr = np.asarray(p_times)
            e_times_arr = np.asarray(e_times)
            c_index_table_path = os.path.join(results_dir, "c_index_table.csv")
            with open(c_index_table_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["p_time"] + [f"e_time_{int(e)}" for e in e_times_arr]
                )
                for i, p in enumerate(p_times_arr):
                    row = [int(p)] + [
                        f"{test_results[i, j]:.6f}"
                        for j in range(len(e_times_arr))
                    ]
                    writer.writerow(row)
            print(
                f"C-index table (p_times x e_times) saved to {c_index_table_path}"
            )

        if output_path is None and model_dir:
            output_path = os.path.join(model_dir, "best_model.keras")
        if output_path:
            print(f"\nSaving model to {output_path}...")
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            model.save(output_path)

        return model, history, test_results
    finally:
        if tee is not None:
            sys.stdout = original_stdout
            tee.close()


def main():

    parser = argparse.ArgumentParser(description="Train BertPCa model")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config file (not implemented yet)",
    )

    args = parser.parse_args()

    data_config = {
        "train_path": config.TRAIN_PATH,
        "val_path": config.VAL_PATH,
        "test_path": config.TEST_PATH,
        "static_features": config.STATIC_FEATURES,
        "dynamic_features": config.DYNAMIC_FEATURES,
        "seq_length": config.SEQ_LENGTH,
        "batch_size": config.BATCH_SIZE,
        "t_max": config.T_MAX,
        "augment_data": config.AUGMENT_DATA,
        "scale_features": config.SCALE_FEATURES,
        "seed": config.SEED,
    }

    model, history, results = train_model(
        data_config=data_config,
        model_config=config.MODEL_CONFIG,
        training_config=config.TRAINING_CONFIG,
        evaluation_config=config.EVALUATION_CONFIG,
        output_path=args.output,
        model_dir=config.MODEL_DIR,
        results_dir=config.RESULTS_DIR,
    )

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
