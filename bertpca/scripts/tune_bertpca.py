#!/usr/bin/env python3
"""
Hyperparameter tuning script for BertPCa using Optuna.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import optuna

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, project_root)

from bertpca import (
    build_bert_pca,
    load_and_preprocess_data,
    training_loop,
)
from bertpca.utils import set_seeds
from config.load_config import load_yaml_config

config = load_yaml_config()

OPTIMIZATION_LOG_FILENAME = "optimization_log.txt"


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


def objective(trial, train_ds, val_ds, y_train_struct, y_val_struct):
    """
    Objective function for Optuna hyperparameter optimization.
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    train_ds : Dataset
        Training dataset
    val_ds : Dataset
        Validation dataset
    y_train_struct : np.ndarray
        Structured training labels
    y_val_struct : np.ndarray
        Structured validation labels
    
    Returns
    -------
    float
        Validation loss to minimize
    """
    # Set seeds for reproducibility
    set_seeds(config.SEED)
    
    # Clear Keras session
    keras.backend.clear_session()
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 5e-4])
    num_encoder_layers = trial.suggest_categorical("num_encoder_layers", [1, 2, 3])
    intermediate_dim = trial.suggest_categorical("intermediate_dim", [32, 64, 128, 256])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 6])
    num_conv_blocks = trial.suggest_categorical("num_conv_blocks", [1, 2, 3])
    filters = trial.suggest_categorical("filters", [32, 64, 128])
    kernel_size = trial.suggest_categorical("kernel_size", [2, 3, 5])
    pool_strides = trial.suggest_categorical("pool_strides", [1, 2])
    pool_size = trial.suggest_categorical("pool_size", [2, 3])
    num_dense_layers = trial.suggest_categorical("num_dense_layers", [1, 2, 3, 4, 5])
    dense_units = trial.suggest_categorical("dense_units", [64, 128, 256])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3, 0.4])
    
    # Build model
    n_features = len(config.STATIC_FEATURES) + len(config.DYNAMIC_FEATURES)
    model = build_bert_pca(
        n_features=n_features,
        seq_length=config.SEQ_LENGTH,
        learning_rate=learning_rate,
        num_encoder_layers=num_encoder_layers,
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        num_conv_blocks=num_conv_blocks,
        filters=filters,
        kernel_size=kernel_size,
        pool_strides=pool_strides,
        pool_size=pool_size,
        num_dense_layers=num_dense_layers,
        dense_units=dense_units,
        activation='relu',
        norm_epsilon=1e-5,
        dropout=dropout,
        gamma=0.0
    )
    
    X_train = np.array(train_ds["features"])
    y_train = np.array(train_ds["labels_surv"])
    X_val = np.array(val_ds["features"])
    y_val = np.array(val_ds["labels_surv"])

    batch_size = config.TRAINING_CONFIG["batch_size"]
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    model, history = training_loop(
        model,
        train_dataset,
        val_dataset,
        y_train=y_train_struct,
        y_val=y_val_struct,
        training_config=config.TRAINING_CONFIG,
        evaluation_config=config.EVALUATION_CONFIG,
        c_index_interval=5,
    )

    min_val_loss = min(history["val_loss"])
    
    # Save best model
    if not hasattr(objective, "best_loss") or min_val_loss < objective.best_loss:
        print(f"New best model found! Loss: {min_val_loss:.6f}")
        objective.best_loss = min_val_loss
        model.save(os.path.join(config.MODEL_DIR, "best_model_tuned.keras"))
    
    return min_val_loss


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for BertPCa")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=config.TUNING_CONFIG["n_trials"],
        help="Number of trials to run",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=config.TUNING_CONFIG["study_name"],
        help="Name for the Optuna study",
    )

    args = parser.parse_args()

    original_stdout = sys.stdout
    tee = None
    results_dir = config.RESULTS_DIR
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        log_path = os.path.join(results_dir, OPTIMIZATION_LOG_FILENAME)
        tee = Tee(sys.stdout, log_path)
        sys.stdout = tee

    try:
        set_seeds(config.SEED)
        objective.best_loss = float("inf")

        print("Loading and preprocessing data...")
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

        study = optuna.create_study(direction=config.TUNING_CONFIG["direction"])

        print(
            f"Starting hyperparameter optimization with {args.n_trials} trials..."
        )

        study.optimize(
            lambda trial: objective(
                trial, train_ds, val_ds, y_train_struct, y_val_struct
            ),
            n_trials=args.n_trials,
        )

        print("\n" + "=" * 50)
        print("Hyperparameter Optimization Results")
        print("=" * 50)
        print(f"Number of finished trials: {len(study.trials)}")
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value (loss): {trial.value:.6f}")
        print("\n  Best parameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        results_path = os.path.join(config.RESULTS_DIR, "tuning_results.txt")
        with open(results_path, "w") as f:
            f.write("Hyperparameter Optimization Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Number of finished trials: {len(study.trials)}\n")
            f.write(f"\nBest trial value (loss): {trial.value:.6f}\n")
            f.write("\nBest parameters:\n")
            for key, value in trial.params.items():
                f.write(f"  {key}: {value}\n")

        log_path = os.path.join(results_dir, OPTIMIZATION_LOG_FILENAME)
        print(f"\nOptimization log saved to {log_path}")
        print(f"Results saved to {results_path}")
    finally:
        if tee is not None:
            sys.stdout = original_stdout
            tee.close()


if __name__ == "__main__":
    main()
