#!/usr/bin/env python3
"""
Hyperparameter tuning for BertPCa on EF or UC functional outcomes.

Key differences vs. bertpca/scripts/tune_bertpca.py:
  - --outcome ef/uc selector with per-outcome configs
  - gamma: weight on ranking_loss added to weibull_loss; forces patient
    differentiation so C-index can escape the constant-prediction trap
  - batch_size: tunable (larger batch → more stable gradients, fewer NaN collapses)
  - learning_rate: extended range down to 1e-6 for stability
  - Custom per-batch loss so ranking loss can be combined without recompiling
  - NaN-safe objective: trials that collapse to all-NaN val loss are pruned

Usage (from repo root):
  python functional_outcomes/scripts/tune_functional.py --outcome uc --n-trials 50
  python functional_outcomes/scripts/tune_functional.py --outcome ef --n-trials 50
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import optuna

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bertpca", "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "bertpca"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "functional_outcomes"))

from bertpca import build_bert_pca, load_and_preprocess_data, set_seeds
from bertpca.loss import weibull_loss, ranking_loss
from config.load_config import load_yaml_config

_CONFIG_DIR = os.path.join(_REPO_ROOT, "functional_outcomes", "config")
_CONFIG_MAP = {
    "ef": os.path.join(_CONFIG_DIR, "config_ef.yaml"),
    "uc": os.path.join(_CONFIG_DIR, "config_uc.yaml"),
}


def _combined_loss(y_true, y_pred, gamma: float) -> tf.Tensor:
    """weibull_loss + gamma * ranking_loss, NaN-safe."""
    wl = weibull_loss(y_true, y_pred)
    if gamma == 0.0:
        return wl
    rl = ranking_loss(y_true, y_pred)
    # ranking_loss can produce NaN on degenerate batches — fall back to weibull only
    if tf.math.is_nan(rl):
        return wl
    return wl + tf.cast(gamma, tf.float32) * tf.cast(rl, tf.float32)


def _run_trial(trial, train_ds, val_ds, config, n_features: int) -> float:
    """
    Optuna objective: returns best (lowest) validation Weibull NLL seen during
    the trial, or a large penalty if all val epochs were NaN.
    """
    set_seeds(config.SEED)
    keras.backend.clear_session()

    # --- Hyperparameter suggestions ---
    learning_rate   = trial.suggest_categorical("learning_rate",    [1e-6, 5e-6, 1e-5, 5e-5])
    batch_size      = trial.suggest_categorical("batch_size",       [16, 32, 64])
    dropout         = trial.suggest_categorical("dropout",          [0.1, 0.2, 0.3, 0.4])
    gamma           = trial.suggest_categorical("gamma",            [0.0, 0.001, 0.01, 0.1])
    num_encoder_layers = trial.suggest_categorical("num_encoder_layers", [1, 2])
    intermediate_dim   = trial.suggest_categorical("intermediate_dim",   [64, 128, 256])
    num_heads          = trial.suggest_categorical("num_heads",          [2, 4])
    num_conv_blocks    = trial.suggest_categorical("num_conv_blocks",    [1, 2, 3])
    filters            = trial.suggest_categorical("filters",            [64, 128])
    kernel_size        = trial.suggest_categorical("kernel_size",        [3, 5])
    num_dense_layers   = trial.suggest_categorical("num_dense_layers",   [2, 3])
    dense_units        = trial.suggest_categorical("dense_units",        [128, 256])

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
        pool_strides=config.MODEL_CONFIG.get("pool_strides", 2),
        pool_size=config.MODEL_CONFIG.get("pool_size", 3),
        num_dense_layers=num_dense_layers,
        dense_units=dense_units,
        activation="relu",
        norm_epsilon=1e-5,
        dropout=dropout,
        gamma=gamma,
    )
    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    X_train = np.array(train_ds["features"])
    y_train = np.array(train_ds["labels_surv"])
    X_val   = np.array(val_ds["features"])
    y_val   = np.array(val_ds["labels_surv"])

    train_tf = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=2048, seed=config.SEED)
        .batch(batch_size)
    )
    val_tf = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    # Shorter epochs for HPT — enough to detect instability and learning
    max_epochs       = 60
    patience         = 8
    reduce_lr_wait   = 4
    reduce_lr_factor = 0.5
    min_lr           = 1e-7
    clip_norm        = 0.5   # tighter than default 1.0 to reduce NaN collapse

    best_val_nll = float("inf")
    patience_ctr = lr_patience_ctr = 0
    nan_epoch_ctr = 0

    for epoch in range(max_epochs):
        # --- Train ---
        for x_batch, y_batch in train_tf:
            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=True)
                loss_value = _combined_loss(y_batch, y_pred, gamma)

            if tf.math.is_nan(loss_value):
                continue  # skip NaN training batch

            grads = tape.gradient(loss_value, model.trainable_weights)
            grads = [
                tf.clip_by_norm(g, clip_norm) if g is not None else g
                for g in grads
            ]
            if any(
                tf.reduce_any(tf.math.is_nan(g)) for g in grads if g is not None
            ):
                continue  # skip NaN gradient batch

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # --- Val (Weibull NLL only — no ranking term in objective) ---
        val_nll_list = []
        for x_v, y_v in val_tf:
            y_pred_v = model(x_v, training=False)
            vl = weibull_loss(y_v, y_pred_v)
            val_nll = float(vl.numpy())
            if not np.isnan(val_nll):
                val_nll_list.append(val_nll)

        if not val_nll_list:
            # Entire val epoch NaN
            nan_epoch_ctr += 1
            patience_ctr   += 1
            lr_patience_ctr += 1
            if nan_epoch_ctr >= 3:
                # Collapsed — prune this trial early
                raise optuna.exceptions.TrialPruned()
            continue

        nan_epoch_ctr = 0
        avg_val_nll = float(np.mean(val_nll_list))

        if avg_val_nll < best_val_nll:
            best_val_nll = avg_val_nll
            patience_ctr    = 0
            lr_patience_ctr = 0
        else:
            patience_ctr    += 1
            lr_patience_ctr += 1

        # LR reduction
        if lr_patience_ctr >= reduce_lr_wait:
            current_lr = float(optimizer.learning_rate.numpy())
            new_lr = max(current_lr * reduce_lr_factor, min_lr)
            optimizer.learning_rate.assign(new_lr)
            lr_patience_ctr = 0

        # Early stopping
        if patience_ctr >= patience:
            break

        # Optuna intermediate reporting for early pruning
        trial.report(avg_val_nll, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_nll if best_val_nll < float("inf") else 1e6


def main():
    parser = argparse.ArgumentParser(description="HPT for BertPCa functional outcomes")
    parser.add_argument("--outcome", choices=["ef", "uc"], required=True,
                        help="Outcome to tune: 'ef' or 'uc'")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--study-name", type=str, default=None,
                        help="Optuna study name (default: bertpca_<outcome>_hpt)")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL for resumable studies (e.g. sqlite:///hpt.db)")
    args = parser.parse_args()

    config = load_yaml_config(_CONFIG_MAP[args.outcome])

    # Resolve absolute paths
    config.TRAIN_PATH = os.path.join(_REPO_ROOT, config.TRAIN_PATH)
    config.VAL_PATH   = os.path.join(_REPO_ROOT, config.VAL_PATH)
    config.TEST_PATH  = os.path.join(_REPO_ROOT, config.TEST_PATH)
    results_dir = os.path.join(_REPO_ROOT, config.RESULTS_DIR, "hpt")
    model_dir   = os.path.join(_REPO_ROOT, config.MODEL_DIR)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    set_seeds(config.SEED)

    print(f"Loading data for {args.outcome.upper()} ...")
    train_ds, val_ds, _, _, _, _ = load_and_preprocess_data(
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
    n_features = len(config.STATIC_FEATURES) + len(config.DYNAMIC_FEATURES)

    study_name = args.study_name or f"bertpca_{args.outcome}_hpt"
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
        pruner=pruner,
    )

    print(f"Starting {args.n_trials} trials for {args.outcome.upper()} ...")
    study.optimize(
        lambda trial: _run_trial(trial, train_ds, val_ds, config, n_features),
        n_trials=args.n_trials,
        catch=(Exception,),
    )

    print("\n" + "=" * 60)
    print(f"HPT complete — {args.outcome.upper()}")
    print(f"Finished trials: {len(study.trials)}")
    best = study.best_trial
    print(f"Best val NLL:    {best.value:.6f}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    out_path = os.path.join(results_dir, f"hpt_results_{args.outcome}.txt")
    with open(out_path, "w") as f:
        f.write(f"HPT results — {args.outcome.upper()}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Finished trials: {len(study.trials)}\n")
        f.write(f"Best val NLL:    {best.value:.6f}\n\n")
        f.write("Best params:\n")
        for k, v in best.params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nAll completed trials (sorted by value):\n")
        completed = [t for t in study.trials if t.value is not None]
        completed.sort(key=lambda t: t.value)
        for t in completed[:20]:
            f.write(f"  Trial {t.number:3d}: val_nll={t.value:.6f}  {t.params}\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
