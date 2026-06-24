"""
Training logic for BertPCa survival analysis model.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

from .evaluation import calculate_time_dependent_c_index

TRAINING_LOG_FILENAME = "training_log.txt"


def training_loop(
    model,
    train_dataset,
    val_dataset,
    training_config: dict,
    y_train=None,
    y_val=None,
    evaluation_config=None,
    c_index_interval: int = 5,
):
    """
    Run the training loop: train and validate per epoch with early stopping
    and learning-rate reduction. Restores best weights at the end.

    Optionally computes and prints time-dependent C-index on the validation
    set every c_index_interval epochs when y_train, y_val,
    and evaluation_config are provided 

    Parameters
    ----------
    model : keras.Model
        Compiled model (must have .optimizer and .loss).
    train_dataset : tf.data.Dataset
        Batched, shuffled training dataset (x, y). 
    val_dataset : tf.data.Dataset
        Batched validation dataset (x, y).
    training_config : dict
        Keys: epochs, early_stopping_patience, reduce_lr_patience,
        reduce_lr_factor, min_lr.
    y_train : np.ndarray, optional
        Structured training labels for C-index.
    y_val : np.ndarray, optional
        Structured validation labels for C-index.
    evaluation_config : dict, optional
        Keys: p_times, e_times, t_max for time-dependent C-index.
    c_index_interval : int
        Compute and print validation C-index every this many epochs (default 5).

    Returns
    -------
    tuple
        (model with best weights restored, history dict with 'loss' and 'val_loss' lists)
    """
    optimizer = model.optimizer
    loss_fn = model.loss
    history = {"loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    lr_patience_counter = 0
    best_weights = None
    epochs = training_config["epochs"]

    steps_per_epoch = train_dataset.cardinality().numpy()
    steps_per_val = val_dataset.cardinality().numpy()

    # repeat datasets to avoid "end of dataset" error
    train_dataset = train_dataset.repeat()
    val_dataset = val_dataset.repeat()

    compute_c_index = (
        y_train is not None
        and y_val is not None
        and evaluation_config is not None
    )
    if compute_c_index:
        val_features_list = []
        for batch_val_idx, (x_batch, _) in enumerate(val_dataset):
            if batch_val_idx >= steps_per_val:
                break
            val_features_list.append(x_batch.numpy())
        val_features = np.concatenate(val_features_list, axis=0)

    for epoch in range(epochs):
        epoch_losses = []

        print(f"\nEpoch {epoch + 1}/{epochs}")
        progbar = Progbar(steps_per_epoch, stateful_metrics=["loss"])

        for batch_idx, (x_batch, y_batch) in enumerate(train_dataset):
            if batch_idx >= steps_per_epoch:
                break

            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=True)
                loss_value = loss_fn(y_batch, y_pred)

            if tf.math.is_nan(loss_value):
                print(
                    f"WARNING: NaN loss detected at epoch {epoch + 1}, batch {batch_idx + 1}!"
                )
                print(
                    f"  y_pred stats: min={tf.reduce_min(y_pred)}, max={tf.reduce_max(y_pred)}, mean={tf.reduce_mean(y_pred)}"
                )
                print(
                    f"  y_batch stats: min={tf.reduce_min(y_batch)}, max={tf.reduce_max(y_batch)}, mean={tf.reduce_mean(y_batch)}"
                )

            grads = tape.gradient(loss_value, model.trainable_weights)

            # clip gradients to prevent exploding gradients
            grads = [
                tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads
            ]

            if any(
                tf.reduce_any(tf.math.is_nan(g)) for g in grads if g is not None
            ):
                print(
                    f"WARNING: NaN gradients detected at epoch {epoch + 1}, batch {batch_idx + 1}!"
                )
                print("Skipping this batch...")
                continue

            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            loss_val = float(loss_value.numpy())
            epoch_losses.append(loss_val)
            progbar.update(batch_idx + 1, values=[("loss", loss_val)])

        avg_train_loss = np.mean(epoch_losses)
        history["loss"].append(avg_train_loss)

        val_losses = []
        for batch_val_idx, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            if batch_val_idx >= steps_per_val:
                break

            y_pred_val = model(x_batch_val, training=False)
            val_loss_value = loss_fn(y_batch_val, y_pred_val)
            val_losses.append(val_loss_value.numpy())

        avg_val_loss = np.mean(val_losses)
        history["val_loss"].append(avg_val_loss)
        print(
            f"Mean Train Loss: {avg_train_loss:.6f}, Mean Val Loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            lr_patience_counter = 0
            best_weights = model.get_weights()
            print(f"  -> New best validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            lr_patience_counter += 1
            if lr_patience_counter >= training_config["reduce_lr_patience"]:
                old_lr = float(optimizer.learning_rate)
                new_lr = old_lr * training_config["reduce_lr_factor"]
                new_lr = max(new_lr, training_config["min_lr"])
                optimizer.learning_rate.assign(new_lr)
                print(
                    f"  -> Reducing learning rate: {old_lr:.2e} -> {new_lr:.2e}"
                )
                lr_patience_counter = 0
            if patience_counter >= training_config["early_stopping_patience"]:
                print(
                    f"  -> Early stopping triggered (patience: {training_config['early_stopping_patience']})"
                )
                break

        # time-dependent C-index on validation set
        if (
            compute_c_index
            and (epoch + 1) % c_index_interval == 0
        ):
            try:
                c_index_val = calculate_time_dependent_c_index(
                    np.asarray(val_features),
                    y_train,
                    y_val,
                    model,
                    p_times=np.array(evaluation_config["p_times"]),
                    e_times=np.array(evaluation_config["e_times"]),
                    t_max=evaluation_config["t_max"],
                    return_mean=True,
                )

                print(
                    f"Mean time-dependent C-index on validation set at epoch {epoch + 1}: {c_index_val:.6f}"
                )
            except Exception:
                pass

    if best_weights is not None:
        print("Restoring best weights...")
        model.set_weights(best_weights)

    return model, history

