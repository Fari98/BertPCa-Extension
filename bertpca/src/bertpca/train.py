"""
Training logic for BertPCa survival analysis model.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

from .evaluation import calculate_time_dependent_c_index
from .loss import ranking_loss as _ranking_loss

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
    gamma: float = 0.0,
    autosave_path: str = None,
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

    epochs       = training_config["epochs"]
    es_patience  = training_config["early_stopping_patience"]
    lr_patience  = training_config["reduce_lr_patience"]
    lr_factor    = training_config["reduce_lr_factor"]
    min_lr       = training_config["min_lr"]

    # C-index patience: number of C-index evaluations without improvement.
    # Keeps total wait comparable to es_patience epochs.
    cindex_patience_limit = max(2, es_patience // c_index_interval)

    best_val_loss   = float("inf")
    best_val_cindex = -float("inf")
    best_weights    = None
    cindex_ever     = False   # True once the first C-index has been computed

    lr_patience_ctr    = 0
    nll_patience_ctr   = 0   # fallback early-stop when C-index unavailable
    cindex_patience_ctr = 0

    steps_per_epoch = train_dataset.cardinality().numpy()
    steps_per_val   = val_dataset.cardinality().numpy()

    train_dataset = train_dataset.repeat()
    val_dataset   = val_dataset.repeat()

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
                if gamma > 0.0:
                    rl = _ranking_loss(y_batch, y_pred)
                    if not (tf.math.is_nan(rl) or tf.math.is_inf(rl)):
                        loss_value = (loss_value
                                      + tf.cast(gamma, loss_value.dtype)
                                      * tf.cast(rl, loss_value.dtype))

            if tf.math.is_nan(loss_value) or tf.math.is_inf(loss_value):
                print(
                    f"WARNING: NaN/Inf loss detected at epoch {epoch + 1}, batch {batch_idx + 1}!"
                )
                print(
                    f"  y_pred stats: min={tf.reduce_min(y_pred)}, max={tf.reduce_max(y_pred)}, mean={tf.reduce_mean(y_pred)}"
                )
                print(
                    f"  y_batch stats: min={tf.reduce_min(y_batch)}, max={tf.reduce_max(y_batch)}, mean={tf.reduce_mean(y_batch)}"
                )
                continue

            grads = tape.gradient(loss_value, model.trainable_weights)

            # Global norm clipping: keeps all gradients proportional while
            # bounding the total update magnitude (better than per-tensor clip
            # when the ranking+weibull combination produces uneven gradient scales).
            grads, _ = tf.clip_by_global_norm(
                [g if g is not None else tf.zeros_like(w)
                 for g, w in zip(grads, model.trainable_weights)],
                1.0,
            )

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

        # --- LR scheduler (always on NLL) ---
        if avg_val_loss < best_val_loss:
            best_val_loss   = avg_val_loss
            lr_patience_ctr = 0
            nll_patience_ctr = 0
            # Before any C-index is computed, best weights = best NLL
            if not cindex_ever:
                best_weights = model.get_weights()
                print(f"  -> New best val NLL: {best_val_loss:.6f}")
                if autosave_path:
                    try:
                        model.save(autosave_path)
                    except Exception as _e:
                        print(f"  -> Auto-save failed: {_e}")
        else:
            lr_patience_ctr  += 1
            nll_patience_ctr += 1
            if lr_patience_ctr >= lr_patience:
                old_lr = float(optimizer.learning_rate)
                new_lr = max(old_lr * lr_factor, min_lr)
                optimizer.learning_rate.assign(new_lr)
                print(f"  -> Reducing learning rate: {old_lr:.2e} -> {new_lr:.2e}")
                lr_patience_ctr = 0

        # NLL early-stop: used only when C-index is never available
        if not compute_c_index and nll_patience_ctr >= es_patience:
            print(f"  -> Early stopping (NLL, patience={es_patience})")
            break

        # --- Val C-index: model selection + early stopping ---
        if compute_c_index and (epoch + 1) % c_index_interval == 0:
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
                cindex_ever = True
                print(f"Val C-index @ ep {epoch + 1}: {c_index_val:.4f}")

                if c_index_val > best_val_cindex:
                    best_val_cindex     = c_index_val
                    cindex_patience_ctr = 0
                    best_weights        = model.get_weights()
                    print(f"  -> New best val C-index: {best_val_cindex:.4f}")
                    if autosave_path:
                        try:
                            model.save(autosave_path)
                        except Exception as _e:
                            print(f"  -> Auto-save failed: {_e}")
                else:
                    cindex_patience_ctr += 1
                    print(
                        f"  -> C-index patience: {cindex_patience_ctr}/{cindex_patience_limit}"
                    )
                    if cindex_patience_ctr >= cindex_patience_limit:
                        print(
                            f"  -> Early stopping (C-index, "
                            f"{cindex_patience_limit} evals ≈ "
                            f"{cindex_patience_limit * c_index_interval} epochs)"
                        )
                        break
            except Exception as _ce:
                print(f"  -> C-index eval failed: {_ce}")

    if best_weights is not None:
        print("Restoring best weights...")
        model.set_weights(best_weights)

    return model, history

