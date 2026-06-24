"""
Dynamic-DeepHit (DDH) baseline — TensorFlow/Keras implementation.

Architecture (simplified DDH, Changhee Lee et al. 2019):
  - PSA sequence (batch, seq_len) → Masking → LSTM(64) → (batch, 64)
  - Static features (batch, n_static) → Dense(64, relu) → (batch, 64)
  - Concatenate → Dense(128, relu) → Dropout(0.3) → Dense(n_bins, softmax)

Loss: discrete-time log-likelihood
  - Event at bin k: log(p_k)
  - Censored at bin k: log(1 - sum(p_1 ... p_k))

Time is discretised into n_bins equal-width bins spanning [0, t_max].
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Tuple


def _discretise_times(times: np.ndarray, n_bins: int, t_max: float) -> np.ndarray:
    """Map continuous times to bin indices [0, n_bins-1]."""
    bins = np.clip((times / t_max * n_bins).astype(int), 0, n_bins - 1)
    return bins


def _pad_psa_sequences(
    df_long: pd.DataFrame,
    seq_length: int,
    t_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build padded PSA matrix and patient-level static arrays from long-format df.

    Returns
    -------
    psa_padded : (n_patients, seq_length)  — zero-padded PSA sequences
    patient_ids : (n_patients,)
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad

    patient_ids = df_long.index.unique().tolist()
    seqs = []
    for pid in patient_ids:
        grp = df_long.loc[[pid]] if not isinstance(df_long.loc[pid], pd.Series) else df_long.loc[[pid]]
        psa_vals = grp["psa"].values.astype(np.float32)
        seqs.append(psa_vals.tolist())

    psa_padded = keras_pad(
        seqs,
        maxlen=seq_length,
        padding="post",
        truncating="post",
        dtype=np.float32,
        value=0.0,
    )
    return psa_padded, np.array(patient_ids)


def build_ddh(
    n_static: int,
    seq_length: int,
    n_bins: int,
    lstm_units: int = 64,
    dense_units: int = 128,
    dropout: float = 0.3,
) -> keras.Model:
    """Build the DDH Keras model."""
    psa_input = keras.Input(shape=(seq_length,), name="psa_seq")
    static_input = keras.Input(shape=(n_static,), name="static")

    # PSA encoder
    x_psa = layers.Reshape((seq_length, 1))(psa_input)
    x_psa = layers.Masking(mask_value=0.0)(x_psa)
    x_psa = layers.LSTM(lstm_units, name="lstm")(x_psa)

    # Static encoder
    x_static = layers.Dense(lstm_units, activation="relu", name="static_enc")(static_input)

    # Joint
    x = layers.Concatenate(name="concat")([x_psa, x_static])
    x = layers.Dense(dense_units, activation="relu", name="joint_dense")(x)
    x = layers.Dropout(dropout, name="joint_dropout")(x)
    output = layers.Dense(n_bins, activation="softmax", name="hazard")(x)

    return keras.Model(inputs=[psa_input, static_input], outputs=output, name="DDH")


def ddh_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Discrete-time survival log-likelihood.

    y_true: (batch, 2) — [bin_index (int), event (0/1)]
    y_pred: (batch, n_bins) — softmax probabilities
    """
    eps = 1e-7
    bin_idx = tf.cast(y_true[:, 0], tf.int32)
    event = tf.cast(y_true[:, 1], tf.float32)
    n_bins = tf.shape(y_pred)[1]

    # Probability mass at the event bin
    idx_oh = tf.one_hot(bin_idx, n_bins)
    p_event = tf.reduce_sum(y_pred * idx_oh, axis=1)

    # Cumulative probability up to and including the event bin
    bin_range = tf.cast(tf.range(n_bins), tf.float32)
    cum_mask = tf.cast(
        bin_range[tf.newaxis, :] <= tf.cast(bin_idx, tf.float32)[:, tf.newaxis],
        tf.float32,
    )
    p_cum = tf.reduce_sum(y_pred * cum_mask, axis=1)

    ll_event = tf.math.log(tf.clip_by_value(p_event, eps, 1.0))
    ll_cens = tf.math.log(tf.clip_by_value(1.0 - p_cum, eps, 1.0))

    ll = event * ll_event + (1.0 - event) * ll_cens
    return -tf.reduce_mean(ll)


def prepare_ddh_data(
    df_long: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int,
    n_bins: int,
    t_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare inputs and labels for DDH training.

    Returns
    -------
    psa_padded : (n_patients, seq_length)
    static_feats : (n_patients, n_static)
    labels : (n_patients, 2) — [bin_index, event]
    """
    psa_padded, patient_ids = _pad_psa_sequences(df_long, seq_length, t_max)

    # One row per patient
    pt = df_long.groupby(level=0)[feature_cols + ["tte", "label"]].first()
    pt = pt.loc[patient_ids]

    static_feats = pt[feature_cols].fillna(pt[feature_cols].median()).values.astype(np.float32)

    bin_idx = _discretise_times(pt["tte"].values, n_bins, t_max)
    labels = np.stack([bin_idx, pt["label"].values.astype(int)], axis=1).astype(np.float32)

    return psa_padded, static_feats, labels


def train_ddh(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int = 16,
    n_bins: int = 36,
    t_max: float = 730.0,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 10,
    seed: int = 42,
) -> keras.Model:
    """Train the DDH model and return the best model (by val loss)."""
    tf.random.set_seed(seed)

    psa_tr, stat_tr, lab_tr = prepare_ddh_data(train_df, feature_cols, seq_length, n_bins, t_max)
    psa_val, stat_val, lab_val = prepare_ddh_data(val_df, feature_cols, seq_length, n_bins, t_max)

    # Normalise static features with train stats
    stat_mean = np.nanmean(stat_tr, axis=0)
    stat_std = np.nanstd(stat_tr, axis=0) + 1e-8
    stat_tr = (stat_tr - stat_mean) / stat_std
    stat_val = (stat_val - stat_mean) / stat_std

    # Normalise PSA by max train PSA
    psa_max = np.nanmax(psa_tr) + 1e-8
    psa_tr = psa_tr / psa_max
    psa_val = psa_val / psa_max

    n_static = stat_tr.shape[1]
    model = build_ddh(n_static=n_static, seq_length=seq_length, n_bins=n_bins)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=ddh_loss,
    )

    best_val_loss = np.inf
    best_weights = None
    patience_counter = 0

    ds_train = (
        tf.data.Dataset.from_tensor_slices(((psa_tr, stat_tr), lab_tr))
        .shuffle(1024, seed=seed)
        .batch(batch_size)
    )
    ds_val = (
        tf.data.Dataset.from_tensor_slices(((psa_val, stat_val), lab_val))
        .batch(batch_size)
    )

    # Store normalisation params on model for later use
    model._ddh_stat_mean = stat_mean
    model._ddh_stat_std = stat_std
    model._ddh_psa_max = psa_max
    model._ddh_n_bins = n_bins
    model._ddh_t_max = t_max
    model._ddh_seq_length = seq_length

    for epoch in range(epochs):
        train_losses = []
        for (x_psa, x_stat), y in ds_train:
            with tf.GradientTape() as tape:
                y_pred = model([x_psa, x_stat], training=True)
                loss = ddh_loss(y, y_pred)
            grads = tape.gradient(loss, model.trainable_weights)
            model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_losses.append(float(loss))

        val_losses = []
        for (x_psa, x_stat), y in ds_val:
            y_pred = model([x_psa, x_stat], training=False)
            val_losses.append(float(ddh_loss(y, y_pred)))

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        if (epoch + 1) % 10 == 0:
            print(f"  DDH Epoch {epoch+1}/{epochs} — train: {avg_train:.4f}, val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_weights = model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  DDH early stopping at epoch {epoch+1}")
                break

    if best_weights is not None:
        model.set_weights(best_weights)
    return model


def evaluate_ddh(
    model: keras.Model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    p_times: np.ndarray,
    e_times: np.ndarray,
) -> np.ndarray:
    """
    Evaluate DDH using the weighted time-dependent C-index.

    At each p_time, mask PSA to only include readings up to p_time, then predict
    P(T ≤ e_time | T > p_time) as the risk score.

    Returns
    -------
    np.ndarray of shape (len(p_times), len(e_times))
    """
    from bertpca.metrics import weighted_c_index

    seq_length = model._ddh_seq_length
    n_bins = model._ddh_n_bins
    t_max = model._ddh_t_max
    stat_mean = model._ddh_stat_mean
    stat_std = model._ddh_stat_std
    psa_max = model._ddh_psa_max

    train_pt = train_df.groupby(level=0)[feature_cols + ["tte", "label"]].first()
    test_pt = test_df.groupby(level=0)[feature_cols + ["tte", "label"]].first()
    train_med = train_pt[feature_cols].median()

    stat_test = test_pt[feature_cols].fillna(train_med).values.astype(np.float32)
    stat_test = (stat_test - stat_mean) / stat_std

    train_times = train_pt["tte"].values
    train_events = train_pt["label"].values
    test_times = test_pt["tte"].values
    test_events = test_pt["label"].values

    c_index = np.zeros((len(p_times), len(e_times)))

    for i, p_time in enumerate(p_times):
        # Build PSA sequences masked to p_time
        psa_masked = []
        for pid in test_pt.index:
            grp = test_df.loc[[pid]] if not isinstance(test_df.loc[pid], pd.Series) else test_df.loc[[pid]]
            psa_sub = grp[grp["times"] <= p_time]["psa"].values.astype(np.float32)
            psa_masked.append(psa_sub.tolist())

        from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad
        psa_arr = keras_pad(psa_masked, maxlen=seq_length, padding="post",
                            truncating="post", dtype=np.float32, value=0.0)
        psa_arr = psa_arr / psa_max

        # Filter test patients who survive beyond p_time (all in days)
        test_time_mask = test_times > p_time
        train_time_mask = train_times > p_time
        if not np.any(test_time_mask) or not np.any(train_time_mask):
            continue

        psa_sub_arr = psa_arr[test_time_mask]
        stat_sub_arr = stat_test[test_time_mask]

        # Predict softmax probabilities
        probs = model.predict([psa_sub_arr, stat_sub_arr], verbose=0)  # (n_test, n_bins)

        for j, e_time in enumerate(e_times):
            e_bin = min(int(e_time / t_max * n_bins), n_bins - 1)
            p_bin = min(int(p_time / t_max * n_bins), n_bins - 1)
            # P(T ≤ e_time) - P(T ≤ p_time)  / P(T > p_time)
            cum_e = probs[:, :e_bin + 1].sum(axis=1)
            cum_p = probs[:, :p_bin + 1].sum(axis=1)
            denom = np.clip(1.0 - cum_p, 1e-7, 1.0)
            risks = (cum_e - cum_p) / denom

            sub_test_times = test_times[test_time_mask] - p_time
            sub_test_events = test_events[test_time_mask]
            sub_train_times = train_times[train_time_mask] - p_time
            sub_train_events = train_events[train_time_mask]

            c_index[i, j] = weighted_c_index(
                sub_train_times, sub_train_events,
                risks,
                sub_test_times, sub_test_events,
                e_time - p_time,
            )

    return c_index