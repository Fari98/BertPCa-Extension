# -*- coding: utf-8 -*-
"""
Loss functions for survival analysis models.
"""

import tensorflow as tf
try:
    from keras.saving import register_keras_serializable   # TF >= 2.12 standalone Keras
except (ImportError, AttributeError):
    try:
        register_keras_serializable = tf.keras.utils.register_keras_serializable  # TF 2.6-2.11
    except AttributeError:
        # No-op fallback: SavedModel format (TF <= 2.10) embeds the loss in the TF graph
        def register_keras_serializable(package="Custom", name=None):
            def decorator(fn):
                return fn
            return decorator


@register_keras_serializable()
def weibull_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Weibull negative log-likelihood loss function for survival analysis.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Tensor with shape (batch_size, 3) -> [time, event, t_last]
        - time: survival time (or censoring time)
        - event: 1 if event occurred, 0 if censored
        - t_last: last observation time
    y_pred : tf.Tensor
        Tensor with shape (batch_size, 2) -> [alpha, beta]
        - alpha: Weibull scale parameter (positive)
        - beta: Weibull shape parameter (positive)
    
    Returns
    -------
    tf.Tensor
        Negative log-likelihood loss value
    """
    EPSILON = 1e-5
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    
    t = y_true[:, 0]  # Survival/censoring time
    event = y_true[:, 1]  # 1 if event happened, 0 if censored
    t_last = y_true[:, 2]  # Last observation time

        # Clamp t_new to be strictly positive (Weibull requires t > 0)
    t_new = tf.maximum(t - t_last, EPSILON)

    # Lower-bound alpha and beta for numerical stability
    alpha = tf.maximum(y_pred[:, 0] + 1, EPSILON)  # Weibull scale parameter
    beta = tf.maximum(y_pred[:, 1] + 1, EPSILON)  # Weibull shape parameter

    # Compute Weibull components with numerical safeguards
    ratio = t_new / alpha
    ratio_clipped = tf.clip_by_value(ratio, EPSILON, 1e6)  # Prevent extreme values
    
    # Log likelihood for observed events
    hazard = (beta / alpha) * (ratio_clipped ** (beta - 1))
    s = tf.exp(-(ratio_clipped ** beta))
    
    # Clip hazard * s to prevent log(0) -> -inf
    hazard_s = tf.clip_by_value(hazard * s, EPSILON, 1e10)
    log_likelihood_event = tf.math.log(hazard_s)

    # Log likelihood for censored events
    s_clipped = tf.clip_by_value(s, EPSILON, 1.0)
    log_likelihood_censored = tf.math.log(s_clipped)

    # Combine log-likelihoods with clipping to prevent -inf
    log_likelihood = event * log_likelihood_event + (1 - event) * log_likelihood_censored
    log_likelihood = tf.clip_by_value(log_likelihood, -1e6, 1e6)  # Prevent extreme log-likelihoods
    
    loss = -tf.reduce_mean(log_likelihood)

    return loss


def coxph_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Cox proportional hazards negative log partial likelihood loss.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Tensor with shape (batch_size, 2) -> [time, event]
        - time: survival time
        - event: event indicator (1=event, 0=censored)
    y_pred : tf.Tensor
        Tensor with shape (batch_size, 1) -> risk score
    
    Returns
    -------
    tf.Tensor
        Negative log partial likelihood loss
    """
    time = y_true[:, 0]
    event = y_true[:, 1]

    # Sort by time in descending order
    sorted_indices = tf.argsort(time, direction='DESCENDING')
    y_pred_sorted = tf.gather(y_pred, sorted_indices)
    event_sorted = tf.gather(event, sorted_indices)

    # Compute the log partial likelihood for Cox PH
    exp_y_pred = tf.exp(y_pred_sorted)
    cumulative_sum = tf.cumsum(exp_y_pred, reverse=True)
    cumulative_sum = tf.maximum(cumulative_sum, 1e-8)

    log_partial_likelihood = y_pred_sorted - tf.math.log(cumulative_sum)

    # Only keep terms where event == 1
    observed_likelihood = tf.boolean_mask(log_partial_likelihood, event_sorted)
    num_events = tf.reduce_sum(event_sorted)

    # Compute final loss, handling zero-event case
    neg_log_likelihood = tf.cond(
        num_events > 0,
        lambda: -tf.reduce_sum(observed_likelihood) / (num_events + 1e-8),
        lambda: tf.constant(0.0)  # Return zero loss if no events
    )

    return neg_log_likelihood


def ranking_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Pairwise ranking loss for survival analysis.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Tensor of shape (batch_size, 3) -> [survival time, event status, t_last]
    y_pred : tf.Tensor
        Tensor of shape (batch_size, 2) -> [alpha (scale), beta (shape)]
    
    Returns
    -------
    tf.Tensor
        Scalar loss value
    """
    SIGMA = 1
    EPSILON = 1e-5

    t = y_true[:, 0]  # Survival/censoring time
    event = y_true[:, 1]  # Event indicator (1 = event, 0 = censored)
    t_last = y_true[:, 2]  # Last observed time

    alpha = y_pred[:, 0] + 1 + EPSILON  # Weibull scale parameter
    beta = y_pred[:, 1] + 1  # Weibull shape parameter

    # Compute all pairwise differences using broadcasting
    s_i = tf.expand_dims(t, axis=1) - tf.expand_dims(t_last, axis=1)
    s_j = tf.transpose(s_i)

    # Create mask: Compare t[i] < t[j] and event[i] == 1
    mask = tf.logical_and(
        s_i < s_j,
        tf.expand_dims(event, axis=1) == 1
    )

    s_i = tf.cast(s_i, tf.float32)
    t_last = tf.cast(t_last, tf.float32)
    beta = tf.cast(beta, tf.float32)

    # Compute risk scores for all patients
    S_num = tf.exp(-(((s_i + tf.expand_dims(t_last, axis=1)) / alpha) ** beta))
    S_den = tf.exp(-((tf.expand_dims(t_last, axis=1) / alpha) ** beta))
    risk_i = 1 - S_num / (S_den + EPSILON)

    S_num_j = tf.exp(-(((s_i + tf.expand_dims(t_last, axis=0)) / alpha) ** beta))
    S_den_j = tf.exp(-((tf.expand_dims(t_last, axis=0) / alpha) ** beta))
    risk_j = 1 - S_num_j / (S_den_j + EPSILON)

    # Compute pairwise ranking loss term
    risk_diff = risk_i - risk_j
    risk_diff_clipped = tf.clip_by_value(risk_diff, -10, 10)
    nu = tf.exp(-risk_diff_clipped / SIGMA)

    # Apply mask to only keep valid pairs
    loss = tf.reduce_sum(tf.where(mask, nu, tf.zeros_like(nu)))

    return loss


def survival_contrastive_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Contrastive loss for survival data.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Tensor with shape (batch_size, 2) -> [times, events]
    y_pred : tf.Tensor
        Tensor with shape (batch_size, embedding_dim) -> embeddings
    
    Returns
    -------
    tf.Tensor
        Scalar survival contrastive loss
    """
    times = y_true[:, 0]
    events = y_true[:, 1]
    margin = 1 / 15
    embeddings = y_pred

    batch_size = tf.shape(times)[0]

    # Expand dims for pairwise comparison
    T_i = tf.expand_dims(times, 0)
    T_j = tf.expand_dims(times, 1)

    E_i = tf.expand_dims(events, 0)
    E_j = tf.expand_dims(events, 1)

    # Pairwise distance between embeddings
    emb_i = tf.expand_dims(embeddings, 0)
    emb_j = tf.expand_dims(embeddings, 1)
    pairwise_dist = tf.norm(emb_i - emb_j, axis=-1)

    # Similar if both had events and survival times are close
    time_diff = tf.abs(T_i - T_j)
    similar_mask = tf.cast(
        tf.logical_and(
            tf.equal(E_i, 1.0) & tf.equal(E_j, 1.0),
            tf.less(time_diff, margin)
        ),
        tf.float32
    )

    # Dissimilar if large time difference and at least one event
    dissimilar_mask = tf.cast(
        (time_diff > margin) & ((E_i + E_j) > 0),
        tf.float32
    )

    # Compute losses
    loss_similar = similar_mask * tf.square(pairwise_dist)
    loss_dissimilar = dissimilar_mask * tf.square(
        tf.maximum(margin - pairwise_dist, 0.0)
    )

    total_loss = 0.5 * tf.reduce_mean(loss_similar + loss_dissimilar)
    return total_loss
