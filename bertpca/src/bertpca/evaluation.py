"""
Evaluation metrics and utilities for survival analysis models.
"""

import numpy as np
import tensorflow as tf
from typing import Union, Tuple, Optional
from .metrics import weighted_c_index


def calculate_time_dependent_c_index(
    features: np.ndarray,
    y_train: Union[np.ndarray, np.recarray],
    y_test: Union[np.ndarray, np.recarray],
    model: tf.keras.Model,
    p_times: np.ndarray,
    e_times: np.ndarray,
    t_max: float,
    return_mean: bool = False,
    epsilon: float = 1e-5
):
    """
    Calculate time-dependent C-index for survival analysis.
    
    
    Parameters
    ----------
    features : np.ndarray
        Input features array of shape (n_samples, n_features, seq_length)
    y_train : Union[np.ndarray, np.recarray]
        Training labels. Can be:
        - Structured array with dtype=[('Status', '?'), ('Survival_in_days', '<f8')]
        - Array of shape (n_samples, 3) with [tte, label, times]
    y_test : Union[np.ndarray, np.recarray]
        Test labels in same format as y_train
    model : tf.keras.Model
        Trained model that outputs Weibull parameters [alpha, beta]
    p_times : np.ndarray
        Prediction times (in days). Required.
    e_times : np.ndarray
        Evaluation times (in days). Required.
    t_max : float
        Maximum follow-up time for scaling (default: 15*365 days)
    return_mean : bool
        If True, return mean C-index. If False, return full 2D array.
    epsilon : float
        Small value to prevent numerical issues
    
    Returns
    -------
    Union[np.ndarray, float]
        If return_mean=False: 2D array of C-indices of shape (len(p_times), len(e_times))
        If return_mean=True: Mean C-index value
    """
    # Normalize labels to structured array format
    y_train_struct = _normalize_labels(y_train)
    y_test_struct = _normalize_labels(y_test)
    
    # Scale times by t_max
    p_times_scaled = p_times / t_max
    e_times_scaled = e_times / t_max

    
    c_index = np.zeros((len(p_times_scaled), len(e_times_scaled)))
    
    # Extract time arrays from features (assuming times are in first feature)
    features_array = np.array(features.copy())
    times_array = features_array[:, 0, :]
    
    # Convert structured arrays to column-stacked format for indexing
    y_train_cols = np.column_stack((
        y_train_struct['Status'].astype(float),
        y_train_struct['Survival_in_days']
    ))
    y_test_cols = np.column_stack((
        y_test_struct['Status'].astype(float),
        y_test_struct['Survival_in_days']
    ))
    
    for i, p_time in enumerate(p_times_scaled):
        p_time_raw = p_times[i]  # raw days — used for survival-time comparisons

        # Mask features to only include observations up to prediction time
        # times_array is pre-scaled by t_max, so compare against scaled p_time
        mask = (times_array <= p_time)
        features_masked = mask[:, np.newaxis, :] * features_array

        # Filter test/train samples that survive beyond prediction time (in days)
        test_survival_times = y_test_struct['Survival_in_days']
        test_time_mask = test_survival_times > p_time_raw

        if not np.any(test_time_mask):
            continue

        features_test = features_masked[test_time_mask, :, :]

        train_survival_times = y_train_struct['Survival_in_days']
        train_time_mask = train_survival_times > p_time_raw

        if not np.any(train_time_mask):
            continue

        # Get model predictions
        y_pred = model(features_test)
        alpha = y_pred[:, 0] + 1 + epsilon
        beta = y_pred[:, 1] + 1

        # Calculate C-index for each evaluation time
        for j, e_time in enumerate(e_times_scaled):
            e_time_raw = e_times[j]  # raw days — passed to weighted_c_index

            # Weibull hazard at scaled e_time (model operates in scaled units)
            risks = (beta / alpha) * (((e_time) / alpha) ** (beta - 1))

            # Residual survival times in days (subtract raw p_time)
            train_times = y_train_cols[train_time_mask, 1] - p_time_raw
            train_events = y_train_cols[train_time_mask, 0]
            test_times = y_test_cols[test_time_mask, 1] - p_time_raw
            test_events = y_test_cols[test_time_mask, 0]

            c_index[i, j] = weighted_c_index(
                train_times,
                train_events,
                risks,
                test_times,
                test_events,
                e_time_raw,
            )

    
    return np.mean(c_index) if return_mean else c_index


def _normalize_labels(y: Union[np.ndarray, np.recarray]) -> np.recarray:
    """
    Normalize labels to structured array format.
    
    Handles labels_surv format: [tte, label, times] where:
    - tte: time-to-event (survival time)
    - label: event indicator (1=event, 0=censored)
    - times: observation times
    
    Parameters
    ----------
    y : Union[np.ndarray, np.recarray]
        Labels in various formats:
        - Structured array with dtype=[('Status', '?'), ('Survival_in_days', '<f8')]
        - Array of shape (n_samples, 3) with [tte, label, times] (labels_surv format)
        - Array of shape (n_samples, 2) with [label, tte] or [tte, label]
    
    Returns
    -------
    np.recarray
        Structured array with dtype=[('Status', '?'), ('Survival_in_days', '<f8')]
        Format: (Status, Survival_in_days) where Status is bool and Survival_in_days is float
    """
    dt = np.dtype([('Status', '?'), ('Survival_in_days', '<f8')])
    
    # Check if already structured array
    if isinstance(y, np.recarray) or (hasattr(y, 'dtype') and y.dtype.names):
        return np.asarray(y, dtype=dt)
    
    # Handle array format
    y_array = np.array(y)
    
    if len(y_array.shape) == 1:
        # Single sample, convert to 2D
        y_array = y_array.reshape(1, -1)
    
    if y_array.shape[1] >= 3:
        # Format: [tte, label, times] - labels_surv format
        # Convert to (label, tte) for (Status, Survival_in_days)
        status = y_array[:, 1].astype(bool)
        survival_time = y_array[:, 0].astype(float)
    elif y_array.shape[1] >= 2:
        # Format: [label, tte] or [tte, label]
        # Try to infer: if first column is mostly 0/1, it's status
        if np.all(np.isin(y_array[:, 0], [0, 1])):
            # Format: [label, tte]
            status = y_array[:, 0].astype(bool)
            survival_time = y_array[:, 1].astype(float)
        else:
            # Format: [tte, label] - less common but possible
            status = y_array[:, 1].astype(bool)
            survival_time = y_array[:, 0].astype(float)
    else:
        raise ValueError(f"Unexpected label shape: {y_array.shape}")
    
    return np.array(list(zip(status, survival_time)), dtype=dt)


# Custom callback to evaluate the C-index, useful when using model.fit()
class CIndexCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to evaluate the C-index on the validation set after each epoch.
    
    This callback computes the time-dependent C-index during training to monitor
    model performance on survival prediction tasks.
    """
    
    def __init__(
        self,
        val_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        p_times: np.ndarray,
        e_times: np.ndarray,
        t_max: float,
        verbose: int = 0
    ):
        """
        Initialize the C-index callback.
        
        Parameters
        ----------
        val_data : Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple of (val_features, val_labels, train_labels)
        p_times : np.ndarray
            Prediction times (in days). Required.
        e_times : np.ndarray
            Evaluation times (in days). Required.
        t_max : float
            Maximum follow-up time for scaling (default: 15*365 days)
        verbose : int
            Verbosity level (0=silent, 1=print)
        """
        self.val_data = val_data
        self.t_max = t_max
        self.p_times = p_times
        self.e_times = e_times
        self.verbose = verbose
        self.val_c_index_history = []
    
    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """
        Evaluate C-index after each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        logs : Optional[dict]
            Dictionary of metrics
        """
        val_features, val_labels, train_labels = self.val_data
        
        # Calculate the C-index on the entire validation set
        c_index_val = calculate_time_dependent_c_index(
            val_features,
            train_labels,
            val_labels,
            self.model,
            p_times=self.p_times,
            e_times=self.e_times,
            t_max=self.t_max,
            return_mean=True
        )
        
        self.val_c_index_history.append(c_index_val)
        
        # Store the C-index for early stopping to monitor
        if logs is not None:
            logs['val_c_index'] = c_index_val
        
        if self.verbose > 0:
            print(f"Epoch {epoch + 1}: val_c_index = {c_index_val:.4f}")
    
    def get_history(self) -> list:
        """Return the history of C-index values."""
        return self.val_c_index_history
