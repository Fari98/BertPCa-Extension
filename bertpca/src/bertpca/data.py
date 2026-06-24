"""
Data preprocessing utilities for survival analysis.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import Dataset
from typing import List, Tuple


def augment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment a DataFrame by creating truncated versions of each group.
    
    For each group, adds versions containing only the first n rows
    (n = len(group) - 1 to 1). This is useful for survival analysis
    where we want to simulate different observation times.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a named index (group identifier)
    
    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with new index values indicating trimmed versions
    
    Raises
    ------
    ValueError
        If the DataFrame doesn't have a named index
    """
    if df.index.name is None:
        raise ValueError("The DataFrame must have a named index.")

    index_name = df.index.name
    augmented_rows = []

    for idx, group in df.groupby(level=index_name):
        for i in range(len(group) - 1, 0, -1):
            trimmed = group.iloc[:i].copy()
            trimmed.index = [f"{idx}_top{i}"] * i
            augmented_rows.append(trimmed)

    result = pd.concat([df] + augmented_rows)
    return result


def pad_sequences_dynamic(
    df_dynamic: pd.DataFrame,
    seq_length: int
) -> pd.Series:
    """
    Organize dynamic features into sequences and pad them to seq_length.
    
    Parameters
    ----------
    df_dynamic : pd.DataFrame
        DataFrame with dynamic features (multi-indexed by patient ID)
    seq_length : int
        Target sequence length for padding
    
    Returns
    -------
    pd.Series
        Series of padded sequences
    """
    seq = df_dynamic.groupby(level=0).agg(list)
    seq_padded = seq.apply(
        lambda x: pad_sequences(
            np.array(x),
            maxlen=seq_length,
            padding='post',
            truncating='post',
            dtype=np.float32
        ),
        axis=1
    )
    seq_padded.name = 'dynamic_feats'
    return seq_padded


def pad_sequences_static(
    df_static: pd.DataFrame,
    seq_length: int
) -> pd.Series:
    """
    Organize static features into sequences and pad them to seq_length.
    
    Parameters
    ----------
    df_static : pd.DataFrame
        DataFrame with static features (single index per patient)
    seq_length : int
        Target sequence length for padding
    
    Returns
    -------
    pd.Series
        Series of padded sequences
    """
    df_static_arr = df_static.astype(np.float32).apply(np.array, axis=1)
    seq_padded = df_static_arr.apply(
        lambda x: pad_sequences(
            np.reshape(x, (-1, 1)),
            maxlen=seq_length,
            padding='post',
            truncating='post',
            dtype=np.float32
        )
    )
    seq_padded.name = 'static_feats'
    return seq_padded


def repeat_first_column(
    arr: np.ndarray,
    n_dynamic_features: int
) -> np.ndarray:
    """
    Repeat the first column conditionally based on non-zero values.
    
    For columns where the first row has non-zero values in positions
    after the first column, replace those positions with values from
    the first column.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (n_features, seq_length)
    n_dynamic_features : int
        Number of dynamic features (rows to keep unchanged)
    
    Returns
    -------
    np.ndarray
        Modified array with repeated first column values
    """
    result = arr.copy()
    
    # Find where the first row has non-zero values (excluding first column)
    non_zero_mask = (arr[0, 1:] != 0)
    
    # Iterate through the mask and repeat the first column conditionally
    for col_idx, should_replace in enumerate(non_zero_mask, start=1):
        if should_replace:
            # Replace only rows after dynamic features
            result[n_dynamic_features:, col_idx] = result[n_dynamic_features:, 0]
    
    return result


def preprocess_data(
    data: pd.DataFrame,
    static_features: List[str],
    dynamic_features: List[str],
    label: str,
    seq_length: int,
    batch_size: int
) -> Tuple[Dataset, tf.data.Dataset]:
    """
    Preprocess survival data into padded sequences for model training.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with multi-index (patient_id, observation_time)
    static_features : List[str]
        List of static feature column names
    dynamic_features : List[str]
        List of dynamic feature column names
    label : str
        Name of the label column
    seq_length : int
        Target sequence length for padding
    batch_size : int
        Batch size for TensorFlow dataset
    
    Returns
    -------
    Tuple[Dataset, tf.data.Dataset]
        Hugging Face dataset and TensorFlow dataset
    """
    # Get last observation per patient for survival labels
    data_stat_last = data.groupby(level=0).last()
    labels_surv = data_stat_last[['tte', 'label', 'times']].to_numpy(
        dtype=np.float32
    ).tolist()
    
    # Step 1: Pad dynamic features
    data_dyn_padded = pad_sequences_dynamic(
        data[dynamic_features],
        seq_length
    )
    
    # Step 2: Extract static features and pad them
    data_stat = data.groupby(level=0).first()
    data_stat_padded = pad_sequences_static(
        data_stat.loc[:, static_features],
        seq_length
    )
    
    # Step 3: Concatenate padded static and dynamic features
    data_padded = pd.concat([data_stat_padded, data_dyn_padded], axis=1)
    
    # Combine static and dynamic features for each row
    data_padded = data_padded.apply(
        lambda x: np.concatenate([x['dynamic_feats'], x['static_feats']]),
        axis=1
    )
    data_padded.name = 'features'
    
    # Step 4: Apply column repetition logic
    data_padded = data_padded.apply(
        lambda arr: repeat_first_column(arr, n_dynamic_features=len(dynamic_features))
    )
    
    # Step 5: Prepare the data dictionary for Hugging Face dataset
    data_dict = {
        "features": data_padded.tolist(),
        "labels": data_stat[label].tolist(),
        "labels_surv": labels_surv
    }
    
    # Step 6: Create the Hugging Face dataset
    ds = Dataset.from_dict(data_dict)
    
    # Step 7: Create TensorFlow dataset
    data_padded_np = np.array(data_padded.to_list())
    data_stat_label_np = data_stat[label].to_numpy()
    labels_surv_np = np.array(labels_surv)
    
    data_padded_tensor = tf.convert_to_tensor(data_padded_np, dtype=tf.float32)
    labels_surv_tensor = tf.convert_to_tensor(labels_surv_np, dtype=tf.float32)
    
    ds_tf = tf.data.Dataset.from_tensor_slices(
        (data_padded_tensor, labels_surv_tensor)
    )
    ds_tf = ds_tf.shuffle(buffer_size=1024).batch(batch_size)
    
    return ds, ds_tf


def load_and_preprocess_data(
    train_path: str,
    val_path: str,
    test_path: str,
    static_features: List[str],
    dynamic_features: List[str],
    seq_length: int,
    batch_size: int,
    t_max: float,
    augment: bool = True,
    scale_features: bool = True
) -> Tuple[Dataset, Dataset, Dataset, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess survival data from CSV files.
    
    Parameters
    ----------
    train_path : str
        Path to training CSV file
    val_path : str
        Path to validation CSV file
    test_path : str
        Path to test CSV file
    static_features : List[str]
        List of static feature column names
    dynamic_features : List[str]
        List of dynamic feature column names
    seq_length : int
        Target sequence length for padding
    batch_size : int
        Batch size for TensorFlow dataset
    t_max : float
        Maximum follow-up time for scaling (default: 15*365 days)
    augment : bool
        Whether to augment training data
    scale_features : bool
        Whether to apply min-max scaling
    
    Returns
    -------
    Tuple containing:
        - train_ds, val_ds, test_ds: Hugging Face datasets
        - y_train_struct, y_val_struct, y_test_struct: Structured arrays for survival labels
    """
    # Load data
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    
    # Set indexes
    train.set_index('id', inplace=True)
    val.set_index('id', inplace=True)
    test.set_index('id', inplace=True)
    
    # Convert to float
    train_s = train.astype(float)
    val_s = val.astype(float)
    test_s = test.astype(float)

    
    # Min-max scaling
    if scale_features:

        features_to_scale = static_features + dynamic_features
        features_to_scale.remove('times') # exclude times from scaling since it will be scaled separately

        # Find scaling parameters from training data
        train_max = train_s.loc[:, features_to_scale].max()
        train_min = train_s.loc[:, features_to_scale].min()
        
        # Apply scaling
        for df in [train_s, val_s, test_s]:
            df.loc[:, features_to_scale] = (
                (df.loc[:, features_to_scale] - train_min) / (train_max - train_min)
            )
            df.loc[:, ['tte', 'times']] = df.loc[:, ['tte', 'times']] / t_max
    
    # Create structured arrays for survival labels
    dt = np.dtype([('Status', '?'), ('Survival_in_days', '<f8')])
    
    train_last = train_s.groupby(level=0).last()
    val_last = val_s.groupby(level=0).last()
    test_last = test_s.groupby(level=0).last()
    
    y_train_struct = np.array(
        list(zip(train_last.label.values, train_last.tte.values)),
        dtype=dt
    )
    y_val_struct = np.array(
        list(zip(val_last.label.values, val_last.tte.values)),
        dtype=dt
    )
    y_test_struct = np.array(
        list(zip(test_last.label.values, test_last.tte.values)),
        dtype=dt
    )
    
    # Augment training data if requested
    if augment:
        train_s = augment_dataframe(train_s)
    
    # Preprocess data
    train_ds, _ = preprocess_data(
        train_s, static_features, dynamic_features, 'label',
        seq_length, batch_size
    )
    val_ds, _ = preprocess_data(
        val_s, static_features, dynamic_features, 'label',
        seq_length, batch_size
    )
    test_ds, _ = preprocess_data(
        test_s, static_features, dynamic_features, 'label',
        seq_length, batch_size
    )
    
    return (
        train_ds, val_ds, test_ds,
        y_train_struct, y_val_struct, y_test_struct
    )
