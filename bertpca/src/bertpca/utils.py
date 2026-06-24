"""
Utility functions for setting random seeds and reproducibility.
"""

import os
import random
import numpy as np
import tensorflow as tf


def set_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Parameters
    ----------
    seed : int
        Random seed value (default: 42)
    """
    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # TensorFlow
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    
    # TensorFlow deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
