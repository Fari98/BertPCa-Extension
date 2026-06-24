"""
BertPCa: A Transformer-based Survival Analysis Model for Prostate Cancer.

This package provides tools for building, training, and evaluating BertPCa, a survival
analysis model using a transformer architecture and Weibull distribution.
"""

__version__ = "0.1.0"

from .models import build_bert_pca
from .data import preprocess_data, augment_dataframe, load_and_preprocess_data
from .evaluation import calculate_time_dependent_c_index, CIndexCallback
from .loss import weibull_loss
from .metrics import weighted_c_index, weighted_brier_score
from .train import training_loop
from .utils import set_seeds

__all__ = [
    "build_bert_pca",
    "preprocess_data",
    "augment_dataframe",
    "load_and_preprocess_data",
    "calculate_time_dependent_c_index",
    "CIndexCallback",
    "weibull_loss",
    "weighted_c_index",
    "weighted_brier_score",
    "training_loop",
    "set_seeds",
]
