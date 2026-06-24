"""
Static survival baselines: Cox Proportional Hazards and Random Survival Forest.

Both models use only pre-surgical / static features (no PSA trajectory).
Trained on the train split; evaluated on the test split.

Requires: scikit-survival  (pip install scikit-survival)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.preprocessing import OneHotEncoder as SksuvEncoder
    _SKSURV_AVAILABLE = True
except ImportError:
    _SKSURV_AVAILABLE = False


def _check_sksurv():
    if not _SKSURV_AVAILABLE:
        raise ImportError(
            "scikit-survival is required for CoxPH/RSF baselines.\n"
            "Install with: pip install scikit-survival"
        )


def _build_survival_array(df: pd.DataFrame) -> np.ndarray:
    """Build structured array (event: bool, time: float) from tte/label columns."""
    dt = np.dtype([("event", bool), ("time", float)])
    return np.array(
        list(zip(df["label"].astype(bool), df["tte"].astype(float))),
        dtype=dt,
    )


def _get_patient_level(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Return one row per patient (first occurrence of each static feature)."""
    return df.groupby(level=0)[feature_cols + ["tte", "label"]].first()


def train_coxph(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    alpha: float = 0.1,
    random_state: int = 42,
) -> "CoxPHSurvivalAnalysis":
    _check_sksurv()
    pt = _get_patient_level(train_df, feature_cols)
    X = pt[feature_cols].fillna(pt[feature_cols].median()).values
    y = _build_survival_array(pt)
    model = CoxPHSurvivalAnalysis(alpha=alpha)
    model.fit(X, y)
    return model


def train_rsf(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    n_estimators: int = 200,
    min_samples_leaf: int = 15,
    random_state: int = 42,
) -> "RandomSurvivalForest":
    _check_sksurv()
    pt = _get_patient_level(train_df, feature_cols)
    X = pt[feature_cols].fillna(pt[feature_cols].median()).values
    y = _build_survival_array(pt)
    model = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def evaluate_static_model(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    e_times: np.ndarray,
) -> dict:
    """
    Evaluate a fitted sksurv model (CoxPH or RSF) at each evaluation time.

    Uses cumulative hazard at e_time as the risk score, then computes the
    weighted C-index against the test survival outcome.

    Returns
    -------
    dict mapping e_time → weighted C-index
    """
    from bertpca.metrics import weighted_c_index

    train_pt = _get_patient_level(train_df, feature_cols)
    test_pt = _get_patient_level(test_df, feature_cols)

    train_med = train_pt[feature_cols].median()
    X_test = test_pt[feature_cols].fillna(train_med).values

    train_times = train_pt["tte"].values
    train_events = train_pt["label"].values
    test_times = test_pt["tte"].values
    test_events = test_pt["label"].values

    # Survival functions for each test patient
    surv_fns = model.predict_survival_function(X_test)

    results = {}
    for e_time in e_times:
        risks = np.array([
            1.0 - float(fn(min(e_time, fn.x[-1]))) for fn in surv_fns
        ])
        results[e_time] = weighted_c_index(
            train_times, train_events,
            risks, test_times, test_events,
            e_time,
        )
    return results
