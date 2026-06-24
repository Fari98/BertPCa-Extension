"""
Survival analysis metrics including weighted C-index and Brier score.

All credit goes to the original authors:
    https://github.com/chl8856/Dynamic-DeepHit/blob/master/utils_eval.py

"""

import numpy as np
from lifelines import KaplanMeierFitter


def CensoringProb(Y: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Estimate censoring probability using Kaplan-Meier estimator.
    
    Parameters
    ----------
    Y : np.ndarray
        Event indicator (1=event, 0=censored)
    T : np.ndarray
        Survival/censoring times
    
    Returns
    -------
    np.ndarray
        Array with time points and corresponding censoring probabilities
    """
    T = T.reshape([-1])  # (N,) - np array
    Y = Y.reshape([-1])  # (N,) - np array

    kmf = KaplanMeierFitter()
    # Censoring prob = survival probability of event "censoring"
    kmf.fit(T, event_observed=(Y == 0).astype(int))
    G = np.asarray(kmf.survival_function_.reset_index()).transpose()
    # Fill 0 with ZoH (to prevent nan values)
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]
    
    return G


def weighted_c_index(
    T_train: np.ndarray,
    Y_train: np.ndarray,
    Prediction: np.ndarray,
    T_test: np.ndarray,
    Y_test: np.ndarray,
    Time: float
) -> float:
    """
    Calculate weighted time-dependent C-index for survival analysis.
    
    This is a cause-specific c(t)-index that accounts for censoring.
    
    Parameters
    ----------
    T_train : np.ndarray
        Training survival/censoring times
    Y_train : np.ndarray
        Training event indicator (1=event, 0=censored)
    Prediction : np.ndarray
        Risk predictions at Time (higher --> more risky)
    T_test : np.ndarray
        Test survival/censoring times
    Y_test : np.ndarray
        Test event indicator (1=event, 0=censored)
    Time : float
        Time horizon for evaluation
    
    Returns
    -------
    float
        Weighted C-index value, or -1 if not computable
    """
    G = CensoringProb(Y_train, T_train)

    N = len(Prediction)
    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))
    Num = 0
    Den = 0
    
    for i in range(N):
        tmp_idx = np.where(G[0, :] >= T_test[i])[0]

        if len(tmp_idx) == 0:
            W = (1. / G[1, -1]) ** 2
        else:
            W = (1. / G[1, tmp_idx[0]]) ** 2

        A[i, np.where(T_test[i] < T_test)] = 1. * W
        Q[i, np.where(Prediction[i] > Prediction)] = 1.  # give weights

        if (T_test[i] <= Time and Y_test[i] == 1):
            N_t[i, :] = 1.

    Num = np.sum(((A) * N_t) * Q)
    Den = np.sum((A) * N_t)

    if Num == 0 and Den == 0:
        result = -1  # not able to compute c-index!
    else:
        result = float(Num / Den)

    return result


def weighted_brier_score(
    T_train: np.ndarray,
    Y_train: np.ndarray,
    Prediction: np.ndarray,
    T_test: np.ndarray,
    Y_test: np.ndarray,
    Time: float
) -> float:
    """
    Calculate weighted Brier score for survival analysis.
    
    Parameters
    ----------
    T_train : np.ndarray
        Training survival/censoring times
    Y_train : np.ndarray
        Training event indicator (1=event, 0=censored)
    Prediction : np.ndarray
        Predicted survival probabilities
    T_test : np.ndarray
        Test survival/censoring times
    Y_test : np.ndarray
        Test event indicator (1=event, 0=censored)
    Time : float
        Time horizon for evaluation
    
    Returns
    -------
    float
        Weighted Brier score
    """
    G = CensoringProb(Y_train, T_train)
    N = len(Prediction)

    W = np.zeros(len(Y_test))
    Y_tilde = (T_test > Time).astype(float)

    for i in range(N):
        tmp_idx1 = np.where(G[0, :] >= T_test[i])[0]
        tmp_idx2 = np.where(G[0, :] >= Time)[0]

        if len(tmp_idx1) == 0:
            G1 = G[1, -1]
        else:
            G1 = G[1, tmp_idx1[0]]

        if len(tmp_idx2) == 0:
            G2 = G[1, -1]
        else:
            G2 = G[1, tmp_idx2[0]]
            
        W[i] = (1. - Y_tilde[i]) * float(Y_test[i]) / G1 + Y_tilde[i] / G2

    y_true = ((T_test <= Time) * Y_test).astype(float)

    return np.mean(W * (Y_tilde - (1. - Prediction)) ** 2)
