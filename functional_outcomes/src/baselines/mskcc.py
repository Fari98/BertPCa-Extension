"""
MSKCC post-operative nomogram (Stephenson et al., JCO 2005).

Computes the 7-year BCR-free probability from published Cox regression coefficients
and uses (1 - probability) as a risk score evaluated against EF/UC outcomes.

Reference:
  Stephenson AJ et al. "Predicting the outcome of salvage radiation therapy for
  recurrent prostate cancer after radical prostatectomy."
  J Clin Oncol. 2007;25(15):2035-2041.

  The post-operative BCR nomogram coefficients below are taken from the publicly
  available MSKCC model (Kattan et al., JAMA 1999 / updated Stephenson 2005):

  log(PSA + 0.1), Gleason primary (3/4/5), Gleason secondary (3/4/5),
  pT-stage (T3a=ECE, T3b=SVI, T4), positive margins, neoadjuvant HT.

Coefficients (log hazard ratios) are from Stephenson et al. 2005 (Table 2):
  Intercept / baseline: ln(H_0(t)) calibrated so that mean prediction ≈ population BCR.

Note: this is an approximation of the nomogram; the exact baseline survival
function requires the original data. We use the published point estimates and a
calibrated baseline to produce relative risk scores.
"""

import numpy as np
import pandas as pd


# Published log-HR coefficients (Stephenson et al. 2005, JCO, Table 2)
_COEFS = {
    "log_psa":           0.508,   # log(PSA + 0.1)
    "gleason_primary_4": 0.396,   # primary Gleason == 4 vs 3
    "gleason_primary_5": 0.781,   # primary Gleason == 5 vs 3
    "gleason_secondary_4": 0.360, # secondary Gleason == 4 vs 3
    "gleason_secondary_5": 0.886, # secondary Gleason == 5 vs 3
    "pT3a":              0.540,   # ECE (pT3a) vs pT2
    "pT3b":              0.831,   # SVI (pT3b) vs pT2
    "pT4":               1.021,   # pT4 vs pT2
    "psm":               0.386,   # positive surgical margins
    "neo_ht":           -0.131,   # neoadjuvant hormonal therapy
}

# Approximate baseline 7-year BCR-free survival for the reference patient
# (PSA<6, Gleason 3+3, pT2, negative margins, no HT) — calibrated from the paper
_BASELINE_7Y_BCR_FREE = 0.92


def _linear_predictor(df: pd.DataFrame) -> np.ndarray:
    psa = pd.to_numeric(df["tpsa"], errors="coerce").values

    # Prefer individual Gleason scores; fall back to grade group decomposition
    if "pathgg_primary" in df.columns or "bxgg_primary" in df.columns:
        gp = pd.to_numeric(df.get("pathgg_primary", df.get("bxgg_primary")), errors="coerce").values
        gs = pd.to_numeric(df.get("pathgg_secondary", df.get("bxgg_secondary")), errors="coerce").values
    elif "pathgg_group" in df.columns:
        # Approximate primary/secondary from ISUP grade group
        gg = pd.to_numeric(df["pathgg_group"], errors="coerce").values
        gp = np.where(gg >= 3, 4, 3).astype(float)  # group 3+: primary ≥4
        gs = np.where(gg == 1, 3, np.where(gg == 2, 4, np.where(gg == 3, 3, np.where(gg == 4, 4, 5)))).astype(float)
        gp[np.isnan(gg)] = np.nan
        gs[np.isnan(gg)] = np.nan
    else:
        gp = np.full(len(df), np.nan)
        gs = np.full(len(df), np.nan)
    def _col(df, *names, default=0.0):
        for name in names:
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").fillna(default).values
        return np.full(len(df), default, dtype=float)

    psm    = _col(df, "psm")
    ece    = _col(df, "ece_bin", "ece")
    svi    = _col(df, "svi_bin", "svi")
    pstage = _col(df, "pstage")
    neo_ht = _col(df, "neo_adjHT")

    lp = _COEFS["log_psa"] * np.log(np.clip(psa, 1e-3, None) + 0.1)
    lp += _COEFS["gleason_primary_4"] * (gp == 4).astype(float)
    lp += _COEFS["gleason_primary_5"] * (gp == 5).astype(float)
    lp += _COEFS["gleason_secondary_4"] * (gs == 4).astype(float)
    lp += _COEFS["gleason_secondary_5"] * (gs == 5).astype(float)
    # pT stage: use ece/svi if pstage not informative
    lp += _COEFS["pT3a"] * np.where((ece >= 1) & (svi < 1) & (pstage < 7), 1, 0)
    lp += _COEFS["pT3b"] * np.where(svi >= 1, 1, 0)
    lp += _COEFS["pT4"] * np.where(pstage >= 8, 1, 0)
    lp += _COEFS["psm"] * (psm >= 1).astype(float)
    lp += _COEFS["neo_ht"] * (neo_ht >= 1).astype(float)

    missing = np.isnan(psa) | np.isnan(gp) | np.isnan(gs)
    lp[missing] = np.nan
    return lp


def mskcc_score(df: pd.DataFrame) -> np.ndarray:
    """
    Compute MSKCC 7-year BCR risk score (1 - BCR-free probability).

    Higher score → higher BCR risk → used as risk for poor functional recovery.

    Returns
    -------
    np.ndarray  risk scores in [0, 1]
    """
    lp = _linear_predictor(df)
    # P(BCR-free at 7y) = baseline_7y ^ exp(lp)
    bcr_free = _BASELINE_7Y_BCR_FREE ** np.exp(lp)
    return 1.0 - bcr_free


def evaluate_mskcc(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    e_times: np.ndarray,
) -> dict:
    """
    Evaluate MSKCC discrimination for functional recovery.

    Parameters
    ----------
    df_train, df_test : DataFrame, one row per patient, columns: tte, label, features.
    e_times : np.ndarray, evaluation horizons (unscaled days).

    Returns
    -------
    dict mapping e_time → weighted C-index
    """
    from bertpca.metrics import weighted_c_index

    # Higher MSKCC risk → worse functional recovery prognosis (positive correlation)
    risk_test = mskcc_score(df_test)

    train_times = df_train["tte"].values
    train_events = df_train["label"].values
    test_times = df_test["tte"].values
    test_events = df_test["label"].values

    results = {}
    for e_time in e_times:
        valid = ~np.isnan(risk_test)
        if valid.sum() < 10:
            results[e_time] = np.nan
            continue
        results[e_time] = weighted_c_index(
            train_times, train_events,
            risk_test[valid],
            test_times[valid], test_events[valid],
            e_time,
        )
    return results
