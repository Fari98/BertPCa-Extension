"""
CAPRA-S score (Cooperberg et al. 2011, Cancer).

Computes the CAPRA-S score (0–12) from post-surgical pathological features and
evaluates its discrimination for a given survival outcome using the weighted C-index.

CAPRA-S is a BCR risk tool; here it is applied cross-domain to EF/UC recovery to
assess whether surgical pathology features correlate with functional recovery.

Score components:
  PSA (<6→0, 6–10→1, ≥10→2)               [tpsa]
  Pathological Gleason (≤6→0, 3+4→1, 4+3→2, ≥4+4→3)  [pathgg_primary, pathgg_secondary]
  Positive surgical margins (no→0, yes→2)  [psm_bin]
  ECE (no→0, yes→1)                        [ece_bin]
  SVI (no→0, yes→2)                        [svi_bin]
  LNI (no→0, yes→4)                        [lni_bin]
"""

import numpy as np
import pandas as pd


def capras_score(df: pd.DataFrame) -> np.ndarray:
    """
    Compute CAPRA-S score for each row in df.

    Expected columns (all numeric, binarized where noted):
      tpsa, pathgg_primary, pathgg_secondary,
      psm (0/1), ece_bin (0/1), svi_bin (0/1), lni_bin (0/1)

    Returns
    -------
    np.ndarray
        CAPRA-S scores (float, NaN where inputs are missing)
    """
    scores = np.zeros(len(df), dtype=float)

    # PSA component
    psa = pd.to_numeric(df["tpsa"], errors="coerce").values
    scores += np.where(psa < 6, 0, np.where(psa <= 10, 1, 2))

    # Gleason component
    if "pathgg_primary" in df.columns and "pathgg_secondary" in df.columns:
        gp = pd.to_numeric(df["pathgg_primary"], errors="coerce").values
        gs = pd.to_numeric(df["pathgg_secondary"], errors="coerce").values
        total_g = gp + gs
        gleason_pts = np.where(
            total_g <= 6, 0,
            np.where((gp == 3) & (gs == 4), 1,
                     np.where((gp == 4) & (gs == 3), 2, 3))
        )
        missing_gleason = np.isnan(gp) | np.isnan(gs)
    elif "pathgg_group" in df.columns:
        # Grade group 1→0 pts, 2→1 pt, 3→2 pts, 4/5→3 pts (CAPRA-S table)
        gg = pd.to_numeric(df["pathgg_group"], errors="coerce").values
        gleason_pts = np.where(gg <= 1, 0, np.where(gg == 2, 1, np.where(gg == 3, 2, 3)))
        gp = gg  # used only for missing-value tracking below
        gs = gg
        missing_gleason = np.isnan(gg)
    else:
        gleason_pts = np.zeros(len(df))
        gp = gs = np.full(len(df), np.nan)
        missing_gleason = np.ones(len(df), dtype=bool)
    scores += gleason_pts

    def _col(df, *names, default=0.0):
        """Return numeric array for the first matching column, or a zero array."""
        for name in names:
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").fillna(default).values
        return np.full(len(df), default, dtype=float)

    # Margins
    psm = _col(df, "psm")
    scores += np.where(psm >= 1, 2, 0)

    # ECE
    ece = _col(df, "ece_bin", "ece")
    scores += np.where(ece >= 1, 1, 0)

    # SVI
    svi = _col(df, "svi_bin", "svi")
    scores += np.where(svi >= 1, 2, 0)

    # LNI
    lni = _col(df, "lni_bin", "lni")
    scores += np.where(lni >= 1, 4, 0)

    # NaN-out rows where PSA or Gleason is missing
    missing = np.isnan(psa) | missing_gleason
    scores[missing] = np.nan

    return scores


def evaluate_capras(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    e_times: np.ndarray,
) -> dict:
    """
    Evaluate CAPRA-S discrimination for functional recovery outcomes.

    Parameters
    ----------
    df_train : DataFrame
        One row per patient with columns: tte, label, + feature columns.
    df_test : DataFrame
        Same format as df_train.
    e_times : np.ndarray
        Evaluation time horizons (unscaled days).

    Returns
    -------
    dict mapping e_time → weighted C-index (or NaN if not computable)
    """
    from bertpca.metrics import weighted_c_index

    scores_test = capras_score(df_test)
    # Higher CAPRA-S → worse prognosis → later/no recovery → negate for recovery endpoint
    risk_test = -scores_test

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
