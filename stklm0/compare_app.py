#!/usr/bin/env python3
"""
BertPCa STKLM0 — Nomogram App

Upload a STKLM0 CSV → CAPRA-S and MSKCC scores are computed automatically.
If the file also contains exp_date / crmort / t_end, IPCW C-index is computed.

Run from repo root:
    streamlit run stklm0/compare_app.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_APP_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_APP_DIR)
for _p in [
    os.path.join(_REPO_ROOT, "bertpca", "src"),
    os.path.join(_REPO_ROOT, "bertpca"),
    os.path.join(_APP_DIR, "scripts"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_RESULTS_DIR = os.path.join(_APP_DIR, "outputs", "results")
E_TIMES = [1825, 3650]   # 5y, 10y

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Prostate Cancer Nomograms",
    page_icon="🏥",
    layout="wide",
)

st.title("Prostate Cancer Nomograms")
st.caption(
    "Upload a STKLM0 patient CSV — CAPRA-S and MSKCC scores are computed automatically. "
    "Include `exp_date`, `crmort`, `t_end` to also get IPCW discrimination metrics."
)

# ---------------------------------------------------------------------------
# Nomogram scoring
# ---------------------------------------------------------------------------

def _capras_score(df_s: pd.DataFrame) -> np.ndarray:
    psa  = pd.to_numeric(df_s.get("d_spsa",      pd.Series(np.nan, index=df_s.index)), errors="coerce").values
    isup = pd.to_numeric(df_s.get("isup_gealson", pd.Series(np.nan, index=df_s.index)), errors="coerce").values
    psm  = pd.to_numeric(df_s.get("pR_bin",       pd.Series(0.0,   index=df_s.index)), errors="coerce").fillna(0).values
    lni  = pd.to_numeric(df_s.get("pN_bin",       pd.Series(0.0,   index=df_s.index)), errors="coerce").fillna(0).values
    pT   = pd.to_numeric(df_s.get("pT_ord",       pd.Series(0.0,   index=df_s.index)), errors="coerce").fillna(0).values
    scores = (
        np.where(psa < 6, 0, np.where(psa <= 10, 1, 2))
        + np.where(isup <= 1, 0, np.where(isup == 2, 1, np.where(isup == 3, 2, 3)))
        + np.where(psm >= 1, 2, 0)
        + np.where(pT >= 1, 1, 0)
        + np.where(pT >= 2, 2, 0)
        + np.where(lni >= 1, 4, 0)
    ).astype(float)
    scores[np.isnan(psa) | np.isnan(isup)] = np.nan
    return scores


def _mskcc_score(df_s: pd.DataFrame) -> np.ndarray:
    psa  = pd.to_numeric(df_s.get("d_spsa",      pd.Series(np.nan, index=df_s.index)), errors="coerce").values
    isup = pd.to_numeric(df_s.get("isup_gealson", pd.Series(np.nan, index=df_s.index)), errors="coerce").values
    psm  = pd.to_numeric(df_s.get("pR_bin",       pd.Series(0.0,   index=df_s.index)), errors="coerce").fillna(0).values
    pT   = pd.to_numeric(df_s.get("pT_ord",       pd.Series(0.0,   index=df_s.index)), errors="coerce").fillna(0).values
    gp = np.where(isup >= 3, 4.0, 3.0)
    gs = np.where(isup == 1, 3.0, np.where(isup == 2, 4.0, np.where(isup == 3, 3.0, 4.0)))
    gp[np.isnan(isup)] = np.nan
    gs[np.isnan(isup)] = np.nan
    ece = (pT >= 1).astype(float)
    svi = (pT >= 2).astype(float)
    pt4 = (pT >= 3).astype(float)
    lp = (0.508 * np.log(np.clip(psa, 1e-3, None) + 0.1)
          + 0.396 * (gp == 4) + 0.781 * (gp == 5)
          + 0.360 * (gs == 4) + 0.886 * (gs == 5)
          + 0.540 * np.where((ece >= 1) & (svi < 1), 1.0, 0.0)
          + 0.831 * svi + 1.021 * pt4
          + 0.386 * (psm >= 1)).astype(float)
    lp[np.isnan(psa) | np.isnan(gp)] = np.nan
    return 1.0 - 0.92 ** np.exp(lp)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

uploaded = st.file_uploader(
    "Upload STKLM0 patient CSV",
    type=["csv"],
    help=(
        "One row per patient. Minimum required: d_spsa, isup_gealson. "
        "Also used when present: pT_ord, pR_bin, pN_bin. "
        "For C-index: exp_date, crmort, t_end, PSA1…PSAn, psadate1…psadaten."
    ),
)

if uploaded is None:
    st.info("Upload a CSV to compute nomogram scores.")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not parse CSV: {exc}")
    st.stop()

if "id" in df_raw.columns:
    df_raw = df_raw.set_index("id")
else:
    df_raw.index = range(len(df_raw))
df_raw.index.name = "id"

st.success(f"{len(df_raw):,} patients loaded")

with st.expander("Preview data"):
    st.dataframe(df_raw.head(10), use_container_width=True)

# ---------------------------------------------------------------------------
# Encode features → score
# ---------------------------------------------------------------------------

try:
    from prepare_stklm0 import encode_stklm0_features
    df_enc = encode_stklm0_features(df_raw)
except Exception:
    df_enc = df_raw.copy()

capras = _capras_score(df_enc)
mskcc  = _mskcc_score(df_enc)

n_capras = int(np.sum(~np.isnan(capras)))
n_mskcc  = int(np.sum(~np.isnan(mskcc)))

# Summary metrics
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.subheader("CAPRA-S")
    st.metric("Patients scored", f"{n_capras} / {len(df_raw)}")
    if n_capras > 0:
        st.metric("Mean score", f"{np.nanmean(capras):.2f}")
        st.metric("Range", f"{np.nanmin(capras):.0f} – {np.nanmax(capras):.0f}")
with col2:
    st.subheader("MSKCC")
    st.metric("Patients scored", f"{n_mskcc} / {len(df_raw)}")
    if n_mskcc > 0:
        st.metric("Mean score", f"{np.nanmean(mskcc):.3f}")
        st.metric("Range", f"{np.nanmin(mskcc):.3f} – {np.nanmax(mskcc):.3f}")

# Per-patient table
st.divider()
st.subheader("Per-patient scores")
scores_df = pd.DataFrame(
    {"CAPRA-S": np.round(capras, 2), "MSKCC": np.round(mskcc, 3)},
    index=df_enc.index,
)
st.dataframe(
    scores_df.style
        .format("{:.3f}", na_rep="—")
        .background_gradient(cmap="RdYlGn_r", subset=["CAPRA-S", "MSKCC"]),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# IPCW C-index — only when outcome columns are present
# ---------------------------------------------------------------------------

has_outcome = all(c in df_raw.columns for c in ["exp_date", "crmort", "t_end"])

if has_outcome:
    st.divider()
    st.subheader("Discrimination — IPCW C-index")

    with st.spinner("Computing IPCW C-index on held-out test split …"):
        try:
            from prepare_stklm0 import (
                build_psa_long_stklm0, assemble_long_format,
                split_and_impute, STATIC_COLS, _parse_date_flexible,
            )
            from bertpca.metrics import weighted_c_index as _wci

            T_MAX = 3650.0
            df_work = df_raw.copy()
            df_work["label"] = (pd.to_numeric(df_work.get("crmort", 0), errors="coerce") == 1).astype(int)
            exp_date = _parse_date_flexible(df_work["exp_date"])
            t_end_s  = _parse_date_flexible(df_work.get("t_end", pd.Series(pd.NaT, index=df_work.index)))
            df_work["tte"] = (t_end_s - exp_date).dt.days.clip(lower=1, upper=T_MAX)

            df_enc2  = encode_stklm0_features(df_work)
            psa_long = build_psa_long_stklm0(df_work, t_max=T_MAX)
            df_long  = assemble_long_format(df_enc2, df_work[["label", "tte"]], psa_long, STATIC_COLS)
            train_df, _, test_df, _ = split_and_impute(df_long, STATIC_COLS)

            train_last = train_df.groupby(level=0).first()
            test_last  = test_df.groupby(level=0).first()
            T_tr = train_last["tte"].values
            Y_tr = train_last["label"].values.astype(float)
            T_te = test_last["tte"].values
            Y_te = test_last["label"].values.astype(float)

            n_ev_te = int(Y_te.sum())
            st.caption(
                f"Test split: {len(T_te)} patients, {n_ev_te} CSM events "
                f"(train: {len(T_tr)}, {int(Y_tr.sum())} events)"
            )

            capras_te = _capras_score(test_last)
            mskcc_te  = _mskcc_score(test_last)

            rows = []
            for name, sc in [("CAPRA-S", capras_te), ("MSKCC", mskcc_te)]:
                valid = ~np.isnan(sc)
                row = {"Method": name}
                for e in E_TIMES:
                    label = f"e={e // 365}y"
                    if valid.sum() >= 5:
                        c = _wci(T_tr, Y_tr, sc[valid], T_te[valid], Y_te[valid], float(e))
                        row[label] = round(float(c), 4) if c != -1 else float("nan")
                    else:
                        row[label] = float("nan")
                vals = [v for v in row.values() if isinstance(v, float) and not np.isnan(v)]
                row["Mean C-index"] = round(float(np.mean(vals)), 4) if vals else float("nan")
                rows.append(row)

            cmp_df = pd.DataFrame(rows).set_index("Method")
            st.dataframe(
                cmp_df.style
                    .format("{:.4f}", na_rep="—")
                    .background_gradient(cmap="RdYlGn", subset=["Mean C-index"], vmin=0.3, vmax=0.8),
                use_container_width=True,
            )

            os.makedirs(_RESULTS_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(_RESULTS_DIR, f"nomogram_comparison_{ts}.csv")
            cmp_df.reset_index().to_csv(out_path, index=False)
            st.caption(f"Auto-saved to: `{out_path}`")

        except Exception as exc:
            st.warning(f"C-index computation failed: {exc}")

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

st.divider()
st.download_button(
    "Download per-patient scores (CSV)",
    scores_df.reset_index().to_csv(index=False).encode(),
    file_name="nomogram_scores_stklm0.csv",
    mime="text/csv",
    use_container_width=True,
)
