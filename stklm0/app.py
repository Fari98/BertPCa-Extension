#!/usr/bin/env python3
"""
BertPCa STKLM0 Streamlit app.

Always runs both modes on every uploaded file:
  1. Milan Model Inference   — BCR and CSM Milan-trained models side-by-side
  2. Train & Evaluate (CSM)  — fresh BertPCa trained and evaluated on the uploaded data

Results are auto-saved under stklm0/outputs/.

Run from repo root:
  streamlit run stklm0/app.py
"""

import os
import sys
import json
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

_MODELS_DIR  = os.path.join(_APP_DIR, "outputs", "models")
_DATA_DIR    = os.path.join(_APP_DIR, "data")
_PRED_DIR    = os.path.join(_APP_DIR, "outputs", "predictions")
_CONFIG_PATH = os.path.join(_APP_DIR, "config", "config_stklm0.yaml")

# Fixed evaluation parameters
E_TIMES = [365, 1825, 3650]   # 1y, 5y, 10y

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BertPCa — STKLM0",
    page_icon="🏥",
    layout="wide",
)

st.title("BertPCa — Prostate Cancer Survival Prediction")
st.caption("Weibull survival model · STKLM0 patient schema")

import tensorflow as tf  # noqa: E402

from bertpca.loss import weibull_loss  # noqa: E402

# TF ≥2.16 ships Keras 3 which breaks H5 functional-model loading; pin <2.16 in
# requirements.txt so Keras 2 is always used and no shims are needed here.
_CUSTOM_OBJECTS = {"weibull_loss": weibull_loss}

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------


def _load_keras_model(path: str):
    """Load a .keras model regardless of whether it is ZIP (Keras ≥2.12) or HDF5 (older)."""
    import zipfile, shutil, tempfile
    if zipfile.is_zipfile(path):
        return tf.keras.models.load_model(path, custom_objects=_CUSTOM_OBJECTS)
    # HDF5 saved by TF ≤2.11 — Keras 3 picks loader by extension, so copy to .h5
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name
    shutil.copy2(path, tmp_path)
    try:
        return tf.keras.models.load_model(tmp_path, custom_objects=_CUSTOM_OBJECTS)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@st.cache_resource(show_spinner="Loading Milan model …")
def _load_milan_model(outcome_key: str):
    path = os.path.join(_MODELS_DIR, f"best_model_milan_{outcome_key}.keras")
    if not os.path.exists(path):
        return None, f"Model not found: `{path}`"
    if os.path.getsize(path) < 512:
        with open(path, "rb") as fh:
            if fh.read(40).startswith(b"version https://git-lfs"):
                return None, (
                    "Model file is a Git LFS pointer — weights were not downloaded. "
                    "Ensure `packages.txt` contains `git-lfs`."
                )
    return _load_keras_model(path), None


@st.cache_data(show_spinner=False)
def _load_milan_params(outcome_key: str):
    milan_path = os.path.join(_MODELS_DIR, f"milan_{outcome_key}_scaling.json")
    if not os.path.exists(milan_path):
        return None, f"Missing scaling file: `{milan_path}`"
    with open(milan_path) as f:
        milan = json.load(f)
    # preprocessing_params.json holds STKLM0 imputer medians; optional — if absent
    # (e.g. first deployment before any training run), inference falls back to
    # no imputation (uploaded data must be complete) and psa_max=1.
    stklm0_path = os.path.join(_DATA_DIR, "preprocessing_params.json")
    stklm0 = {}
    if os.path.exists(stklm0_path):
        with open(stklm0_path) as f:
            stklm0 = json.load(f)
    return {
        "static_features":  milan["available_static"],
        "dynamic_features": milan.get("dynamic_features", ["times", "psa"]),
        "imputer_medians":  stklm0.get("imputer_medians", {}),
        "train_max":        milan["train_max"],
        "train_min":        milan["train_min"],
        "t_max":            milan["t_max"],
        "psa_max":          stklm0.get("psa_max", 1.0),
    }, None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _milan_inference(df_raw: pd.DataFrame, outcome_key: str):
    from predict_stklm0 import preprocess_for_inference, compute_survival
    from bertpca.data import preprocess_data

    model, err = _load_milan_model(outcome_key)
    if err:
        return None, err
    params, err = _load_milan_params(outcome_key)
    if err:
        return None, err

    df_long, t_last_series = preprocess_for_inference(df_raw, params)
    patient_ids = df_long.index.unique().tolist()
    if not patient_ids:
        return None, "No patients with valid PSA data after preprocessing."

    ds, _ = preprocess_data(
        df_long, params["static_features"], ["times", "psa"], "label",
        seq_length=16, batch_size=len(patient_ids),
    )
    features  = np.array(ds["features"])
    raw_preds = model.predict(features, verbose=0)
    alpha_raw, beta_raw = raw_preds[:, 0], raw_preds[:, 1]
    t_last_days = t_last_series.loc[patient_ids].values
    probs = compute_survival(alpha_raw, beta_raw, t_last_days / params["t_max"], E_TIMES, params["t_max"])

    out = pd.DataFrame({
        "patient_id":  patient_ids,
        "alpha_raw":   np.round(alpha_raw, 4),
        "beta_raw":    np.round(beta_raw, 4),
        "t_last_days": np.round(t_last_days, 1),
    })
    for j, et in enumerate(E_TIMES):
        out[f"P(T>{int(et)}d)"] = np.round(probs[:, j], 4)
    out["risk_score"] = np.round(1.0 - probs[:, 0], 4)
    return out, None


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def _save_predictions(df: pd.DataFrame, outcome_key: str) -> str:
    os.makedirs(_PRED_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_PRED_DIR, f"predictions_{outcome_key}_{ts}.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _show_inference_block(df: pd.DataFrame, outcome_key: str):
    prob_cols = [c for c in df.columns if c.startswith("P(T>")]
    cols = st.columns(len(prob_cols) + 1)
    for i, col in enumerate(prob_cols):
        cols[i].metric(f"Mean {col}", f"{df[col].mean():.3f}")
    cols[-1].metric("Mean risk", f"{df['risk_score'].mean():.3f}")

    display = ["patient_id"] + prob_cols + ["risk_score", "t_last_days", "alpha_raw", "beta_raw"]
    fmt = {c: "{:.3f}" for c in prob_cols + ["risk_score", "alpha_raw", "beta_raw"]}
    st.dataframe(
        df[display].style.format(fmt).background_gradient(subset=["risk_score"], cmap="RdYlGn_r"),
        width="stretch", height=320,
    )

    saved = _save_predictions(df, outcome_key)
    st.caption(f"Auto-saved to `{saved}`")

    st.download_button(
        f"Download {outcome_key.upper()} predictions (CSV)",
        df.to_csv(index=False).encode(),
        file_name=f"bertpca_{outcome_key}_predictions.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

uploaded = st.file_uploader(
    "Upload patient CSV (STKLM0 schema)",
    type=["csv"],
    help="One row per patient. Required columns: id, exp_date, d_diaage, d_spsa, "
         "isup_gealson, t_clean, isup_RP, pT, pR, pRlenght, pN, "
         "PSA1…PSA135, psadate1…psadate135.",
)

if uploaded is None:
    st.info(
        "Upload a CSV file to run inference and train a new model.\n\n"
        "Generate test data first:  `python stklm0/scripts/generate_test_data.py`"
    )
    st.stop()

try:
    df_raw = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not parse CSV: {exc}")
    st.stop()

id_col = "id" if "id" in df_raw.columns else df_raw.columns[0]
df_raw = df_raw.set_index(id_col)
df_raw.index.name = "id"

n_psa = sum(1 for c in df_raw.columns if c.startswith("PSA") and not c.startswith("psadate"))
st.success(f"{len(df_raw):,} patients · {len(df_raw.columns):,} columns · {n_psa} PSA columns")

required = ["exp_date", "d_diaage", "d_spsa", "isup_gealson",
            "t_clean", "isup_RP", "pT", "pR", "pRlenght", "pN"]
missing = [c for c in required if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns: `{'`, `'.join(missing)}`")
    st.stop()
if n_psa == 0:
    st.error("No PSA columns found (expected `PSA1`, `PSA2`, …).")
    st.stop()

with st.expander("Preview data"):
    st.dataframe(df_raw.head(10), width="stretch")

# ── Outcome selection ────────────────────────────────────────────────────────
sel_col, run_col = st.columns([3, 1])
with sel_col:
    selected_outcomes = st.multiselect(
        "Outcomes to predict (Milan models)",
        options=["BCR", "CSM"],
        default=["BCR", "CSM"],
        help="BCR = Biochemical Recurrence · CSM = Cancer-Specific Mortality",
    )
with run_col:
    st.write("")  # vertical alignment spacer
    st.write("")
    do_run = st.button("Run", type="primary", use_container_width=True)

if not do_run:
    st.stop()

if not selected_outcomes:
    st.warning("Select at least one outcome.")
    st.stop()

# Accumulated for the download-all bundle
_bundle: dict[str, bytes] = {}   # {filename_in_zip: bytes}

# ── Milan Inference ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Milan Model Inference")
outcome_cols = st.columns(len(selected_outcomes))
for (outcome_key, col) in zip([o.lower() for o in selected_outcomes], outcome_cols):
    with col:
        st.markdown(f"**{outcome_key.upper()}**")
        with st.spinner(f"Running Milan {outcome_key.upper()} …"):
            result, err = _milan_inference(df_raw, outcome_key)
        if err:
            st.error(err)
        else:
            _show_inference_block(result, outcome_key)
            _bundle[f"predictions_{outcome_key}.csv"] = result.to_csv(index=False).encode()

# ── Download all results ──────────────────────────────────────────────────────
if _bundle:
    import io, zipfile as _zf
    st.markdown("---")
    buf = io.BytesIO()
    with _zf.ZipFile(buf, "w", compression=_zf.ZIP_DEFLATED) as zf:
        for fname, data in _bundle.items():
            zf.writestr(fname, data)
    buf.seek(0)
    st.download_button(
        "Download all results (.zip)",
        buf.getvalue(),
        file_name=f"bertpca_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )
