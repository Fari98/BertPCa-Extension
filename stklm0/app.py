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
_RESULTS_DIR = os.path.join(_APP_DIR, "outputs", "results")
_CONFIG_PATH = os.path.join(_APP_DIR, "config", "config_stklm0.yaml")

# Fixed evaluation parameters
E_TIMES  = [365, 1825, 3650]   # 1y, 5y, 10y
P_TIMES  = [180, 365, 730]     # 6m, 1y, 2y (landmarks for training eval)

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

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Milan model …")
def _load_milan_model(outcome_key: str):
    import tensorflow as tf
    from bertpca.loss import weibull_loss
    path = os.path.join(_MODELS_DIR, f"best_model_milan_{outcome_key}.keras")
    if not os.path.exists(path):
        return None, f"Model not found: `{path}`"
    # Detect Git LFS pointer (file < 512 bytes and starts with "version https://")
    size = os.path.getsize(path)
    if size < 512:
        with open(path, "rb") as fh:
            head = fh.read(40)
        if head.startswith(b"version https://git-lfs"):
            return None, (
                "Model file is a Git LFS pointer — the actual weights were not downloaded. "
                "Ensure `git-lfs` is installed on the server (`packages.txt` must contain `git-lfs`) "
                "and the repo was cloned with `git lfs pull`."
            )
    return tf.keras.models.load_model(path, custom_objects={"weibull_loss": weibull_loss}), None


@st.cache_data(show_spinner=False)
def _load_milan_params(outcome_key: str):
    milan_path  = os.path.join(_MODELS_DIR, f"milan_{outcome_key}_scaling.json")
    stklm0_path = os.path.join(_DATA_DIR, "preprocessing_params.json")
    missing = [p for p in [milan_path, stklm0_path] if not os.path.exists(p)]
    if missing:
        return None, "Missing param files:\n" + "\n".join(f"  • {p}" for p in missing)
    with open(milan_path) as f:
        milan = json.load(f)
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
# Training
# ---------------------------------------------------------------------------

def _prepare_in_memory(df_raw: pd.DataFrame, t_max: float):
    from prepare_stklm0 import (
        encode_stklm0_features, build_psa_long_stklm0,
        assemble_long_format, split_and_impute, STATIC_COLS,
    )
    df = df_raw.copy()
    df["label"] = (pd.to_numeric(df.get("crmort", 0), errors="coerce") == 1).astype(int)
    exp_date = pd.to_datetime(df["exp_date"], errors="coerce")
    t_end    = pd.to_datetime(df.get("t_end",  pd.NaT), errors="coerce")
    df["tte"] = (t_end - exp_date).dt.days.clip(lower=1, upper=t_max)

    df_static = encode_stklm0_features(df)
    psa_long  = build_psa_long_stklm0(df, t_max=t_max)
    df_long   = assemble_long_format(df_static, df[["label", "tte"]], psa_long, STATIC_COLS)
    train, val, test, imp = split_and_impute(df_long, STATIC_COLS)
    return train, val, test, STATIC_COLS


def _train_and_evaluate(df_raw: pd.DataFrame, log_fn=None):
    import tempfile
    import tensorflow as tf
    from tensorflow import keras
    from bertpca import build_bert_pca, training_loop, calculate_time_dependent_c_index, set_seeds
    from config.load_config import load_yaml_config

    def log(msg):
        if log_fn:
            log_fn(msg)

    config = load_yaml_config(_CONFIG_PATH)
    set_seeds(config.SEED)

    log("Preparing data …")
    train, val, test, static_cols = _prepare_in_memory(df_raw, config.T_MAX)
    log(f"Split — train: {train.index.nunique()} · val: {val.index.nunique()} · test: {test.index.nunique()} patients")

    tmp = tempfile.mkdtemp()
    for name, split in [("train", train), ("val", val), ("test", test)]:
        split.reset_index().to_csv(os.path.join(tmp, f"{name}.csv"), index=False)

    log("Building TensorFlow datasets …")
    train_ds, val_ds, test_ds, y_train, y_val, y_test = (
        __import__("bertpca").load_and_preprocess_data(
            os.path.join(tmp, "train.csv"),
            os.path.join(tmp, "val.csv"),
            os.path.join(tmp, "test.csv"),
            static_cols, config.DYNAMIC_FEATURES,
            config.SEQ_LENGTH, config.BATCH_SIZE,
            config.T_MAX, config.AUGMENT_DATA, config.SCALE_FEATURES,
        )
    )

    n_features = len(static_cols) + len(config.DYNAMIC_FEATURES)
    log(f"Building model ({n_features} features) …")
    keras.backend.clear_session()
    model = build_bert_pca(n_features=n_features, seq_length=config.SEQ_LENGTH, **config.MODEL_CONFIG)

    X_train = np.array(train_ds["features"])
    y_train_surv = np.array(train_ds["labels_surv"])
    X_val   = np.array(val_ds["features"])
    y_val_surv   = np.array(val_ds["labels_surv"])

    bs = config.TRAINING_CONFIG["batch_size"]
    train_tf = tf.data.Dataset.from_tensor_slices((X_train, y_train_surv)).shuffle(1024).batch(bs)
    val_tf   = tf.data.Dataset.from_tensor_slices((X_val,   y_val_surv)).batch(bs)

    log("Training BertPCa … (this may take several minutes)")
    model, _ = training_loop(
        model, train_tf, val_tf,
        y_train=y_train, y_val=y_val,
        training_config=config.TRAINING_CONFIG,
        evaluation_config=config.EVALUATION_CONFIG,
        c_index_interval=999,
    )

    log("Evaluating on test set …")
    c_matrix = calculate_time_dependent_c_index(
        np.array(test_ds["features"]), y_train, y_test, model,
        p_times=np.array(P_TIMES, dtype=float),
        e_times=np.array(E_TIMES,  dtype=float),
        t_max=config.EVALUATION_CONFIG["t_max"],
        return_mean=False,
    )
    return model, c_matrix


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def _save_predictions(df: pd.DataFrame, outcome_key: str) -> str:
    os.makedirs(_PRED_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_PRED_DIR, f"predictions_{outcome_key}_{ts}.csv")
    df.to_csv(path, index=False)
    return path


def _save_c_matrix(c_matrix: np.ndarray, tag: str) -> str:
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_RESULTS_DIR, f"c_index_{tag}_{ts}.csv")
    df_c = pd.DataFrame(
        c_matrix,
        index=[f"p={int(p)}d" for p in P_TIMES],
        columns=[f"e={int(e)}d" for e in E_TIMES],
    )
    df_c.to_csv(path)
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


def _show_c_matrix_block(c_matrix: np.ndarray):
    df_c = pd.DataFrame(
        c_matrix,
        index=[f"p={int(p)}d" for p in P_TIMES],
        columns=[f"e={int(e)}d" for e in E_TIMES],
    )
    st.dataframe(
        df_c.style.format("{:.4f}").background_gradient(cmap="RdYlGn", vmin=0.3, vmax=0.8),
        width="stretch",
    )
    st.metric("Mean C-Index", f"{float(np.nanmean(c_matrix)):.4f}")

    saved = _save_c_matrix(c_matrix, "stklm0_train")
    st.caption(f"Auto-saved to `{saved}`")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

uploaded = st.file_uploader(
    "Upload patient CSV (STKLM0 schema)",
    type=["csv"],
    help="One row per patient. Required: id, exp_date, d_diaage, d_spsa, isup_gealson, "
         "t_clean, isup_RP, pT, pR, pRlenght, pN, PSA1…PSA135, psadate1…psadate135. "
         "Training also needs crmort and t_end.",
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

if not st.button("Run", type="primary", use_container_width=True):
    st.stop()

# ── Milan Inference ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Milan Model Inference")
bcr_col, csm_col = st.columns(2)

for outcome_key, col in [("bcr", bcr_col), ("csm", csm_col)]:
    with col:
        st.markdown(f"**{outcome_key.upper()}**")
        with st.spinner(f"Running Milan {outcome_key.upper()} …"):
            result, err = _milan_inference(df_raw, outcome_key)
        if err:
            st.error(err)
        else:
            _show_inference_block(result, outcome_key)

# ── Train & Evaluate ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Train & Evaluate New Model (CSM)")

missing_train = [c for c in ["crmort", "t_end"] if c not in df_raw.columns]
if missing_train:
    st.warning(
        f"Columns `{'`, `'.join(missing_train)}` not found — skipping training. "
        "Add them to enable this mode."
    )
else:
    log_box = st.empty()
    log_lines: list[str] = []

    def _log(msg: str):
        log_lines.append(msg)
        log_box.info("  \n".join(log_lines))

    with st.spinner("Training in progress …"):
        try:
            model, c_matrix = _train_and_evaluate(df_raw, log_fn=_log)
        except Exception as exc:
            log_box.empty()
            st.error(f"Training failed: {exc}")
            st.exception(exc)
            st.stop()

    log_box.empty()
    st.success("Training complete.")
    _show_c_matrix_block(c_matrix)

    model_path = os.path.join(_MODELS_DIR, "app_trained_stklm0_csm.keras")
    os.makedirs(_MODELS_DIR, exist_ok=True)
    model.save(model_path)
    st.caption(f"Model saved to `{model_path}`")
