#!/usr/bin/env python3
"""
BertPCa STKLM0 — Model Comparison App

Evaluates a saved BertPCa model against CoxPH, RSF, and DDH baselines on
uploaded STKLM0 data.  No BertPCa re-training — the model is loaded from disk.

Run from repo root:
    streamlit run stklm0/compare_app.py
"""

import glob as _glob
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
    os.path.join(_REPO_ROOT, "functional_outcomes"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_MODELS_DIR  = os.path.join(_APP_DIR, "outputs", "models")
_RESULTS_DIR = os.path.join(_APP_DIR, "outputs", "results")
_CONFIG_PATH = os.path.join(_APP_DIR, "config", "config_stklm0.yaml")

E_TIMES = [365, 1825, 3650]
P_TIMES = [365, 730]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BertPCa — STKLM0 Comparison",
    page_icon="🏥",
    layout="wide",
)

st.title("BertPCa vs Baselines — STKLM0")
st.caption("Cancer-Specific Mortality · Weibull survival model · IPCW time-dependent C-index")

# CUDA setup for Windows
if sys.platform == "win32":
    _env_root = os.path.dirname(sys.executable)
    for _cd in [
        os.path.join(_env_root, "Library", "bin"),
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin",
    ]:
        if os.path.isdir(_cd) and _cd not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _cd + os.pathsep + os.environ.get("PATH", "")

import tensorflow as tf  # noqa: E402

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    st.success(f"GPU: {', '.join(g.name for g in gpus)} · TF {tf.__version__}")
else:
    st.info(f"CPU-only · TF {tf.__version__}")

from bertpca.loss import weibull_loss  # noqa: E402

_CUSTOM_OBJECTS = {"weibull_loss": weibull_loss}


# ---------------------------------------------------------------------------
# Sidebar — model selection
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Model")

    keras_files = sorted(
        _glob.glob(os.path.join(_MODELS_DIR, "*.keras")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not keras_files:
        st.error(f"No `.keras` models found in `{_MODELS_DIR}`")
        st.stop()

    model_labels = [os.path.basename(p) for p in keras_files]
    selected_label = st.selectbox(
        "BertPCa model to evaluate",
        model_labels,
        index=0,
        help="Models sorted newest-first. Select which one to compare against baselines.",
    )
    selected_model_path = keras_files[model_labels.index(selected_label)]

    mtime = datetime.fromtimestamp(os.path.getmtime(selected_model_path))
    st.caption(f"Saved: {mtime.strftime('%Y-%m-%d %H:%M')}")

    st.divider()
    st.header("Baselines")
    run_coxph = st.checkbox("CoxPH", value=True)
    run_rsf   = st.checkbox("RSF",   value=True)
    run_ddh   = st.checkbox("DDH",   value=True)


# ---------------------------------------------------------------------------
# Data upload
# ---------------------------------------------------------------------------

uploaded = st.file_uploader(
    "Upload STKLM0 patient CSV",
    type=["csv"],
    help=(
        "One row per patient. Required columns: id, exp_date, crmort, t_end, "
        "d_diaage, d_spsa, isup_gealson, t_clean, isup_RP, pT, pR, pRlenght, pN, "
        "PSA1…PSAn, psadate1…psadaten."
    ),
)

if uploaded is None:
    st.info("Upload a STKLM0 CSV to start the comparison.")
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

n_psa = sum(1 for c in df_raw.columns if c.startswith("PSA") and not c.startswith("psadate"))
st.success(f"{len(df_raw):,} patients · {len(df_raw.columns):,} columns · {n_psa} PSA columns")

required = ["exp_date", "crmort", "t_end",
            "d_diaage", "d_spsa", "isup_gealson", "t_clean", "isup_RP", "pT", "pR", "pRlenght", "pN"]
missing = [c for c in required if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns: `{'`, `'.join(missing)}`")
    st.stop()
if n_psa == 0:
    st.error("No PSA columns found (expected `PSA1`, `PSA2`, …).")
    st.stop()

with st.expander("Preview data"):
    st.dataframe(df_raw.head(10), use_container_width=True)

st.divider()

if not st.button("Run Comparison", type="primary", use_container_width=True):
    st.stop()

# ---------------------------------------------------------------------------
# Prepare data
# ---------------------------------------------------------------------------

from config.load_config import load_yaml_config  # noqa: E402

config = load_yaml_config(_CONFIG_PATH)

progress = st.progress(0, text="Preparing data …")
log_box  = st.empty()
log_lines: list[str] = []

def _log(msg: str):
    log_lines.append(msg)
    log_box.info("  \n".join(log_lines[-12:]))


try:
    from prepare_stklm0 import (
        encode_stklm0_features, build_psa_long_stklm0,
        assemble_long_format, split_and_impute, STATIC_COLS,
    )
    from prepare_stklm0 import _parse_date_flexible
except Exception as exc:
    st.error(f"Could not import prepare_stklm0: {exc}")
    st.stop()

_log("Encoding features …")
df_work = df_raw.copy()
df_work["label"] = (pd.to_numeric(df_work.get("crmort", 0), errors="coerce") == 1).astype(int)
exp_date = _parse_date_flexible(df_work["exp_date"])
t_end    = _parse_date_flexible(df_work.get("t_end", pd.Series(pd.NaT, index=df_work.index)))
df_work["tte"] = (t_end - exp_date).dt.days.clip(lower=1, upper=config.T_MAX)

df_static = encode_stklm0_features(df_work)
psa_long  = build_psa_long_stklm0(df_work, t_max=config.T_MAX)
df_long   = assemble_long_format(df_static, df_work[["label", "tte"]], psa_long, STATIC_COLS)
train_df, val_df, test_df, _ = split_and_impute(df_long, STATIC_COLS)

n_train = train_df.index.nunique()
n_val   = val_df.index.nunique()
n_test  = test_df.index.nunique()
n_ev_tr = int(train_df.groupby(level=0)["label"].first().sum())
n_ev_te = int(test_df.groupby(level=0)["label"].first().sum())
event_rate = n_ev_tr / max(n_train, 1)

_log(
    f"Split — train: {n_train} ({n_ev_tr} events, {event_rate:.1%}), "
    f"val: {n_val}, test: {n_test} ({n_ev_te} events)"
)
progress.progress(10, text="Data prepared.")

# ---------------------------------------------------------------------------
# Build TF datasets for BertPCa evaluation
# ---------------------------------------------------------------------------

import tempfile  # noqa: E402

_log("Building TF datasets …")
tmp = tempfile.mkdtemp()
for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
    split.reset_index().to_csv(os.path.join(tmp, f"{name}.csv"), index=False)

from bertpca import load_and_preprocess_data  # noqa: E402

train_ds, val_ds, test_ds, y_train_struct, y_val_struct, y_test_struct = (
    load_and_preprocess_data(
        os.path.join(tmp, "train.csv"),
        os.path.join(tmp, "val.csv"),
        os.path.join(tmp, "test.csv"),
        STATIC_COLS, config.DYNAMIC_FEATURES,
        config.SEQ_LENGTH, config.BATCH_SIZE,
        config.T_MAX, config.AUGMENT_DATA, config.SCALE_FEATURES,
    )
)
progress.progress(20, text="TF datasets ready.")

# ---------------------------------------------------------------------------
# Evaluate BertPCa (selected model)
# ---------------------------------------------------------------------------

from bertpca import calculate_time_dependent_c_index  # noqa: E402

_log(f"Loading BertPCa model: {selected_label} …")
try:
    model = tf.keras.models.load_model(selected_model_path, custom_objects=_CUSTOM_OBJECTS)
    _log("Model loaded.")
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()

_log("Computing BertPCa C-index on test split …")
bertpca_mat = calculate_time_dependent_c_index(
    np.array(test_ds["features"]),
    y_train_struct,
    y_test_struct,
    model,
    p_times=np.array(P_TIMES, dtype=float),
    e_times=np.array(E_TIMES, dtype=float),
    t_max=config.T_MAX,
    return_mean=False,
)

# Raw predictions for diagnostics
raw_preds = model.predict(np.array(test_ds["features"]), verbose=0)
alpha_vals = raw_preds[:, 0]
beta_vals  = raw_preds[:, 1]

progress.progress(50, text="BertPCa done.")

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

baseline_results: dict = {}
train_val_df = pd.concat([train_df, val_df])

def _run_coxph():
    from src.baselines.coxph_rsf import train_coxph, evaluate_static_model
    _log("CoxPH: training …")
    cox = train_coxph(train_df, STATIC_COLS)
    res = evaluate_static_model(cox, train_df, test_df, STATIC_COLS, list(E_TIMES))
    mat = np.full((len(P_TIMES), len(E_TIMES)), np.nan)
    for j, e in enumerate(E_TIMES):
        mat[:, j] = res.get(e, np.nan)
    return mat

def _run_rsf():
    from src.baselines.coxph_rsf import train_rsf, evaluate_static_model
    _log("RSF: training …")
    rsf = train_rsf(train_df, STATIC_COLS)
    res = evaluate_static_model(rsf, train_df, test_df, STATIC_COLS, list(E_TIMES))
    mat = np.full((len(P_TIMES), len(E_TIMES)), np.nan)
    for j, e in enumerate(E_TIMES):
        mat[:, j] = res.get(e, np.nan)
    return mat

def _run_ddh():
    from src.baselines.ddh import train_ddh, evaluate_ddh
    _log("DDH: training …")
    ddh = train_ddh(
        train_df, val_df, STATIC_COLS,
        seq_length=config.SEQ_LENGTH, n_bins=36, t_max=config.T_MAX,
        epochs=100, batch_size=32, patience=10,
    )
    return evaluate_ddh(
        ddh, train_val_df, test_df, STATIC_COLS,
        np.array(P_TIMES, dtype=float), np.array(E_TIMES, dtype=float),
    )

baseline_fns = {}
if run_coxph: baseline_fns["CoxPH"] = _run_coxph
if run_rsf:   baseline_fns["RSF"]   = _run_rsf
if run_ddh:   baseline_fns["DDH"]   = _run_ddh

n_bl = len(baseline_fns)
for i, (name, fn) in enumerate(baseline_fns.items()):
    try:
        baseline_results[name] = fn()
        _log(f"{name}: mean C-index = {float(np.nanmean(baseline_results[name])):.4f}")
    except Exception as exc:
        baseline_results[name] = None
        _log(f"{name} failed: {exc}")
    progress.progress(50 + 40 * (i + 1) // max(n_bl, 1), text=f"Baselines … {name}")

progress.progress(100, text="Done.")
log_box.empty()

# ---------------------------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------------------------

def _masked_mean(mat):
    """Mean of valid cells only (exclude -1 sentinels and NaN)."""
    if mat is None:
        return np.nan
    v = np.where(mat == -1.0, np.nan, mat.astype(float))
    return float(np.nanmean(v))

def _col_mean(mat, j):
    if mat is None:
        return np.nan
    v = np.where(mat[:, j] == -1.0, np.nan, mat[:, j].astype(float))
    return float(np.nanmean(v))

rows = []

for name, mat in baseline_results.items():
    row = {"Method": name, "Type": "Baseline", "Mean C-index": round(_masked_mean(mat), 4)}
    for j, e in enumerate(E_TIMES):
        row[f"e={e//365}y"] = round(_col_mean(mat, j), 4) if mat is not None else np.nan
    rows.append(row)

bertpca_mean = _masked_mean(bertpca_mat)
bertpca_row = {
    "Method": f"BertPCa ({selected_label})",
    "Type": "Weibull",
    "Mean C-index": round(bertpca_mean, 4),
}
for j, e in enumerate(E_TIMES):
    bertpca_row[f"e={e//365}y"] = round(_col_mean(bertpca_mat, j), 4)
rows.append(bertpca_row)

cmp_df = pd.DataFrame(rows).set_index("Method")

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

st.subheader("Model Comparison")
st.caption(
    "IPCW time-dependent C-index on held-out test split. "
    "Values averaged over p_times=[365d, 730d]. NaN = insufficient events at that horizon."
)

numeric_cols = [c for c in cmp_df.columns if c not in ("Type",)]
for col in numeric_cols:
    cmp_df[col] = pd.to_numeric(cmp_df[col], errors="coerce")

styled = (
    cmp_df.style
    .format("{:.4f}", subset=numeric_cols, na_rep="—")
    .background_gradient(cmap="RdYlGn", subset=["Mean C-index"], vmin=0.3, vmax=0.8)
)
st.dataframe(styled, use_container_width=True)

st.divider()

# Per-prediction-time breakdown
st.subheader("BertPCa — Full C-index Grid")
bertpca_df = pd.DataFrame(
    bertpca_mat,
    index=[f"p={p//365}y" for p in P_TIMES],
    columns=[f"e={e//365}y" for e in E_TIMES],
)
valid_mask = bertpca_df != -1.0
display_df = bertpca_df.where(valid_mask)
st.dataframe(
    display_df.style
    .format("{:.4f}", na_rep="—")
    .background_gradient(cmap="RdYlGn", vmin=0.3, vmax=0.8),
    use_container_width=True,
)
mean_valid = float(np.nanmean(np.where(bertpca_mat == -1.0, np.nan, bertpca_mat)))
st.metric("Mean C-index (valid cells only)", f"{mean_valid:.4f}")

# Model diagnostics
st.divider()
with st.expander("Model diagnostics — raw predictions"):
    st.caption(
        "If alpha_raw ≈ −1.0 (constant) and beta_raw is astronomically large for all patients, "
        "the model has collapsed numerically and predictions are meaningless."
    )
    col1, col2 = st.columns(2)
    col1.metric("alpha_raw — mean", f"{alpha_vals.mean():.4f}")
    col1.metric("alpha_raw — std",  f"{alpha_vals.std():.4f}")
    col2.metric("beta_raw — mean",  f"{beta_vals.mean():.4e}")
    col2.metric("beta_raw — std",   f"{beta_vals.std():.4e}")

    if alpha_vals.std() < 0.01 or beta_vals.mean() > 1e10:
        st.error(
            "⚠️  Model collapse detected: alpha is essentially constant "
            f"({alpha_vals.mean():.3f} ± {alpha_vals.std():.4f}) and/or "
            f"beta is numerically exploded ({beta_vals.mean():.2e}). "
            "Re-train the model with a lower learning rate, heavier dropout, "
            "or gradient clipping."
        )

    pred_df = pd.DataFrame({
        "patient_idx": range(len(alpha_vals)),
        "alpha_raw":   np.round(alpha_vals, 4),
        "beta_raw":    np.round(beta_vals,  4),
    })
    st.dataframe(pred_df.head(20), use_container_width=True)

# ---------------------------------------------------------------------------
# Save & download
# ---------------------------------------------------------------------------

os.makedirs(_RESULTS_DIR, exist_ok=True)
_ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
_cmp_path = os.path.join(_RESULTS_DIR, f"model_comparison_{_ts}.csv")
cmp_df.reset_index().to_csv(_cmp_path, index=False)

_bertpca_path = os.path.join(_RESULTS_DIR, f"c_index_bertpca_{_ts}.csv")
bertpca_df.to_csv(_bertpca_path)

st.divider()
col_dl1, col_dl2 = st.columns(2)
col_dl1.download_button(
    "Download comparison table (CSV)",
    cmp_df.reset_index().to_csv(index=False).encode(),
    file_name="model_comparison_stklm0.csv",
    mime="text/csv",
)
col_dl2.download_button(
    "Download BertPCa C-index grid (CSV)",
    bertpca_df.to_csv().encode(),
    file_name="bertpca_cindex_stklm0.csv",
    mime="text/csv",
)
st.caption(f"Results auto-saved: `{_cmp_path}`")
