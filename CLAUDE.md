# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands must be run from the `bertpca/` directory.

```bash
# Install dependencies
pip install uv
uv lock
uv sync

# Install dev dependencies (pytest, black, flake8)
uv sync --extra dev

# Preprocess PBC2 dataset into train/val/test splits
uv run python scripts/preprocess_pbc_dataset.py

# Train with defaults from config/config.yaml
uv run python scripts/train_bertpca.py

# Train and save model to a specific path
uv run python scripts/train_bertpca.py --output outputs/models/my_model.keras

# Hyperparameter tuning (Optuna)
uv run python scripts/tune_bertpca.py --n-trials 50 --study-name my_study
```

Outputs are written to `outputs/models/` (saved model) and `outputs/results/` (C-index table, training log).

## Architecture

### Data flow

1. `config/config.yaml` is the single source of truth for all paths, features, and hyperparameters.
2. `config/load_config.py:load_yaml_config()` parses it into a `SimpleNamespace` used by all scripts.
3. CSV data files in `data/` have a multi-index on `id` (patient) with one row per longitudinal observation.
4. `src/bertpca/data.py:load_and_preprocess_data()` orchestrates: min-max scaling (fit on train only), optional augmentation, sequence padding, and label structuring. Returns Hugging Face `Dataset` objects and NumPy structured arrays.
5. The preprocessed data goes into `src/bertpca/train.py:training_loop()`, which runs a manual `tf.GradientTape` loop (not `model.fit()`) with early stopping and LR reduction on plateau.

### Model (`src/bertpca/models.py`)

`build_bert_pca()` returns a compiled Keras model with this pipeline:
- Input shape: `(batch, n_features, seq_length)` — features-first, not the usual time-first
- Transformer encoder layers on the transposed input `(batch, seq_length, n_features)`, then transposed back
- Residual skip connection from the original transposed input
- 1D convolutional blocks (Conv1D + AveragePooling1D) on `(batch, seq_length, features)`
- Dense stack → Flatten
- Two output heads: `alpha` (scale) and `beta` (shape) for a Weibull distribution; output shape `(batch, 2)`
- Compiled with RMSprop and `weibull_loss`

### Loss and labels (`src/bertpca/loss.py`)

`weibull_loss` expects `y_true` of shape `(batch, 3)` = `[tte, event, t_last]` where times are **already scaled by `t_max`**. The loss computes the Weibull negative log-likelihood over the interval `[t_last, tte]`, not from time 0.

Additional losses (`coxph_loss`, `ranking_loss`, `survival_contrastive_loss`) are available but not wired into the default training pipeline.

### Evaluation (`src/bertpca/evaluation.py`, `src/bertpca/metrics.py`)

`calculate_time_dependent_c_index()` evaluates at a grid of `p_times` × `e_times`. It masks each patient's feature sequence to only include observations up to `p_time`, then computes Weibull hazard at `e_time`. The underlying `weighted_c_index()` applies IPCW (inverse probability of censoring weighting) estimated from the training set via Kaplan-Meier.

### Data augmentation (`src/bertpca/data.py:augment_dataframe`)

For each patient with `n` observations, creates `n-1` additional truncated copies of their timeline. This simulates making predictions at earlier follow-up points and is applied only to training data.

### Static vs. dynamic features

Both feature types are padded to `seq_length`. Static features (one value per patient) are tiled across the sequence: `repeat_first_column()` propagates the first column's values into later time steps wherever observations exist. Dynamic features vary per time step and are post-padded with zeros.

## Functional outcomes extension (`functional_outcomes/`)

Extends BertPCa to predict post-surgical **Erectile Function** (EF, IIEF EF domain ≥17) and **Urinary Continence** (UC, ICIQ=1) recovery from the Milan hospital dataset.

**Run order:**
```bash
# 1. Prepare datasets (from repo root)
python functional_outcomes/scripts/prepare_dataset.py

# 2. Train BertPCa on EF or UC
python functional_outcomes/scripts/train_functional.py --outcome ef
python functional_outcomes/scripts/train_functional.py --outcome uc

# 3. Run all baselines (CAPRA-S, MSKCC, CoxPH, RSF, DDH) + optional BertPCa comparison
python functional_outcomes/scripts/run_baselines.py --outcome all
python functional_outcomes/scripts/run_baselines.py --outcome ef --bertpca-model functional_outcomes/outputs/models/best_model_ef.keras
```

**Key design decisions:**
- Source: `data/Master_Prostate_Milan_2025-09-22.RData` → `dat.def` object.
- `tte` for BOTH recovered and censored patients comes directly from `ttIIEF_17`/`ttICIQ` (the dataset encodes the last assessment time for censored patients in the same column). No PSA-based censoring is used.
- PSA observations are **not** filtered by `tte` — the full PSA trajectory (capped at `t_max=365` days) provides context for the transformer, and PSA is masked at prediction time during evaluation by `calculate_time_dependent_c_index`.
- `t_max = 365` days for both outcomes (max observed `ttIIEF_17` ≈ 358 days; `ttICIQ` clipped at 365).
- `ece`, `svi`, `lni` are binarized (0→0, ≥1→1) before use.
- Median imputation for missingness (fit on train, applied to val/test).

**Baselines** (`functional_outcomes/src/baselines/`):
- `capras.py`: CAPRA-S score (Cooperberg 2011) — formula-based, no training.
- `mskcc.py`: MSKCC nomogram (Stephenson 2005) — published Cox coefficients.
- `coxph_rsf.py`: CoxPH and RSF via `scikit-survival` — trained on static features only.
- `ddh.py`: Dynamic-DeepHit — LSTM (PSA sequence) + Dense (static features) → discrete-time softmax, TF/Keras, same framework as BertPCa.

Install research dependencies: `pip install scikit-survival pyreadr` (or `uv sync --extra research` inside `bertpca/`).

## Data cleaning notebook (`data/1_data_cleaning.ipynb`)

This notebook prepares the real prostate cancer (BCR) dataset from the raw Milan hospital Excel file (`Master_Prostate_Milan_*.xlsx`, ~11k patients × 438 features) and produces the train/val/test CSVs consumed by the model. It is not part of the `bertpca` package and must be run manually in Jupyter before training on the BCR dataset.

**Pipeline steps:**

1. **Static feature cleaning**: drops patients with no PSA timepoints; forces numeric conversion; removes near-duplicate columns (>99% value match via `remove_matching_cols`); drops features with >30% missing values; removes patients with >20% missing feature values.

2. **Feature engineering**:
   - Gleason grade groups (biopsy `bxgg_group` and pathological `pathgg_group`) via `get_gleason_group()`
   - CAPRA Gleason component (`gleason_capra`) via `get_gleason_capra()`
   - `percposnodes` = positive / total nodes (raw counts then dropped)
   - `percposcore` imputed by mean within `clinstage2` group
   - `ece`, `svi`, `lni`, `nerve_sparing` binarized; `ece`/`svi` imputed from pathological stage
   - `histology_type` and `technique` one-hot encoded; `neo_adjHT_type` (170+ free-text values) and `complicanze_tipo` dropped

3. **PSA timeline construction** (wide → long format):
   - Converts `psa_1`…`psa_68` + `date_psa_1`…`date_psa_68` to days since surgery
   - Corrects known erroneous date substrings (e.g. `'2919'` → `'2019'`)
   - Removes readings with negative elapsed days and enforces strictly monotonic timestamps (iterative removal loop)
   - **Truncates each patient's sequence at first BCR event** (PSA ≥ 0.2 ng/mL) — readings after BCR are dropped
   - Caps follow-up at 15 years (`t_max = 15 * 365`)
   - Removes intervals where `start == stop`; keeps only patients with >1 follow-up
   - Produces a `start`/`stop`/`bcr`/`psa` long-format DataFrame (`df_outcomes_dyn`)

4. **Train/val/test split**: 80/10/10 split stratified by BCR status, applied before imputation to prevent leakage.

5. **Imputation** (`impute_and_scale()`): categorical columns → mode imputation; numerical columns → median imputation. Fit on training set only, then applied to val and test.

6. **Outputs**: `train_bcr_no_dummies.csv`, `val_bcr_no_dummies.csv`, `test_bcr_no_dummies.csv` (referenced in `data/old/`), plus `full_dataset_clean.csv` and `df_outcomes.csv`.

The BCR dataset uses static pre-surgical features (e.g. `pathgg_group`, `percposcore`, `percposnodes`, `tpsa`, `psm`, `svi`) and PSA as the single dynamic feature, matching the commented-out blocks in `config/config.yaml`.

## Switching datasets

The config file has commented-out blocks for the prostate cancer BCR dataset. Toggle between datasets by editing `config/config.yaml`: swap `train_path`/`val_path`/`test_path`, `static_features`, `dynamic_features`, `t_max`, `p_times`, and `e_times`. The `data/old/` directory holds the BCR CSV splits produced by the cleaning notebook.

## STKLM0 extension (`stklm0/`)

Cross-institutional evaluation using a new dataset (STKLM0 schema) for **cancer-specific mortality (CSM)** prediction. Two experimental steps:

**Step 1** — Train on Milan data, evaluate on STKLM0 (external validation):
```bash
# 1. Prepare Milan data with STKLM0-compatible features
python stklm0/scripts/prepare_milan.py --outcome bcr   # or csm or both

# 2. Prepare STKLM0 data
python stklm0/scripts/prepare_stklm0.py --input data/stklm0.csv

# 3. Train on Milan, evaluate on STKLM0
python stklm0/scripts/train_milan_eval_stklm0.py --outcome bcr
python stklm0/scripts/train_milan_eval_stklm0.py --outcome csm
```

**Step 2** — Train and evaluate entirely on STKLM0:
```bash
python stklm0/scripts/train_eval_stklm0.py
```

**Inference on new data** (STKLM0 schema, no training):
```bash
python stklm0/scripts/predict_stklm0.py \
    --input new_patients.csv \
    --model stklm0/outputs/models/best_model_stklm0.keras \
    --params stklm0/data/preprocessing_params.json \
    --e-times 365 1825 3650
```

**STKLM0 feature schema**: `d_diaage`, `d_spsa`, `isup_gealson`, `t_clean_ord`, `isup_RP`, `pT_ord`, `pR_bin`, `pRlenght`, `pN_bin` + dynamic `times`/`psa`. Outcome: `crmort` (0=censored, 1=CSM event, 2=other death→censored). `t_max = 3650` days (10 years).

**Feature dropping rule**: if a feature is not present in the Milan training data, it is dropped from that run — never imputed or filled.

**Preprocessing params** (`preprocessing_params.json`): saved by `prepare_stklm0.py`, contains imputer medians, `train_max`/`train_min`, `t_max`, `psa_max`. Used by `predict_stklm0.py` to apply identical scaling without re-fitting.

**Cross-dataset scaling**: in `train_milan_eval_stklm0.py`, STKLM0 test data is scaled using Milan train statistics (not STKLM0 statistics), so the model sees the same value ranges as during training.

## Key invariants

- **`times` is never feature-scaled** — it is divided by `t_max` separately in `load_and_preprocess_data()`.
- **Scaling parameters come from training data only** — `train_max`/`train_min` are applied to val and test.
- **Structured label arrays** for survival metrics use `dtype=[('Status', '?'), ('Survival_in_days', '<f8')]`; `_normalize_labels()` in `evaluation.py` converts between formats.
- **`weibull_loss` must be registered** via `@register_keras_serializable()` for model serialization to `.keras` format.
