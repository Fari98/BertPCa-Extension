#!/usr/bin/env python3
"""
Re-evaluate saved EF and UC models with the fixed residual-time risk score.
Run from repo root:
  python functional_outcomes/scripts/reeval_models.py
"""
import os, sys, csv
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "bertpca", "src"))
sys.path.insert(0, os.path.join(_ROOT, "bertpca"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
from config.load_config import load_yaml_config
from bertpca import load_and_preprocess_data, calculate_time_dependent_c_index
from bertpca.loss import weibull_loss

_CFG_DIR   = os.path.join(_ROOT, "functional_outcomes", "config")
_MODEL_DIR = os.path.join(_ROOT, "functional_outcomes", "outputs", "models")
_RES_DIR   = os.path.join(_ROOT, "functional_outcomes", "outputs", "results")

for outcome in ["ef", "uc"]:
    print(f"\n{'='*50}\n  {outcome.upper()} — re-evaluation\n{'='*50}")

    cfg = load_yaml_config(os.path.join(_CFG_DIR, f"config_{outcome}_uri.yaml"))
    cfg.TRAIN_PATH = os.path.join(_ROOT, cfg.TRAIN_PATH)
    cfg.VAL_PATH   = os.path.join(_ROOT, cfg.VAL_PATH)
    cfg.TEST_PATH  = os.path.join(_ROOT, cfg.TEST_PATH)

    _, _, te, y_tr, _, y_te = load_and_preprocess_data(
        cfg.TRAIN_PATH, cfg.VAL_PATH, cfg.TEST_PATH,
        cfg.STATIC_FEATURES, cfg.DYNAMIC_FEATURES,
        cfg.SEQ_LENGTH, cfg.BATCH_SIZE, cfg.T_MAX,
        cfg.AUGMENT_DATA, cfg.SCALE_FEATURES,
    )

    model_path = os.path.join(_MODEL_DIR, f"pipeline_model_{outcome}.keras")
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        continue

    model = tf.keras.models.load_model(
        model_path, custom_objects={"weibull_loss": weibull_loss}
    )

    p_times = np.array(cfg.EVALUATION_CONFIG["p_times"])
    e_times = np.array(cfg.EVALUATION_CONFIG["e_times"])
    t_max   = cfg.EVALUATION_CONFIG["t_max"]

    res = calculate_time_dependent_c_index(
        np.array(te["features"]), y_tr, y_te, model,
        p_times=p_times, e_times=e_times, t_max=t_max, return_mean=False,
    )

    valid = np.where(res == -1.0, np.nan, res)
    mean_c = float(np.nanmean(valid))
    print(f"  Mean C-index: {mean_c:.4f}")
    print(f"  C-index table:")
    header = ["p_time"] + [f"e={int(e)}d" for e in e_times]
    print("  " + "\t".join(header))
    for i, p in enumerate(p_times):
        row = [f"{int(p)}d"] + [
            f"{res[i,j]:.4f}" if res[i,j] != -1.0 else "  -1  "
            for j in range(len(e_times))
        ]
        print("  " + "\t".join(row))

    out_dir = os.path.join(_RES_DIR, f"{outcome}_uri", "pipeline")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "c_index_table_reeval.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p_time"] + [f"e_time_{int(e)}" for e in e_times])
        for i, p in enumerate(p_times):
            w.writerow([int(p)] + [f"{res[i,j]:.6f}" for j in range(len(e_times))])
    with open(os.path.join(out_dir, "mean_c_index_reeval.txt"), "w") as f:
        f.write(f"{mean_c:.6f}\n")
    print(f"  Saved to {out_dir}")
