"""
Load configuration from config.yaml.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import yaml


def get_config_path(path=None):
    """Return path to config.yaml"""
    if path is not None:
        return path
    return Path(__file__).resolve().parent / "config.yaml"


def load_yaml_config(path=None):
    """
    Load and parse config.yaml. Paths under data_dir are expanded.
    Returns a namespace with attributes: SEED, TRAIN_PATH, VAL_PATH, TEST_PATH,
    STATIC_FEATURES, DYNAMIC_FEATURES, BATCH_SIZE, SEQ_LENGTH, T_MAX,
    AUGMENT_DATA, SCALE_FEATURES, MODEL_CONFIG, TRAINING_CONFIG,
    EVALUATION_CONFIG, TUNING_CONFIG, OUTPUT_DIR, MODEL_DIR, RESULTS_DIR.
    """
    config_path = get_config_path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_dir = raw.get("data_dir", "data")
    raw["train_path"] = os.path.join(data_dir, raw["train_path"])
    raw["val_path"] = os.path.join(data_dir, raw["val_path"])
    raw["test_path"] = os.path.join(data_dir, raw["test_path"])

    # Training config: inject batch_size from top level if not in training
    training = dict(raw["training"])
    if "batch_size" not in training:
        training["batch_size"] = raw["batch_size"]
    raw["training"] = training

    # Evaluation config: inject t_max from top level
    evaluation = dict(raw["evaluation"])
    if "t_max" not in evaluation:
        evaluation["t_max"] = raw["t_max"]
    raw["evaluation"] = evaluation

    cfg = SimpleNamespace(
        SEED=raw["seed"],
        TRAIN_PATH=raw["train_path"],
        VAL_PATH=raw["val_path"],
        TEST_PATH=raw["test_path"],
        DATA_DIR=raw["data_dir"],
        STATIC_FEATURES=raw["static_features"],
        DYNAMIC_FEATURES=raw["dynamic_features"],
        BATCH_SIZE=raw["batch_size"],
        SEQ_LENGTH=raw["seq_length"],
        T_MAX=raw["t_max"],
        AUGMENT_DATA=raw["augment_data"],
        SCALE_FEATURES=raw["scale_features"],
        MODEL_CONFIG=raw["model"],
        TRAINING_CONFIG=raw["training"],
        EVALUATION_CONFIG=raw["evaluation"],
        TUNING_CONFIG=raw["tuning"],
        OUTPUT_DIR=raw["output_dir"],
        MODEL_DIR=raw["model_dir"],
        RESULTS_DIR=raw["results_dir"],
    )
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    return cfg
