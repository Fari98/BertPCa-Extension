#!/usr/bin/env python3
"""
Preprocess PBC dataset: split into train/val/test by unique patient IDs.
Treats transplant/death as composite outcome (label 2 -> 1).
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")

pbc = pd.read_csv(os.path.join(data_dir, "pbc2_cleaned.csv"))

# Consider transplant/death as composite outcome
pbc["label"] = pbc["label"].replace({2: 1})

# Split by unique patient IDs
unique_ids = pbc["id"].unique()
train_ids, temp_ids = train_test_split(unique_ids, test_size=0.4, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

pbc_train = pbc[pbc["id"].isin(train_ids)].reset_index(drop=True)
pbc_val = pbc[pbc["id"].isin(val_ids)].reset_index(drop=True)
pbc_test = pbc[pbc["id"].isin(test_ids)].reset_index(drop=True)

print("Train shape:", pbc_train.shape)
print("Validation shape:", pbc_val.shape)
print("Test shape:", pbc_test.shape)

pbc_train.to_csv(os.path.join(data_dir, "pbc_train.csv"), index=False)
pbc_val.to_csv(os.path.join(data_dir, "pbc_val.csv"), index=False)
pbc_test.to_csv(os.path.join(data_dir, "pbc_test.csv"), index=False)
