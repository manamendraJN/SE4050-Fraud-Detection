"""
SE4050 - Credit Card Fraud Detection
Shared Preprocessing Script

IMPORTANT: This script should be run ONCE and its outputs (the split CSVs)
shared with the whole team via the repo. Everyone must train/validate/test
on these exact same splits so the 4 models are fairly comparable.

Usage:
    python 01_preprocessing.py

Outputs (written to ./data/processed/):
    train.csv, val.csv, test.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

RANDOM_SEED = 42
DATA_PATH = "data/creditcard.csv"      # adjust path once in repo
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Load
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)
print("Class distribution:\n", df["Class"].value_counts())

# ---------------------------------------------------------
# 2. Data cleaning
# ---------------------------------------------------------
n_before = len(df)
df = df.drop_duplicates()
print(f"Removed {n_before - len(df)} duplicate rows")

assert df.isnull().sum().sum() == 0, "Unexpected missing values found"

# ---------------------------------------------------------
# 3. Feature engineering: Amount / Time
# ---------------------------------------------------------
# Log-transform Amount to reduce heavy right skew (add 1 to handle 0-value txns)
df["Amount_log"] = np.log1p(df["Amount"])

# ---------------------------------------------------------
# 4. Stratified split: 70% train, 15% val, 15% test
#    Stratify preserves the ~0.17% fraud ratio in every split
# ---------------------------------------------------------
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["Class"], random_state=RANDOM_SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["Class"], random_state=RANDOM_SEED
)

print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
    fraud_pct = split["Class"].mean() * 100
    print(f"  {name} fraud rate: {fraud_pct:.3f}%")

# ---------------------------------------------------------
# 5. Scaling — fit ONLY on training data to avoid leakage
#    Scale Time, Amount_log. V1-V28 are already PCA-transformed,
#    left as-is (standard practice for this dataset).
# ---------------------------------------------------------
scale_cols = ["Time", "Amount_log"]
scaler = StandardScaler()
train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
val_df[scale_cols] = scaler.transform(val_df[scale_cols])
test_df[scale_cols] = scaler.transform(test_df[scale_cols])

# Drop raw Amount now that we have Amount_log scaled
train_df = train_df.drop(columns=["Amount"])
val_df = val_df.drop(columns=["Amount"])
test_df = test_df.drop(columns=["Amount"])

# ---------------------------------------------------------
# 6. Save splits — everyone on the team loads these directly
# ---------------------------------------------------------
train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
val_df.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

print(f"\nSaved processed splits to {OUTPUT_DIR}/")
print("Random seed used:", RANDOM_SEED)
print("\nShare this script + the printed fraud rates with your team so")
print("everyone confirms they're using the identical splits.")
