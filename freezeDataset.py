import os
import pandas as pd

from data_pipeline import build_dataset

# =====================================
# Build dataset ONLY ONCE
# =====================================

print("\n[INFO] Building final fixed dataset...")

X, y = build_dataset()

# Convert safely
X = pd.DataFrame(X)

# Add label column
X["label"] = y

# =====================================
# Save dataset permanently
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)

SAVE_PATH = os.path.join(
    DATASET_DIR,
    "fixed_password_dataset.csv"
)

X.to_csv(
    SAVE_PATH,
    index=False
)

print("\n====================================")
print("Dataset Frozen Successfully")
print("====================================")
print(f"\nSaved to:\n{SAVE_PATH}")
print(f"\nShape: {X.shape}")