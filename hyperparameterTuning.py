import os
import itertools
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# =====================================================
# PATH SETUP
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "fixed_password_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "hyperparameterTuning"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# LOAD DATASET
# =====================================================

print("\n====================================")
print("Loading Fixed Dataset...")
print("====================================")

df = pd.read_csv(DATASET_PATH)

X = df.drop("label", axis=1)
y = df["label"]

print("Dataset Shape:", df.shape)


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# =====================================================
# SCALING FOR LR
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================================
# MODELS
# =====================================================

lr = LogisticRegression(
    max_iter=1000
)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"
)


# =====================================================
# TRAIN MODELS
# =====================================================

print("\nTraining Logistic Regression...")
lr.fit(X_train_scaled, y_train)

print("Training Random Forest...")
rf.fit(X_train, y_train)

print("Training XGBoost...")
xgb.fit(X_train, y_train)


# =====================================================
# GET PROBABILITIES
# =====================================================

lr_prob = lr.predict_proba(X_test_scaled)
rf_prob = rf.predict_proba(X_test)
xgb_prob = xgb.predict_proba(X_test)


# =====================================================
# GENERATE WEIGHT COMBINATIONS
# =====================================================

print("\nGenerating Weight Combinations...")

weights = np.arange(0.0, 1.1, 0.1)

weight_combinations = []

for w1, w2, w3 in itertools.product(weights, repeat=3):

    total = round(w1 + w2 + w3, 1)

    if total == 1.0:
        weight_combinations.append((w1, w2, w3))

print(f"Total Combinations: {len(weight_combinations)}")


# =====================================================
# EVALUATE COMBINATIONS
# =====================================================

results = []

best_f1 = 0
best_weights = None

print("\n====================================")
print("Running Ensemble Weight Tuning...")
print("====================================")

for w_lr, w_rf, w_xgb in weight_combinations:

    final_prob = (
        (w_lr * lr_prob) +
        (w_rf * rf_prob) +
        (w_xgb * xgb_prob)
    )

    final_pred = np.argmax(final_prob, axis=1)

    accuracy = accuracy_score(y_test, final_pred)

    precision = precision_score(
        y_test,
        final_pred,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        final_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        final_pred,
        average="weighted"
    )

    results.append({
        "LR Weight": w_lr,
        "RF Weight": w_rf,
        "XGB Weight": w_xgb,
        "Accuracy": round(accuracy, 6),
        "Precision": round(precision, 6),
        "Recall": round(recall, 6),
        "F1 Score": round(f1, 6)
    })

    if f1 > best_f1:
        best_f1 = f1
        best_weights = (w_lr, w_rf, w_xgb)


# =====================================================
# SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

csv_path = os.path.join(
    OUTPUT_DIR,
    "ensemble_weight_tuning_results.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


# =====================================================
# FINAL OUTPUT
# =====================================================

print("\n====================================")
print("TUNING COMPLETED")
print("====================================")

print("\nBest Weight Combination:")
print(f"LR  : {best_weights[0]}")
print(f"RF  : {best_weights[1]}")
print(f"XGB : {best_weights[2]}")

print(f"\nBest F1 Score: {best_f1:.6f}")

print(f"\nResults saved to:\n{csv_path}")

print("\nTop 10 Results:\n")
print(results_df.head(10))