"""
Ensemble Model Ablation Study
FINAL CORRECT VERSION:
Uses FIXED frozen dataset + WEIGHTED ENSEMBLE VOTING

Final MLScore:
0.2 * Logistic Regression
0.5 * Random Forest
0.3 * XGBoost
"""

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

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

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")

os.makedirs(CSV_DIR, exist_ok=True)


# =====================================================
# STEP 1: LOAD FIXED FROZEN DATASET
# =====================================================

print("\n====================================")
print("Loading Fixed Frozen Dataset...")
print("====================================")

df = pd.read_csv(DATASET_PATH)

print("\nDataset Loaded Successfully")
print("Shape:", df.shape)

# Features + Label split
X = df.drop("label", axis=1)
y = df["label"]


# =====================================================
# STEP 2: TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================================
# STEP 3: DEFINE MODELS
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
# STEP 4: SINGLE MODEL EVALUATION
# =====================================================

def evaluate_single_model(model, name):
    print(f"\nRunning Single Model: {name}")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    print(f"Accuracy: {acc:.6f}")
    print(f"F1 Score: {f1:.6f}")

    return {
        "Experiment": name,
        "Accuracy": round(acc, 6),
        "F1 Score": round(f1, 6)
    }


# =====================================================
# STEP 5: WEIGHTED ENSEMBLE FUNCTION
# =====================================================

def evaluate_weighted_ensemble(use_lr, use_rf, use_xgb, name):
    print(f"\nRunning Ensemble Model: {name}")

    probs = []

    # ---------------------------------
    # Logistic Regression
    # Weight = 0.2
    # ---------------------------------

    if use_lr:
        lr.fit(X_train, y_train)
        lr_prob = lr.predict_proba(X_test)
        probs.append(0.2 * lr_prob)

    # ---------------------------------
    # Random Forest
    # Weight = 0.5
    # ---------------------------------

    if use_rf:
        rf.fit(X_train, y_train)
        rf_prob = rf.predict_proba(X_test)
        probs.append(0.5 * rf_prob)

    # ---------------------------------
    # XGBoost
    # Weight = 0.3
    # ---------------------------------

    if use_xgb:
        xgb.fit(X_train, y_train)
        xgb_prob = xgb.predict_proba(X_test)
        probs.append(0.3 * xgb_prob)

    # ---------------------------------
    # Final Weighted Probability
    # ---------------------------------

    final_prob = np.sum(probs, axis=0)

    final_preds = np.argmax(final_prob, axis=1)

    acc = accuracy_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds, average="weighted")

    print(f"Accuracy: {acc:.6f}")
    print(f"F1 Score: {f1:.6f}")

    return {
        "Experiment": name,
        "Accuracy": round(acc, 6),
        "F1 Score": round(f1, 6)
    }


# =====================================================
# STEP 6: RUN ENSEMBLE ABLATION STUDY
# =====================================================

print("\n====================================")
print("Running Ensemble Learning Ablation Study")
print("====================================")

results = []


# ---------------------------------
# Individual Models
# ---------------------------------

results.append(
    evaluate_single_model(
        lr,
        "Logistic Regression"
    )
)

results.append(
    evaluate_single_model(
        rf,
        "Random Forest"
    )
)

results.append(
    evaluate_single_model(
        xgb,
        "XGBoost"
    )
)


# ---------------------------------
# Dual Ensembles
# ---------------------------------

results.append(
    evaluate_weighted_ensemble(
        use_lr=True,
        use_rf=True,
        use_xgb=False,
        name="LR + RF"
    )
)

results.append(
    evaluate_weighted_ensemble(
        use_lr=False,
        use_rf=True,
        use_xgb=True,
        name="RF + XGBoost"
    )
)

results.append(
    evaluate_weighted_ensemble(
        use_lr=True,
        use_rf=False,
        use_xgb=True,
        name="LR + XGBoost"
    )
)


# ---------------------------------
# Triple Ensemble
# ---------------------------------

results.append(
    evaluate_weighted_ensemble(
        use_lr=True,
        use_rf=True,
        use_xgb=True,
        name="LR + RF + XGBoost"
    )
)


# =====================================================
# STEP 7: SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(results)

save_path = os.path.join(
    CSV_DIR,
    "model_ablation_results.csv"
)

results_df.to_csv(
    save_path,
    index=False
)

print("\n====================================")
print("Ensemble Learning Ablation Study Completed")
print("====================================\n")

print(results_df)

print(f"\nResults saved to:\n{save_path}")