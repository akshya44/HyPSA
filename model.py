"""
This file will:
- Cross-validate all models (5-fold stratified CV) to confirm generalization
- Train Models on the full training set
- Evaluate on the held-out test set
- Save trained models and evaluation plots

Uses a FIXED frozen dataset for reproducibility.
Run freezeDataset.py once before running this file.
"""

import os
import joblib
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


# =====================================================
# PATH SETUP
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "fixed_password_dataset.csv"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def cross_validate_models():
    """
    Runs 5-fold Stratified Cross-Validation on each individual model and the
    weighted ensemble to confirm that results are not specific to a single
    train/test split.

    Reports mean +/- std for both Accuracy and Weighted F1 Score.
    """

    print("\n====================================")
    print("5-Fold Stratified Cross-Validation")
    print("====================================")

    df = pd.read_csv(DATASET_PATH)
    X = df.drop("label", axis=1)
    y = df["label"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_weighted"]

    # Logistic Regression needs scaling — wrap in a Pipeline so CV handles it cleanly
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", CalibratedClassifierCV(
            LogisticRegression(class_weight="balanced", max_iter=1000),
            method="sigmoid"
        ))
    ])

    rf = RandomForestClassifier(random_state=42)

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )

    models_cv = [
        ("Logistic Regression", lr_pipeline),
        ("Random Forest",       rf),
        ("XGBoost",             xgb),
    ]

    cv_results_summary = []

    for name, model in models_cv:
        results = cross_validate(model, X, y_enc, cv=cv, scoring=scoring, n_jobs=-1)

        mean_acc = results["test_accuracy"].mean()
        std_acc  = results["test_accuracy"].std()
        mean_f1  = results["test_f1_weighted"].mean()
        std_f1   = results["test_f1_weighted"].std()

        print(f"\n{name}")
        print(f"  Accuracy : {mean_acc:.4f} +/- {std_acc:.4f}")
        print(f"  F1 Score : {mean_f1:.4f} +/- {std_f1:.4f}")

        cv_results_summary.append({
            "Model":       name,
            "Accuracy":    f"{mean_acc:.4f} +/- {std_acc:.4f}",
            "F1 (weighted)": f"{mean_f1:.4f} +/- {std_f1:.4f}"
        })

    print("\n====================================")
    print("Cross-Validation Summary Table")
    print("====================================")
    summary_df = pd.DataFrame(cv_results_summary)
    print(summary_df.to_string(index=False))

    # Save CV results
    cv_csv_path = os.path.join(RESULTS_DIR, "cross_validation_results.csv")
    summary_df.to_csv(cv_csv_path, index=False)
    print(f"\nCV results saved to: {cv_csv_path}")


def train_models():

    # =====================================================
    # LOAD FIXED FROZEN DATASET
    # =====================================================

    print("\n====================================")
    print("Loading Fixed Frozen Dataset...")
    print("====================================")

    df = pd.read_csv(DATASET_PATH)

    print(f"\nDataset Loaded Successfully")
    print(f"Shape: {df.shape}")

    # Features + Label split
    X = df.drop("label", axis=1)
    y = df["label"]

    # =====================================================
    # LABEL ENCODING (MULTI-CLASS FIX)
    # =====================================================

    le = LabelEncoder()
    y = le.fit_transform(y)

    # =====================================================
    # STRATIFIED SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # =====================================================
    # SCALING (ONLY FOR LOGISTIC REGRESSION)
    # =====================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =====================================================
    # MODELS
    # =====================================================

    lr = CalibratedClassifierCV(
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        ),
        method="sigmoid"
    )

    rf = RandomForestClassifier(
        random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )

    # =====================================================
    # TRAINING
    # =====================================================

    print("\nTraining Logistic Regression...")
    lr.fit(X_train_scaled, y_train)

    print("Training Random Forest...")
    rf.fit(X_train, y_train)

    print("Training XGBoost...")
    xgb.fit(X_train, y_train)
    
    # =====================================================
    # ENSEMBLE PREDICTIONS
    # =====================================================

    print("\nGenerating Ensemble Predictions...")

    lr_probs = lr.predict_proba(X_test_scaled)
    rf_probs = rf.predict_proba(X_test)
    xgb_probs = xgb.predict_proba(X_test)

    # Equal weights
    ensemble_probs = (
        0.2 * lr_probs +
        0.5 * rf_probs +
        0.3 * xgb_probs
    )

    ensemble_pred = ensemble_probs.argmax(axis=1)

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    print("\n====================================")
    print("Feature Importance (Random Forest)")
    print("====================================")

    importances = rf.feature_importances_
    feature_names = X.columns

    for name, score in zip(feature_names, importances):
        print(f"{name}: {score:.4f}")

    plot_feature_importance(importances, feature_names)

    # =====================================================
    # EVALUATION
    # =====================================================

    evaluate(
        "Logistic Regression",
        lr,
        X_test_scaled,
        y_test
    )

    evaluate(
        "Random Forest",
        rf,
        X_test,
        y_test
    )

    evaluate(
        "XGBoost",
        xgb,
        X_test,
        y_test
    )
    # =====================================================
    # Ensemble Evaluation
    # =====================================================
    plot_confusion_matrix(
        y_test,
        ensemble_pred,
        "Ensemble"       
    )
    print("\n=============================================")
    print("Ensemble Model")
    print("\n=============================================")
    
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            ensemble_pred
        )
    )
    print(
        "Confusion Matrix: \n",
        confusion_matrix(
            y_test,
            ensemble_pred
        )
    )
    # =====================================================
    # SAVE MODELS
    # =====================================================

    joblib.dump(
        lr,
        os.path.join(MODEL_DIR, "lr_model.pkl")
    )

    joblib.dump(
        rf,
        os.path.join(MODEL_DIR, "rf_model.pkl")
    )

    joblib.dump(
        xgb,
        os.path.join(MODEL_DIR, "xgb_model.pkl")
    )

    joblib.dump(
        scaler,
        os.path.join(MODEL_DIR, "scaler.pkl")
    )

    joblib.dump(
        le,
        os.path.join(MODEL_DIR, "label_encoder.pkl")
    )

    print("\n====================================")
    print("Models Saved Successfully")
    print("====================================")


# =====================================================
# CONFUSION MATRIX PLOT
# =====================================================

def plot_confusion_matrix(y_test, y_pred, title):

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Weak", "Medium", "Strong"],
        yticklabels=["Weak", "Medium", "Strong"]
    )

    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()

    save_path = os.path.join(
        RESULTS_DIR,
        f"{title}_confusion_matrix.png"
    )

    plt.savefig(save_path)

    print(f"{title} confusion matrix saved.")

    plt.close()


# =====================================================
# FEATURE IMPORTANCE PLOT
# =====================================================

def plot_feature_importance(importances, feature_names):

    import numpy as np

    indices = np.argsort(importances)

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(8, 5))

    plt.barh(
        sorted_features,
        sorted_importances,
        color="skyblue"
    )

    plt.xlabel("Importance Score")
    plt.title(
        "Feature Importance of Password Features (Random Forest)"
    )

    plt.tight_layout()

    save_path = os.path.join(
        RESULTS_DIR,
        "feature_importance.png"
    )

    plt.savefig(save_path)

    print("Feature importance plot saved.")

    plt.close()


# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate(name, model, X_test, y_test):

    y_pred = model.predict(X_test)

    plot_confusion_matrix(
        y_test,
        y_pred,
        name
    )

    print(f"\n====================================")
    print(name)
    print("====================================")

    print(
        "Accuracy:",
        accuracy_score(y_test, y_pred)
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print(
        "Confusion Matrix:\n",
        confusion_matrix(y_test, y_pred)
    )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    cross_validate_models()  # Validate generalization first
    train_models()           # Then train and save final models