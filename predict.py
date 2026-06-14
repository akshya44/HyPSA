import joblib
import pandas as pd
import numpy as np
from features import (
    extractFeatures,
    hasDictionaryPattern,
    hasKeyboardPattern,
    hasSequencePattern
)

# ===== LOAD MODELS =====
lr = joblib.load("models/lr_model.pkl")
rf = joblib.load("models/rf_model.pkl")
xgb = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ===== FEATURE ORDER =====
FEATURE_ORDER = [
    "length",
    "maxRepetition",
    "shannonEntropy",
    "sequenceCount",
    "dictionaryScore",
    "keyboardScore"
]


def predict_password(password):
    # ===== FEATURE EXTRACTION =====
    features = extractFeatures(password)
    X = pd.DataFrame(
        [[features[f] for f in FEATURE_ORDER]],
        columns=FEATURE_ORDER
    )

    # ===== MODEL PREDICTIONS =====
    X_scaled = scaler.transform(X)

    lr_prob = lr.predict_proba(X_scaled)[0]
    rf_prob = rf.predict_proba(X)[0]
    xgb_prob = xgb.predict_proba(X)[0]

    # ===== INDIVIDUAL MODEL SCORES =====
    # XGBoost is best → highest weight
    lr_score = (
        0.0 * lr_prob[0] +
        0.5 * lr_prob[1] +
        1.0 * lr_prob[2]
    )

    rf_score = (
        0.0 * rf_prob[0] +
        0.5 * rf_prob[1] +
        1.0 * rf_prob[2]
    )

    xgb_score = (
        0.0 * xgb_prob[0] +
        0.5 * xgb_prob[1] +
        1.0 * xgb_prob[2]
    )

    # ===== WEIGHTED ML SCORE =====
    # XGBoost highest weight because it is the best performing one
    ml_score = (
        0.2 * lr_score +
        0.5 * rf_score +
        0.3 * xgb_score
    )

    # ===== ENTROPY SCORE =====
    entropy_score = features["shannonEntropy"]

    # ===== BASE HYBRID SCORE =====
    final_score = (0.4 * entropy_score) + (0.6 * ml_score)

    # ===== RULE-BASED PENALTIES =====
    kb = hasKeyboardPattern(password)
    seq = hasSequencePattern(password)
    dict_flag = hasDictionaryPattern(password)

    penalty = 1.0
    reasons = []

    if dict_flag and (
        features["sequenceCount"] > 0.4 or
        features["keyboardScore"] > 0.4
    ):
        print(f"\nPassword: {password}")
        print("Final Verdict: WEAK")
        print("Reason: Highly predictable password (dictionary + pattern)")
        return

    if dict_flag and len(password) < 12:
        penalty *= 0.75
        reasons.append("Dictionary pattern detected")

    if features["sequenceCount"] > 0.4:
        penalty *= 0.9
        reasons.append("Sequential pattern detected")

    if features["keyboardScore"] > 0.4:
        penalty *= 0.9
        reasons.append("Minor Keyboard pattern detected")

    final_score *= penalty

    # ===== OUTPUT =====
    print(f"\nPassword: {password}")
    print(f"LR Score: {lr_score:.2f}")
    print(f"RF Score: {rf_score:.2f}")
    print(f"XGB Score: {xgb_score:.2f}")
    print(f"Entropy Score: {entropy_score:.2f}")
    print(f"ML Score: {ml_score:.2f}")
    print(f"Final Hybrid Score: {final_score:.2f}")

    # ===== FINAL DECISION =====
    if final_score >= 0.50:
        verdict = "STRONG"
    elif final_score >= 0.30:
        verdict = "MEDIUM"
    else:
        verdict = "WEAK"

    print("Final Verdict:", verdict)

    if reasons:
        print("Reason:", ", ".join(reasons))


# ===== CLI LOOP =====
if __name__ == "__main__":
    while True:
        pwd = input("\nEnter password (or 'exit'): ")

        if pwd.lower() == "exit":
            break

        predict_password(pwd)