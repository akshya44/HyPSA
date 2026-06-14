"""
Computational Cost Analysis
UPDATED VERSION:
Uses FIXED frozen dataset instead of rebuilding dataset every run
"""

import time
import os
import psutil
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# =====================================================
# STEP 1: LOAD FIXED FROZEN DATASET
# =====================================================

print("\n====================================")
print("Loading Fixed Frozen Dataset...")
print("====================================")

df = pd.read_csv(DATASET_PATH)

print("\nDataset Loaded Successfully")
print("Shape:", df.shape)

# Split features and labels
X = df.drop("label", axis=1)
y = df["label"]


# =====================================================
# STEP 2: TRAIN-TEST SPLIT
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

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
}


# =====================================================
# STEP 4: HELPER FUNCTION
# =====================================================

def measure_model_cost(model, model_name):

    print(f"\nRunning: {model_name}")

    cpu_before = psutil.cpu_percent(interval=1)

    def train():
        model.fit(X_train, y_train)

    process = psutil.Process()

    memory_before = process.memory_info().rss / (1024 * 1024)

    start_train = time.time()

    train()

    end_train = time.time()

    memory_after = process.memory_info().rss / (1024 * 1024)

    training_time = end_train - start_train
    peak_memory = max(memory_before, memory_after)

    cpu_after = psutil.cpu_percent(interval=1)
    cpu_avg = (cpu_before + cpu_after) / 2

    start_test = time.time()

    predictions = model.predict(X_test)

    end_test = time.time()

    inference_time = end_test - start_test
    per_sample_time = inference_time / len(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return {
        "Model": model_name,
        "Training Time (s)": round(training_time, 6),
        "Inference Time (s)": round(inference_time, 6),
        "Per Sample Time (s)": round(per_sample_time, 8),
        "Peak Memory (MB)": round(float(peak_memory), 6),
        "CPU Usage (%)": round(cpu_avg, 6),
        "Accuracy": round(accuracy, 6)
    }


# =====================================================
# STEP 5: PLOT FUNCTION
# =====================================================

def save_plot(results_df, column_name, filename, title, ylabel):
    plt.figure(figsize=(10, 6), dpi=300)

    bars = plt.bar(
        results_df["Model"],
        results_df[column_name]
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Models", fontsize=12)

    plt.xticks(rotation=0, fontsize=10, ha="center")
    plt.yticks(fontsize=10)

    plt.tight_layout()

    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.6f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    save_path = os.path.join(
        PLOTS_DIR,
        filename
    )

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {save_path}")


# =====================================================
# STEP 6: MAIN EXECUTION
# =====================================================

def main():
    results = []

    print("\n====================================")
    print("Running Computational Cost Analysis")
    print("====================================")

    for name, model in models.items():
        result = measure_model_cost(model, name)
        results.append(result)
    
    ensemble_result = measure_ensemble_cost()
    results.append(ensemble_result)

    results_df = pd.DataFrame(results)

    # ---------------------------------
    # Save CSV
    # ---------------------------------

    csv_path = os.path.join(
        CSV_DIR,
        "computational_cost_summary.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    print("\nCSV Saved Successfully")
    print(csv_path)

    # ---------------------------------
    # Generate Graphs
    # ---------------------------------

    save_plot(
        results_df,
        "Training Time (s)",
        "training_time_plot.png",
        "Training Time Comparison",
        "Seconds"
    )

    save_plot(
        results_df,
        "Inference Time (s)",
        "inference_time_plot.png",
        "Inference Time Comparison",
        "Seconds"
    )

    save_plot(
        results_df,
        "Peak Memory (MB)",
        "memory_usage_plot.png",
        "Memory Usage Comparison",
        "MB"
    )

    save_plot(
        results_df,
        "CPU Usage (%)",
        "cpu_usage_plot.png",
        "CPU Usage Comparison",
        "CPU Usage (%)"
    )

    save_plot(
        results_df,
        "Accuracy",
        "accuracy_plot.png",
        "Accuracy Comparison",
        "Accuracy Score"
    )

    # ---------------------------------
    # Final Results
    # ---------------------------------

    print("\n====================================")
    print("Final Results")
    print("====================================\n")

    print(results_df)

# =====================================================
# ENSEMBLE COST ANALYSIS
# =====================================================

def measure_ensemble_cost():

    print("\nRunning: Ensemble Model")

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

    cpu_before = psutil.cpu_percent(interval=1)

    def train():
        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)

    process = psutil.Process()

    memory_before = process.memory_info().rss / (1024 * 1024)

    start_train = time.time()

    train()

    end_train = time.time()

    memory_after = process.memory_info().rss / (1024 * 1024)

    training_time = end_train - start_train
    peak_memory = max(memory_before, memory_after)

    cpu_after = psutil.cpu_percent(interval=1)
    cpu_avg = (cpu_before + cpu_after) / 2

    start_test = time.time()

    lr_probs = lr.predict_proba(X_test)
    rf_probs = rf.predict_proba(X_test)
    xgb_probs = xgb.predict_proba(X_test)

    ensemble_probs = (
        0.2 * lr_probs +
        0.5 * rf_probs +
        0.3 * xgb_probs
    )

    predictions = ensemble_probs.argmax(axis=1)

    end_test = time.time()

    inference_time = end_test - start_test
    per_sample_time = inference_time / len(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return {
        "Model": "Ensemble Model",
        "Training Time (s)": round(training_time, 6),
        "Inference Time (s)": round(inference_time, 6),
        "Per Sample Time (s)": round(per_sample_time, 8),
        "Peak Memory (MB)": round(float(peak_memory), 6),
        "CPU Usage (%)": round(cpu_avg, 6),
        "Accuracy": round(accuracy, 6)
    }
# =====================================================
# REQUIRED FOR WINDOWS + memory_profiler
# =====================================================

if __name__ == "__main__":
    main()