# 🔒 HyPSA — Hybrid Password Strength Analyzer

> A multi-layer hybrid framework for password strength estimation combining **Shannon Entropy**, **Ensemble Machine Learning**, and **Rule-Based Fallback Penalties**.

**Supervised by:** Dr. Suneeta Satpathy  
**Academic Year:** 2025–2026 | Final Year Research Project (B.Tech CSE)

---

## 📌 Abstract

Traditional password strength meters rely on simple rule-based checks (character counts, symbol presence) that fail against semantically predictable passwords like `P@ssw0rd!` or `Qwerty2025!`. HyPSA addresses this by combining three independent analysis layers:

1. **Layer 1 — Statistical Randomness**: Normalized Shannon Entropy, password length, and maximum character repetition rate.
2. **Layer 2 — Structural Pattern Detection**: Sequence detection (e.g., `123`, `abc`, `1357`) and keyboard adjacency scoring (e.g., `qwerty`, `asdfgh`).
3. **Layer 3 — Semantic Similarity**: Fuzzy dictionary matching against 10,000 most common leaked credentials (RapidFuzz + RockYou corpus).

These features feed into a **weighted ensemble** of Logistic Regression, Random Forest, and XGBoost, whose output is then combined with the entropy score and post-processed through rule-based hard penalties to produce a final hybrid security score.

---

## 🏗️ System Architecture

```
Password Input
     │
     ▼
┌─────────────────────────────────────┐
│         Feature Extraction          │
│  ┌───────────┐  ┌────────────────┐  │
│  │  Layer 1  │  │    Layer 2     │  │
│  │ Entropy   │  │ Seq + Keyboard │  │
│  │ Length    │  │    Patterns    │  │
│  │ Repetition│  └────────────────┘  │
│  └───────────┘  ┌────────────────┐  │
│                 │    Layer 3     │  │
│                 │ Dict Similarity│  │
│                 └────────────────┘  │
└─────────────────┬───────────────────┘
                  │  6 normalized features [0–1]
                  ▼
┌─────────────────────────────────────┐
│         ML Ensemble (Weighted)      │
│  LR (20%) + RF (50%) + XGB (30%)   │
└─────────────────┬───────────────────┘
                  │  ML Score
                  ▼
┌─────────────────────────────────────┐
│  Hybrid Score = 0.4×Entropy         │
│              + 0.6×ML Score         │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│    Rule-Based Penalty Layer         │
│  (Dictionary + Pattern hard rules)  │
└─────────────────┬───────────────────┘
                  │
                  ▼
         WEAK / MEDIUM / STRONG
```

---

## 📊 Model Performance

> Run `python model.py` to reproduce all results. Outputs saved to `results/`.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | — | — |
| Random Forest | — | — |
| XGBoost | — | — |
| **Ensemble (LR+RF+XGB)** | **—** | **—** |

> ⚠️ Run `python model.py` after `python freezeDataset.py` to populate this table with your actual results.

### Feature Importance (Random Forest)
| Rank | Feature | Importance |
|---|---|---|
| 1 | Shannon Entropy | ~67% |
| 2 | Max Repetition | ~9% |
| 3 | Length | ~7% |
| 4 | Dictionary Score | ~varies |
| 5 | Keyboard Score | ~varies |
| 6 | Sequence Count | ~varies |

*Source: `analysis/feature_analysis.txt`*

---

## 📁 Project Structure

```
HyPSA-main/
│
├── app.py                        # Streamlit web application (main demo)
├── model.py                      # Model training, evaluation, and saving
├── features.py                   # All feature extraction logic (3 layers)
├── entropy.py                    # Shannon entropy implementation
├── data_pipeline.py              # Dataset generation from RockYou + synthetic
├── freezeDataset.py              # Freeze dataset once for reproducibility
├── predict.py                    # CLI password prediction tool
├── hyperparameterTuning.py       # Exhaustive ensemble weight grid search
├── model_ablation_studies.py     # Ablation study: single vs dual vs triple ensemble
├── computational_cost_analysis.py# Training time, inference time, memory, CPU
├── test_features.py              # Unit tests for feature extraction
│
├── datasets/
│   ├── 10000_common_passwords.csv  # Dictionary for semantic matching
│   └── fixed_password_dataset.csv  # Frozen training dataset (generated)
│
├── models/                       # Saved trained model files (.pkl)
│   ├── lr_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── analysis/                     # Feature analysis scripts and outputs
├── hyperparameterTuning/         # Tuning results CSV
└── results/                      # Confusion matrices, feature importance plots
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- RockYou dataset (`rockyou.txt`) placed in `datasets/` for dataset generation

### Install Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- `scikit-learn` — ML models and preprocessing
- `xgboost` — Gradient boosting classifier
- `rapidfuzz` — Fast fuzzy string matching for dictionary layer
- `streamlit` — Web application interface
- `pandas`, `numpy`, `matplotlib`, `seaborn` — Data processing and visualization

---

## 🚀 How to Run

### Step 1 — Generate and Freeze the Training Dataset
> Only needed once. Requires `datasets/rockyou.txt`.
```bash
python freezeDataset.py
```

### Step 2 — Train All Models
```bash
python model.py
```
Outputs confusion matrices and feature importance plots to `results/`.

### Step 3 — Launch the Web Application
```bash
streamlit run app.py
```

### Step 4 (Optional) — CLI Mode
```bash
python predict.py
```

---

## 🔬 Research Scripts

| Script | Purpose |
|---|---|
| `hyperparameterTuning.py` | Grid search over all LR/RF/XGB weight combinations; finds optimal (0.2, 0.5, 0.3) |
| `model_ablation_studies.py` | Tests every model subset (LR, RF, XGB, LR+RF, RF+XGB, LR+XGB, LR+RF+XGB) |
| `computational_cost_analysis.py` | Measures training time, inference latency, peak memory, and CPU per model |
| `test_features.py` | Unit tests validating feature extraction correctness on known password samples |
| `analysis/feature_analysis.py` | Random Forest importance + Mutual Information ranking across all features |
| `analysis/correlation_analysis.py` | Pairwise Pearson correlation between all extracted features |

---

## 🧪 Key Design Decisions

### Why Shannon Entropy over Theoretical Entropy?
Theoretical entropy overestimates strength by assuming a **uniform distribution** across all possible characters. Shannon entropy instead measures the **actual observed randomness** within the specific password string, making it a more accurate representation of information-theoretic unpredictability.

### Why Remove Character Count Features?
Early feature analysis (`analysis/feature_analysis.txt`) showed that lowercase count, uppercase count, digit count, and special character count have **low mutual information** and **high redundancy** with Shannon Entropy (which already captures character diversity). Removing them reduces feature dilution and improves model interpretability.

### Why a Weighted Ensemble (not a single model)?
Ablation studies (`model_ablation_studies.py`) confirmed that the LR+RF+XGB combination consistently outperforms any individual model or dual-model combination. Weights (LR=0.2, RF=0.5, XGB=0.3) were determined via exhaustive grid search (`hyperparameterTuning.py`).

### Why a Frozen Dataset?
Dataset generation uses RockYou (real weak passwords) plus synthetically generated medium/strong passwords. To ensure **reproducibility**, the dataset is frozen once via `freezeDataset.py` and reused across all experiments, preventing training variance from re-generation.

### Why Rule-Based Penalties on Top of ML?
ML models learn statistical patterns but cannot enforce hard security constraints. For example, a password may score moderate ML probability for "medium" class but still contain an exact dictionary word — a condition that should deterministically reduce the score. Rule-based penalties handle these adversarial edge cases that statistical models may miss.

---

## 📈 Dataset

| Class | Source | Count |
|---|---|---|
| Weak | RockYou corpus (real leaked) | ~15,000 |
| Weak | Mutated weak passwords (leet substitutions + symbols) | ~10,000 |
| Weak | Synthetically generated tricky patterns | ~10,000 |
| Medium | Synthetic word + number + symbol combos | ~35,000 |
| Strong | Cryptographically random character combinations | ~35,000 |
| **Total** | | **~105,000** |

Stratified 80/20 train-test split with `random_state=42` for reproducibility.

---

## ⚠️ Known Limitations

- **Passphrase gap**: Long multi-word passphrases (e.g., `correct-horse-battery-staple`) may be underscored due to dictionary word presence, despite high entropy. Future work could add a passphrase detection layer.
- **Language scope**: Dictionary is English-only; non-English common passwords are not covered.
- **Context-agnostic**: The system does not know the service being protected (e.g., banking vs. gaming). Context-aware strength policies are not implemented.
- **Leet-speak variations**: While mutated passwords are included in training, novel leet-speak patterns may not always be detected.

---

## 📄 License

This project is developed for academic research purposes under the supervision of Dr. Suneeta Satpathy, School of Computer Engineering, KIIT University.