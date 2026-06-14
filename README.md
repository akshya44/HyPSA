<div align="center">

# 🔒 HyPSA — Hybrid Password Strength Analyzer

**A multi-layer hybrid framework for password strength estimation combining Shannon Entropy, Ensemble Machine Learning, and Rule-Based Fallback Penalties.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.3-006C85)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-Academic-green)](LICENSE)
[![KIIT University](https://img.shields.io/badge/KIIT-Final%20Year%20Project-purple)](https://kiit.ac.in)

> **Supervised by:** Dr. Suneeta Satpathy, School of Computer Engineering, KIIT University  
> **Academic Year:** 2025–2026 | B.Tech CSE — Final Year Research Project

</div>

---

## 📌 Abstract

Traditional password strength meters rely on simple rule-based checks (character counts, symbol presence) that fail against **semantically predictable** passwords like `P@ssw0rd!` or `Qwerty2025!`. HyPSA addresses this by combining **three independent analysis layers** into a unified hybrid scoring framework:

| Layer | Technique | What it catches |
|---|---|---|
| **Layer 1** — Statistical | Normalized Shannon Entropy, length, max repetition | Low-entropy strings, repeated characters |
| **Layer 2** — Structural | Sequence detection, keyboard adjacency scoring | `123abc`, `qwerty`, `asdfgh`, skip-step patterns |
| **Layer 3** — Semantic | Fuzzy dictionary matching (RapidFuzz + RockYou corpus) | Dictionary words, leet-speak variants, common credentials |

These six normalized features feed into a **weighted ensemble** of Logistic Regression, Random Forest, and XGBoost — whose output is blended with Shannon Entropy and post-processed through **rule-based hard penalties** to produce a final hybrid security score.

---

## 🏗️ System Architecture

```
                        Password Input
                              │
                              ▼
          ┌───────────────────────────────────────┐
          │            Feature Extraction          │
          │                                        │
          │  ┌─────────────┐  ┌────────────────┐  │
          │  │   Layer 1   │  │    Layer 2     │  │
          │  │─────────────│  │────────────────│  │
          │  │Shannon Ent. │  │Seq. Detection  │  │
          │  │Length       │  │Keyboard Graph  │  │
          │  │MaxRepetition│  └────────────────┘  │
          │  └─────────────┘  ┌────────────────┐  │
          │                   │    Layer 3     │  │
          │                   │────────────────│  │
          │                   │RapidFuzz Dict  │  │
          │                   │Exact + Fuzzy   │  │
          │                   └────────────────┘  │
          └──────────────────┬────────────────────┘
                             │  6 normalized features [0–1]
                             ▼
          ┌───────────────────────────────────────┐
          │       Weighted ML Ensemble             │
          │   LR (20%) + RF (50%) + XGB (30%)     │
          │   [Weights via exhaustive grid search] │
          └──────────────────┬────────────────────┘
                             │  ml_score ∈ [0, 1]
                             ▼
          ┌───────────────────────────────────────┐
          │   Hybrid Score = 0.4 × Entropy        │
          │              + 0.6 × ML Score         │
          └──────────────────┬────────────────────┘
                             │
                             ▼
          ┌───────────────────────────────────────┐
          │      Rule-Based Penalty Layer          │
          │  Dictionary + Structural hard rules   │
          └──────────────────┬────────────────────┘
                             │
                             ▼
                  ┌──────────────────┐
                  │  WEAK / MEDIUM / │
                  │     STRONG       │
                  └──────────────────┘
```

---

## 📊 Model Performance

> All results are from **5-fold Stratified Cross-Validation** on a frozen dataset of ~105,000 passwords.  
> Run `python model.py` to reproduce. Outputs saved to `results/`.

### Cross-Validation Results

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Logistic Regression | 92.11% ± 0.3% | 92.10% ± 0.3% |
| Random Forest | 92.11% ± 0.3% | 92.10% ± 0.3% |
| XGBoost | 92.11% ± 0.3% | 92.10% ± 0.3% |
| **Ensemble (LR+RF+XGB)** | **92.11%** | **92.10%** |

> Ensemble weights (LR=0.2, RF=0.5, XGB=0.3) were determined via exhaustive grid search (`hyperparameterTuning.py`).

### Feature Importance (Random Forest — from `analysis/feature_analysis.txt`)

| Rank | Feature | RF Importance | Mutual Information |
|---|---|---|---|
| 1 | Shannon Entropy | **67.42%** | 1.097 |
| 2 | Max Repetition | 9.10% | 0.300 |
| 3 | Length | 7.03% | 0.187 |
| 4 | Dictionary Score | ~varies | 0.009 |
| 5 | Keyboard Score | ~varies | 0.019 |
| 6 | Sequence Count | ~varies | 0.010 |

### Ablation Study — Impact of Removing Each Feature

| Experiment | Accuracy | F1 Score | Δ Accuracy |
|---|---|---|---|
| All Features (Baseline) | 92.11% | 92.10% | — |
| Without Shannon Entropy | 85.41% | 85.39% | **−6.70%** |
| Without Dictionary Score | 91.79% | 91.77% | −0.33% |
| Without Keyboard Score | 91.86% | 91.84% | −0.25% |
| Without Length | 92.07% | 92.05% | −0.04% |
| Without Max Repetition | 92.10% | 92.08% | −0.01% |
| Without Sequence Count | 92.21% | 92.19% | +0.10% |

> Shannon Entropy is by far the most critical feature — its removal causes a **6.70% accuracy drop**.

---

## 📁 Project Structure

```
HyPSA-main/
│
├── app.py                          # 🌐 Streamlit web application (main demo)
├── model.py                        # 🤖 Model training, evaluation, and saving
├── features.py                     # ⚙️  All feature extraction logic (3 layers)
├── entropy.py                      # 📐 Shannon entropy implementation
├── data_pipeline.py                # 🔄 Dataset generation (RockYou + synthetic)
├── freezeDataset.py                # 🧊 Freeze dataset once for reproducibility
├── predict.py                      # 🖥️  CLI password prediction tool
├── hyperparameterTuning.py         # 🔧 Exhaustive ensemble weight grid search
├── model_ablation_studies.py       # 🔬 Ablation: single vs dual vs triple ensemble
├── computational_cost_analysis.py  # ⏱️  Training time, inference, memory, CPU
├── test_features.py                # ✅  Unit tests for feature extraction
├── requirements.txt                # 📦 All Python dependencies
│
├── datasets/
│   ├── 10000_common_passwords.csv  # Dictionary for semantic matching (Layer 3)
│   └── fixed_password_dataset.csv  # Frozen training dataset (~105K samples)
│
├── models/                         # Saved trained model files (.pkl)
│   ├── lr_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── analysis/                       # Research output scripts and logs
│   ├── feature_analysis.py         # RF importance + Mutual Information ranking
│   ├── feature_analysis.txt        # Saved output from feature analysis run
│   ├── correlation_analysis.py     # Pairwise Pearson correlation
│   ├── ablation_study_oneByOne.py  # Per-feature ablation script
│   └── ablation_study_oneByOne.txt # Saved ablation results
│
├── hyperparameterTuning/
│   └── ensemble_weight_tuning_results.csv  # Grid search results
│
├── screenshots/
│   └── theoreticalEntropy vs shannonEntropy.png
│
└── results/                        # Confusion matrices, feature importance plots
```

---

## ⚙️ Setup & Installation

### Prerequisites
- **Python 3.9 or higher**
- RockYou dataset (`rockyou.txt`) placed in `datasets/` — only required for re-generating the training dataset from scratch

### 1. Clone the Repository

```bash
git clone https://github.com/akshya44/HyPSA.git
cd HyPSA
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.50.0 | Web application UI |
| `scikit-learn` | 1.6.1 | ML models & preprocessing |
| `xgboost` | 2.1.3 | Gradient boosting classifier |
| `rapidfuzz` | 3.14.3 | Fast fuzzy string matching |
| `pandas` | 2.3.3 | Data manipulation |
| `numpy` | 2.2.6 | Numerical computation |
| `matplotlib` / `seaborn` | latest | Result visualization |
| `joblib` | 1.5.3 | Model serialization |

---

## 🚀 How to Run

### ▶️ Option 1 — Launch the Streamlit Web App (Recommended)

The pre-trained models are already included in the `models/` folder. Just run:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to use the interactive analyzer.

---

### 🔁 Option 2 — Retrain Models from Scratch

> Only needed if you want to reproduce training from the frozen dataset.

**Step 1 — Generate & Freeze the Training Dataset** *(requires `datasets/rockyou.txt`)*
```bash
python freezeDataset.py
```

**Step 2 — Train All Models**
```bash
python model.py
```
This runs 5-fold cross-validation, trains LR + RF + XGBoost, evaluates on the held-out test set, saves confusion matrices and feature importance plots to `results/`, and saves all trained `.pkl` files to `models/`.

---

### 🖥️ Option 3 — CLI Mode

```bash
python predict.py
```

---

## 🔬 Research Scripts

| Script | Purpose |
|---|---|
| `hyperparameterTuning.py` | Grid search over all LR/RF/XGB weight combinations; identifies optimal (0.2, 0.5, 0.3) |
| `model_ablation_studies.py` | Tests every model subset: LR, RF, XGB, LR+RF, RF+XGB, LR+XGB, LR+RF+XGB |
| `computational_cost_analysis.py` | Measures training time, inference latency, peak memory, and CPU per model |
| `test_features.py` | Unit tests validating feature extraction on known password samples |
| `analysis/feature_analysis.py` | RF importance + Mutual Information ranking across all features |
| `analysis/correlation_analysis.py` | Pairwise Pearson correlation between extracted features |
| `analysis/ablation_study_oneByOne.py` | Per-feature removal ablation study |

---

## 🧠 Key Design Decisions

### Why Shannon Entropy over Theoretical Entropy?

Theoretical entropy overestimates strength by assuming a **uniform distribution** across all possible characters. Shannon entropy measures the **actual observed randomness** within the specific password string — a more accurate representation of information-theoretic unpredictability.

> Example: `aaaaaaa1` has high theoretical entropy (8-char mixed) but near-zero Shannon entropy (highly repetitive).

### Why Only 6 Features?

Early feature analysis (`analysis/feature_analysis.txt`) showed that **lowercase count, uppercase count, digit count, and special character count** have low mutual information and high redundancy with Shannon Entropy (which already captures character diversity). Removing them:
- Reduces feature dilution and noise
- Prevents misleading bias ("more symbols = stronger")
- Improves model interpretability

### Why a Weighted Ensemble?

Ablation studies (`model_ablation_studies.py`) confirmed the LR+RF+XGB combination consistently outperforms any individual or dual-model combination. The weights (LR=0.2, RF=0.5, XGB=0.3) were determined via exhaustive grid search.

### Why a Frozen Dataset?

Dataset generation uses real weak passwords (RockYou) + synthetically generated medium/strong passwords. To ensure **reproducibility across all experiments**, the dataset is frozen once via `freezeDataset.py` and reused — preventing training variance from re-generation.

### Why Rule-Based Penalties on Top of ML?

ML models learn statistical patterns but **cannot enforce hard security constraints**. A password may score moderate ML probability for "medium" class but still contain an exact dictionary word — a condition that should deterministically reduce the score. Rule-based penalties handle these adversarial edge cases that statistical models miss.

---

## 📈 Dataset Composition

| Class | Source | Approx. Count |
|---|---|---|
| Weak | RockYou corpus (real leaked passwords) | ~15,000 |
| Weak | Mutated weak passwords (leet substitutions + symbols) | ~10,000 |
| Weak | Synthetically generated tricky patterns | ~10,000 |
| Medium | Synthetic word + number + symbol combinations | ~35,000 |
| Strong | Cryptographically random character combinations | ~35,000 |
| **Total** | | **~105,000** |

- Stratified 80/20 train-test split with `random_state=42` for reproducibility.
- 5-fold Stratified Cross-Validation used for all model evaluation.

---

## 🖼️ Shannon Entropy vs Theoretical Entropy

![Shannon Entropy vs Theoretical Entropy](screenshots/theoreticalEntropy%20vs%20shannonEntropy.png)

> Shannon entropy reflects the actual information content of the password, while theoretical entropy overestimates strength for low-diversity passwords.

---

## ⚠️ Known Limitations

- **Passphrase gap**: Long multi-word passphrases (e.g., `correct-horse-battery-staple`) may be underscored due to dictionary word presence, despite high entropy. A passphrase detection layer could address this in future work.
- **Language scope**: The dictionary is English-only; non-English common passwords are not covered.
- **Context-agnostic**: The system does not know the service being protected (e.g., banking vs. gaming). Context-aware strength policies are not implemented.
- **Leet-speak variants**: While mutated passwords are included in training, novel leet-speak patterns may not always be detected perfectly.

---

## 🤝 Contributing

This is an academic research project. If you find issues or want to extend the work:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is developed for **academic research purposes** under the supervision of **Dr. Suneeta Satpathy**, School of Computer Engineering, KIIT University, Bhubaneswar, India.

---

<div align="center">

**HyPSA** © 2026 | KIIT University | Final Year Research Project (B.Tech CSE)

*"Combining mathematical rigor with machine intelligence to solve the human problem of password predictability."*

</div>