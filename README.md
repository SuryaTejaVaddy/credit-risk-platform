# Credit Risk Intelligence Platform

> An end-to-end AI/ML system for credit default prediction, fraud detection, financial sentiment analysis, and model explainability — built entirely in GitHub Codespaces with no API keys required.

---

## Table of Contents

1. [Situation — The Problem](#situation--the-problem)
2. [Task — What We Set Out to Build](#task--what-we-set-out-to-build)
3. [Action — How We Built It](#action--how-we-built-it)
4. [Result — What We Achieved](#result--what-we-achieved)
5. [Project Structure](#project-structure)
6. [Quick Start](#quick-start)
7. [Running Each Step](#running-each-step)
8. [Dashboard Pages](#dashboard-pages)
9. [API Reference](#api-reference)
10. [Branch Strategy](#branch-strategy)
11. [Tech Stack](#tech-stack)

---

## Situation — The Problem

Financial institutions lose billions every year to two compounding problems:

1. **Credit default** — lending to applicants who cannot repay, driven by poor risk assessment models that rely on simple scorecards or logistic regression.
2. **Payment fraud** — fraudulent transactions that slip through rule-based detection systems, especially on imbalanced datasets where fraud represents less than 0.2% of all transactions.

On top of these core risks:
- Existing models are **black boxes** — regulators and loan officers cannot explain why a loan was declined.
- Models trained on historical data become **stale** — data drift causes silent accuracy degradation.
- Many systems exhibit **demographic bias** — different accuracy rates across gender or age groups create legal and ethical exposure.

---

## Task — What We Set Out to Build

Design and deliver a production-grade credit intelligence platform that:

- Predicts the **probability of default** on the UCI Credit Default dataset (30,000 real applicants)
- Detects **payment fraud** on the OpenML European credit card dataset (284,807 real transactions, 0.17% fraud rate)
- Performs **financial sentiment analysis** using a transformer model (FinBERT) on financial text
- Provides **full explainability** via SHAP values at both global and individual prediction levels
- Passes a **fairness audit** across gender groups using Fairlearn
- Monitors **data drift** between training and inference distributions
- Exposes everything through a **7-page Streamlit dashboard** and a **FastAPI REST endpoint**
- Is fully reproducible in **GitHub Codespaces** — one command, no local setup

**Accuracy target:** Beat 79% baseline → reach 81%+ through preprocessing, feature engineering, and advanced ensembles.

---

## Action — How We Built It

### Step 1 — Real Data, Not Synthetic

Downloaded two real-world datasets automatically:

| Dataset | Source | Size | Class Imbalance |
|---|---|---|---|
| Credit Default | UCI ML Repository | 30,000 applicants | 22% default rate |
| Credit Card Fraud | OpenML (European bank) | 284,807 transactions | 0.17% fraud rate |

Both downloads include automatic fallback to synthetic data if the remote is unavailable, so the pipeline never breaks.

### Step 2 — Preprocessing & Feature Engineering (77 features from 38 raw)

Raw UCI data contains documented noise. Applied:

| Fix | Detail |
|---|---|
| Category correction | EDUCATION values 0/5/6 → collapsed to "Other (4)"; MARRIAGE value 0 → "Other (3)" |
| Payment signal split | PAY=-2 (no consumption) separated from PAY=-1 (paid duly) as binary flags |
| Winsorisation | Bill amounts clipped at 1st–99th percentile; payment amounts at 0–99th |
| Log transforms | Payment amounts and credit limit log1p-transformed to reduce right skew |
| Engineered features | `max_delay`, `consecutive_delays`, `delay_trend`, `utilization`, `pay_ratio`, `cumulative_gap`, `risk_score`, `nonpayment_score`, `delay_x_utilization`, and 15+ more |

### Step 3 — Credit Scoring Models

Trained a full model zoo with Optuna hyperparameter tuning:

| Model | Trials | Notes |
|---|---|---|
| Logistic Regression | — | Baseline |
| Random Forest | — | 300 trees, balanced class weights |
| XGBoost | 60 Optuna trials | scale_pos_weight tuned |
| LightGBM | 60 Optuna trials | num_leaves + min_child_samples tuned |
| CatBoost | 40 Optuna trials | auto_class_weights, Bayesian optimisation |
| Deep MLP (PyTorch) | — | 4 residual blocks, BatchNorm, GELU, OneCycleLR |
| Stacking Ensemble | — | XGB + LGBM + RF + CatBoost → Logistic meta-learner |
| Soft Voting Ensemble | — | XGB + LGBM + CatBoost + MLP probability average |

**Imbalance handling:** SMOTEENN (SMOTE oversampling + Edited Nearest Neighbours boundary cleaning), replacing plain SMOTE.

**Threshold optimisation:** Grid search over 0.25–0.75 (step 0.005) on validation set to maximise accuracy.

### Step 4 — Fraud Detection (3 complementary approaches)

| Model | Type | AUC |
|---|---|---|
| Isolation Forest | Unsupervised anomaly detection | 0.9539 |
| Random Forest | Supervised (SMOTE-balanced) | 0.9831 |
| PyTorch Autoencoder | Reconstruction-error anomaly score | 0.9615 |

### Step 5 — Explainability (SHAP)

- **Global importance** — SHAP bar chart showing top features across 500 test samples
- **Beeswarm plot** — directional feature impact distribution
- **Waterfall plot** — individual applicant decision decomposition
- Handles both TreeExplainer (tree models) and KernelExplainer (MLP/stacking) automatically

### Step 6 — Fairness Audit (Fairlearn)

Evaluated across gender groups (Male / Female):
- Demographic Parity Difference
- Equalized Odds Difference
- Per-group accuracy, precision, and recall

### Step 7 — FinBERT Sentiment Analysis

Real transformer inference using `ProsusAI/finbert`:
- 3,000 financial sentences classified as **positive / negative / neutral**
- Batch inference (batch size 32) with confidence scores
- Live sentence scorer in the dashboard

### Step 8 — Data Drift Monitoring (Evidently AI)

- Compares training vs. inference feature distributions
- HTML drift report generated automatically
- Mean-shift fallback if Evidently API version changes

---

## Result — What We Achieved

### Credit Default Prediction

| Branch / Experiment | Test Accuracy | Test AUC | Features |
|---|---|---|---|
| Baseline (main) | **79.0%** | 0.7708 | 38 |
| + Preprocessing improvements | **80.89%** | 0.7760 | 77 |
| + SMOTEENN + CatBoost + MLP + SoftVoting | **81.40%** | 0.7772 | 77 |

Best model leaderboard (advanced branch):

| Rank | Model | Val AUC |
|---|---|---|
| 1 | LightGBM_Tuned | 0.7806 |
| 2 | SoftVotingEnsemble | 0.7792 |
| 3 | XGBoost_Tuned | 0.7785 |
| 4 | RandomForest | 0.7741 |
| 5 | CatBoost_Tuned | 0.7738 |

> Note: The UCI Credit Default dataset has a well-documented information ceiling of ~82–83% accuracy. Published academic papers on this exact dataset report AUC in the 0.77–0.79 range. Our results are at the upper end of what is achievable on this data.

### Fraud Detection

| Model | AUC |
|---|---|
| Isolation Forest | 0.9539 |
| Random Forest | **0.9831** |
| PyTorch Autoencoder | 0.9615 |

### Fairness (Advanced Branch)

| Group | Accuracy | Precision | Recall |
|---|---|---|---|
| Male | 75.8% | 49.6% | 60.2% |
| Female | 79.9% | 51.8% | 55.8% |
| **Demographic Parity Diff** | | | **0.0658** |
| **Equalized Odds Diff** | | | **0.0556** |

Values near 0 indicate a fair model. Significant improvement over the baseline (0.25 / 0.25).

### FinBERT Sentiment

| Metric | Value |
|---|---|
| Sentences processed | 3,000 |
| Mean confidence | 89.1% |
| Positive rate | 50.0% |
| Negative rate | 50.0% |

### Test Suite

```
6/6 tests passed
```

---

## Project Structure

```
credit-risk-platform/
│
├── .devcontainer/
│   └── devcontainer.json          # Codespaces auto-setup (pip install -r requirements.txt)
│
├── scripts/
│   ├── 01_download_data.py        # Download UCI + OpenML datasets (with synthetic fallback)
│   ├── 02_train_models.py         # Full model training: baselines + Optuna tuning + ensembles
│   ├── 03_fraud_detection.py      # Isolation Forest + RF + PyTorch Autoencoder
│   ├── 04_explainability.py       # SHAP global/individual + Fairlearn audit
│   ├── 05_sentiment_analysis.py   # FinBERT inference on 3,000 financial sentences
│   └── run_pipeline.py            # Orchestrates all 6 steps end-to-end
│
├── src/
│   ├── features/
│   │   └── engineer.py            # preprocess_credit_data() + engineer_credit_features()
│   └── monitoring/
│       └── drift_report.py        # Evidently AI drift report (with fallback)
│
├── api/
│   └── main.py                    # FastAPI: POST /predict, GET /health
│
├── dashboard/
│   └── app.py                     # Streamlit 7-page dashboard
│
├── tests/
│   └── test_pipeline.py           # 6 pytest tests covering features + models
│
├── data/
│   ├── raw/                       # Downloaded datasets (gitignored)
│   ├── processed/                 # Feature CSV + plots (roc_curves, shap_*.png)
│   └── feature_store/             # Saved models + scaler + threshold (.pkl)
│
└── requirements.txt               # All dependencies pinned
```

---

## Quick Start

### GitHub Codespaces (Recommended)

1. Click **Code** → **Codespaces** → **New codespace** (select 4-core machine)
2. Wait ~3 minutes for the devcontainer to install dependencies automatically
3. Run the full pipeline:

```bash
python scripts/run_pipeline.py
```

4. Launch the dashboard:

```bash
streamlit run dashboard/app.py --server.address 0.0.0.0
```

5. Open the **Ports** tab → click the globe icon next to port **8501**

---

## Running Each Step

Run steps individually if you want to skip or re-run specific parts:

```bash
# Step 1 — Download real datasets (UCI credit + OpenML fraud)
python scripts/01_download_data.py

# Step 2 — Train all credit scoring models (takes ~20-30 min with Optuna)
python scripts/02_train_models.py

# Step 3 — Train fraud detection models
python scripts/03_fraud_detection.py

# Step 4 — SHAP explainability + Fairlearn fairness audit
python scripts/04_explainability.py

# Step 5 — FinBERT financial sentiment analysis
python scripts/05_sentiment_analysis.py

# Step 6 — Data drift monitoring report
python src/monitoring/drift_report.py

# Run all tests
pytest tests/ -v

# Launch dashboard
streamlit run dashboard/app.py --server.address 0.0.0.0

# Launch MLflow experiment tracker
mlflow ui --backend-store-uri ./mlruns --port 5000 --host 0.0.0.0

# Launch REST API
uvicorn api.main:app --port 8000 --host 0.0.0.0
```

---

## Dashboard Pages

| Page | What It Shows |
|---|---|
| **Overview** | Dataset stats, class distribution, default rate by age group |
| **Model Performance** | AUC leaderboard, ROC curves for all models |
| **Live Scoring** | Interactive applicant form → real-time default probability gauge |
| **SHAP Explainability** | Global feature importance, beeswarm, individual waterfall plot |
| **Fairness Audit** | Per-gender accuracy/precision/recall, actual vs predicted default rates |
| **Fraud Insights** | 284K transaction analysis, fraud rate by hour, amount distributions |
| **Sentiment NLP** | FinBERT label distribution, confidence histogram, live sentence scorer |

---

## API Reference

The FastAPI server exposes a prediction endpoint at `http://localhost:8000`.

### Health check

```bash
GET /health
```

### Predict default probability

```bash
POST /predict
Content-Type: application/json

{
  "LIMIT_BAL": 200000,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 35,
  "PAY_0": 0,
  "BILL_AMT1": 50000,
  "PAY_AMT1": 20000
  ... (full feature set)
}
```

Response:
```json
{
  "default_probability": 0.23,
  "decision": "APPROVE",
  "risk_level": "LOW"
}
```

---

## Branch Strategy

| Branch | Purpose | Key Change |
|---|---|---|
| `main` | Stable, production-ready | Merged best improvements |
| `feature/preprocessing-improvements` | Data cleaning + feature engineering | 38 → 77 features, 79% → 80.89% accuracy |
| `feature/advanced-modeling` | Advanced models + sampling | SMOTEENN + CatBoost + MLP + SoftVoting → 81.40% |

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.11 |
| **ML / Boosting** | XGBoost, LightGBM, CatBoost, scikit-learn |
| **Deep Learning** | PyTorch (tabular MLP, fraud autoencoder) |
| **NLP / Transformers** | HuggingFace Transformers, ProsusAI/finbert |
| **Imbalance Handling** | imbalanced-learn (SMOTEENN) |
| **Hyperparameter Tuning** | Optuna (60–40 trials per model) |
| **Explainability** | SHAP (TreeExplainer + KernelExplainer) |
| **Fairness** | Fairlearn (MetricFrame, demographic parity) |
| **Experiment Tracking** | MLflow |
| **Drift Monitoring** | Evidently AI |
| **API** | FastAPI + Uvicorn |
| **Dashboard** | Streamlit + Plotly |
| **Testing** | pytest |
| **Environment** | GitHub Codespaces (devcontainer) |

---

## Key Learnings

- **Data quality beats model complexity** — fixing EDUCATION/MARRIAGE encoding and winsorising bill amounts contributed more than switching from XGBoost to CatBoost
- **SMOTEENN > SMOTE** — removing noisy boundary samples after oversampling improved decision boundary clarity and fairness metrics
- **Ensemble diversity matters** — combining gradient boosting + neural network in the soft voter improved robustness over any single model
- **UCI ceiling is real** — the dataset's information limit is ~82–83% accuracy; academic papers confirm this. Chasing higher accuracy requires either different features or a richer dataset
- **Fairness improved with better models** — demographic parity difference dropped from 0.25 (baseline) to 0.07 (advanced branch) as a side effect of better calibration
