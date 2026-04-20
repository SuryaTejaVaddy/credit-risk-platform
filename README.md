# Credit Risk Intelligence Platform

End-to-end AI/ML system for credit default prediction and fraud detection.
No API keys required. Runs entirely in GitHub Codespaces.

## Quick Start (Codespaces)

1. Click the green **Code** button → **Codespaces** → **New codespace** (4-core)
2. Wait ~5 minutes for automatic setup
3. Run the full pipeline:
   ```bash
   python scripts/run_pipeline.py
   ```
4. Launch the dashboard:
   ```bash
   streamlit run dashboard/app.py --server.address 0.0.0.0
   ```
5. Open **Ports** tab → click globe icon next to port **8501**

## Architecture

| Layer | Tools |
|-------|-------|
| ML Models | XGBoost, LightGBM, Stacking Ensemble |
| Fraud Detection | Isolation Forest + Autoencoder (PyTorch) |
| Explainability | SHAP + Fairlearn |
| MLOps | MLflow + Evidently AI |
| API | FastAPI |
| Dashboard | Streamlit (6 pages) |

## Results

| Model | Val AUC | Val AP |
|-------|---------|--------|
| Logistic Regression | 0.776 | 0.521 |
| XGBoost | 0.801 | 0.573 |
| LightGBM | 0.804 | 0.581 |
| **Stacking Ensemble** | **0.812** | **0.597** |

## Project Structure

```
credit-risk-platform/
├── .devcontainer/devcontainer.json   # Codespaces config
├── scripts/
│   ├── 01_download_data.py           # Generate datasets
│   ├── 02_train_models.py            # Credit scoring models
│   ├── 03_fraud_detection.py         # Fraud detection models
│   ├── 04_explainability.py          # SHAP + fairness audit
│   └── run_pipeline.py               # Run all steps at once
├── src/
│   ├── features/engineer.py          # Feature engineering
│   └── monitoring/drift_report.py    # Evidently drift report
├── api/main.py                       # FastAPI endpoints
├── dashboard/app.py                  # Streamlit dashboard (6 pages)
├── tests/test_pipeline.py            # Pytest suite
└── data/
    ├── processed/                    # Features + plots (committed)
    └── feature_store/                # Trained models (committed)
```

## Branch Strategy

- `main` — stable, production-ready
- `feature/data-pipeline` — ETL + feature engineering
- `feature/ml-models` — credit scoring + fraud detection
- `feature/nlp-explainability` — SHAP + fairness
- `feature/mlops` — MLflow + Evidently + tests
- `feature/dashboard` — Streamlit + FastAPI
