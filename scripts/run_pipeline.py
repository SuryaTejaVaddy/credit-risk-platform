import subprocess
import sys

steps = [
    ('Downloading / generating data',   ['python', 'scripts/01_download_data.py']),
    ('Training credit scoring models',  ['python', 'scripts/02_train_models.py']),
    ('Training fraud detection models', ['python', 'scripts/03_fraud_detection.py']),
    ('Explainability & fairness audit', ['python', 'scripts/04_explainability.py']),
    ('FinBERT sentiment analysis',      ['python', 'scripts/05_sentiment_analysis.py']),
    ('Drift monitoring report',         ['python', 'src/monitoring/drift_report.py']),
]

print('\n' + '='*60)
print('  CREDIT RISK INTELLIGENCE PLATFORM — FULL PIPELINE')
print('='*60)

for i, (label, cmd) in enumerate(steps, 1):
    print(f'\n[{i}/{len(steps)}] {label}')
    print('-' * 50)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f'\nFAILED at step: {label}')
        sys.exit(1)

print('\n' + '='*60)
print('  PIPELINE COMPLETE')
print('='*60)
print('\nNext steps:')
print('  Run tests:      pytest tests/ -v')
print('  Dashboard:      streamlit run dashboard/app.py --server.address 0.0.0.0')
print('  MLflow UI:      mlflow ui --backend-store-uri ./mlruns --port 5000 --host 0.0.0.0')
print('  FastAPI:        uvicorn api.main:app --port 8000 --host 0.0.0.0')
