import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs('data/feature_store', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, RocCurveDisplay)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.features.engineer import engineer_credit_features

print('=== Credit Scoring Model Training ===')

df_raw = pd.read_csv('data/raw/credit_default.csv')
df     = engineer_credit_features(df_raw)
df.to_csv('data/processed/credit_features.csv', index=False)
print(f'Features engineered: {df.shape[1]} columns')

X = df.drop(columns=['ID', 'target'], errors='ignore')
y = df['target']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42, stratify=y_temp)
print(f'Train:{len(X_train)}  Val:{len(X_val)}  Test:{len(X_test)}')

X_bal, y_bal = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_train, y_train)

scaler = StandardScaler()
Xtr    = scaler.fit_transform(X_bal)
Xv     = scaler.transform(X_val)
Xte    = scaler.transform(X_test)
joblib.dump(scaler, 'data/feature_store/scaler.pkl')
print('Scaler saved.')

mlflow.set_tracking_uri('mlruns')
mlflow.set_experiment('credit-risk-scoring')
results, trained = {}, {}

def run(name, model):
    with mlflow.start_run(run_name=name):
        model.fit(Xtr, y_bal)
        p   = model.predict_proba(Xv)[:, 1]
        auc = roc_auc_score(y_val, p)
        ap  = average_precision_score(y_val, p)
        mlflow.log_metrics({'val_roc_auc': auc, 'val_avg_precision': ap})
        mlflow.sklearn.log_model(model, name)
    print(f'  {name:28s}  AUC={auc:.4f}  AP={ap:.4f}')
    return model, auc

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, C=0.1, random_state=42),
    'RandomForest':       RandomForestClassifier(n_estimators=150, max_depth=8,
                                                  class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost':            XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8, scale_pos_weight=4,
                                         random_state=42, eval_metric='logloss', verbosity=0),
    'LightGBM':           LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                          class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1),
}

print('\nTraining models...')
for name, m in models.items():
    tm, auc = run(name, m)
    results[name] = auc
    trained[name] = tm

print('\nTraining Stacking Ensemble...')
est = [
    ('xgb',  XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                             subsample=0.8, verbosity=0, random_state=42)),
    ('lgbm', LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                              verbose=-1, random_state=42, n_jobs=-1)),
    ('rf',   RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)),
]
stack = StackingClassifier(estimators=est,
                             final_estimator=LogisticRegression(C=1.0, max_iter=500),
                             cv=5, n_jobs=-1)
tm, auc = run('StackingEnsemble', stack)
results['StackingEnsemble'] = auc
trained['StackingEnsemble'] = tm

best_name  = max(results, key=results.get)
best_model = trained[best_name]
print(f'\nBest: {best_name}  val_AUC={results[best_name]:.4f}')

yp = best_model.predict_proba(Xte)[:, 1]
print(f'Test AUC: {roc_auc_score(y_test, yp):.4f}')
print(classification_report(y_test, (yp > 0.5).astype(int),
                              target_names=['No Default', 'Default']))

fig, ax = plt.subplots(figsize=(9, 6))
for name, m in trained.items():
    RocCurveDisplay.from_predictions(y_test, m.predict_proba(Xte)[:, 1], name=name, ax=ax)
ax.plot([0,1], [0,1], 'k--', lw=0.8)
ax.set_title('ROC Curves — All Models', fontsize=14)
plt.tight_layout()
plt.savefig('data/processed/roc_curves.png', dpi=150, bbox_inches='tight')
print('ROC curve saved.')

joblib.dump(best_model, 'data/feature_store/best_credit_model.pkl')
print(f'Model saved: data/feature_store/best_credit_model.pkl')
