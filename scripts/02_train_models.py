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
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, RocCurveDisplay, accuracy_score)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.features.engineer import engineer_credit_features, preprocess_credit_data

print('=== Credit Scoring Model Training ===')

df_raw   = pd.read_csv('data/raw/credit_default.csv')
df_clean = preprocess_credit_data(df_raw)
df       = engineer_credit_features(df_clean)
df.to_csv('data/processed/credit_features.csv', index=False)
print(f'After preprocessing + engineering: {df.shape[1]} columns')

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

def run(name, model, Xtr_=None, y_=None):
    Xtr__ = Xtr_ if Xtr_ is not None else Xtr
    y__   = y_   if y_   is not None else y_bal
    with mlflow.start_run(run_name=name):
        model.fit(Xtr__, y__)
        p   = model.predict_proba(Xv)[:, 1]
        auc = roc_auc_score(y_val, p)
        ap  = average_precision_score(y_val, p)
        mlflow.log_metrics({'val_roc_auc': auc, 'val_avg_precision': ap})
        mlflow.sklearn.log_model(model, name)
    print(f'  {name:30s}  AUC={auc:.4f}  AP={ap:.4f}')
    return model, auc

# ── Baseline models ────────────────────────────────────────────────────────────
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, C=0.1, random_state=42),
    'RandomForest':       RandomForestClassifier(n_estimators=300, max_depth=10,
                                                  min_samples_leaf=2,
                                                  class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost':            XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8,
                                         scale_pos_weight=4, gamma=0.1,
                                         random_state=42, eval_metric='logloss', verbosity=0),
    'LightGBM':           LGBMClassifier(n_estimators=300, max_depth=7, learning_rate=0.05,
                                          num_leaves=63, min_child_samples=20,
                                          class_weight='balanced', random_state=42,
                                          verbose=-1, n_jobs=-1),
}

print('\nTraining baseline models...')
for name, m in models.items():
    tm, auc = run(name, m)
    results[name] = auc
    trained[name] = tm

# ── Optuna tuning: XGBoost ─────────────────────────────────────────────────────
print('\nOptuna tuning XGBoost (60 trials)...')

def xgb_objective(trial):
    p = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 600),
        'max_depth':         trial.suggest_int('max_depth', 3, 8),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
        'gamma':             trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.5, 3.0),
        'scale_pos_weight':  trial.suggest_float('scale_pos_weight', 2, 6),
    }
    m = XGBClassifier(**p, random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1)
    m.fit(Xtr, y_bal)
    return roc_auc_score(y_val, m.predict_proba(Xv)[:, 1])

xgb_study = optuna.create_study(direction='maximize')
xgb_study.optimize(xgb_objective, n_trials=60, show_progress_bar=False)
best_xgb = XGBClassifier(**xgb_study.best_params, random_state=42,
                           eval_metric='logloss', verbosity=0, n_jobs=-1)
tm, auc = run('XGBoost_Tuned', best_xgb)
results['XGBoost_Tuned'] = auc
trained['XGBoost_Tuned'] = tm
print(f'  Best XGBoost params: {xgb_study.best_params}')

# ── Optuna tuning: LightGBM ────────────────────────────────────────────────────
print('\nOptuna tuning LightGBM (60 trials)...')

def lgbm_objective(trial):
    p = {
        'n_estimators':       trial.suggest_int('n_estimators', 200, 600),
        'max_depth':          trial.suggest_int('max_depth', 3, 9),
        'learning_rate':      trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'num_leaves':         trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples':  trial.suggest_int('min_child_samples', 5, 50),
        'subsample':          trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':   trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':          trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda':         trial.suggest_float('reg_lambda', 0, 2.0),
    }
    m = LGBMClassifier(**p, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)
    m.fit(Xtr, y_bal)
    return roc_auc_score(y_val, m.predict_proba(Xv)[:, 1])

lgbm_study = optuna.create_study(direction='maximize')
lgbm_study.optimize(lgbm_objective, n_trials=60, show_progress_bar=False)
best_lgbm = LGBMClassifier(**lgbm_study.best_params, class_weight='balanced',
                             random_state=42, verbose=-1, n_jobs=-1)
tm, auc = run('LightGBM_Tuned', best_lgbm)
results['LightGBM_Tuned'] = auc
trained['LightGBM_Tuned'] = tm

# ── Stacking Ensemble ──────────────────────────────────────────────────────────
print('\nTraining Stacking Ensemble (tuned base learners)...')
est = [
    ('xgb',  best_xgb),
    ('lgbm', best_lgbm),
    ('rf',   RandomForestClassifier(n_estimators=200, max_depth=8,
                                     class_weight='balanced', random_state=42, n_jobs=-1)),
]
stack = StackingClassifier(estimators=est,
                             final_estimator=LogisticRegression(C=1.0, max_iter=500),
                             cv=5, n_jobs=-1, passthrough=True)
tm, auc = run('StackingEnsemble', stack)
results['StackingEnsemble'] = auc
trained['StackingEnsemble'] = tm

# ── Select best model ──────────────────────────────────────────────────────────
best_name  = max(results, key=results.get)
best_model = trained[best_name]
print(f'\nBest: {best_name}  val_AUC={results[best_name]:.4f}')

# ── Threshold optimisation for accuracy ───────────────────────────────────────
probs_val = best_model.predict_proba(Xv)[:, 1]
best_thresh, best_acc = 0.5, 0.0
for t in np.arange(0.3, 0.7, 0.01):
    acc = accuracy_score(y_val, (probs_val > t).astype(int))
    if acc > best_acc:
        best_acc, best_thresh = acc, t
print(f'Optimal threshold: {best_thresh:.2f}  (val accuracy={best_acc:.4f})')
joblib.dump(best_thresh, 'data/feature_store/best_threshold.pkl')

# ── Test set evaluation ────────────────────────────────────────────────────────
yp    = best_model.predict_proba(Xte)[:, 1]
ypred = (yp > best_thresh).astype(int)
print(f'Test AUC:      {roc_auc_score(y_test, yp):.4f}')
print(f'Test Accuracy: {accuracy_score(y_test, ypred):.4f}')
print(classification_report(y_test, ypred, target_names=['No Default', 'Default']))

# ── ROC curves ────────────────────────────────────────────────────────────────
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
