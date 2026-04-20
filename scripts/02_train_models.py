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
import logging
logging.getLogger('mlflow').setLevel(logging.ERROR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, RocCurveDisplay, accuracy_score)
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print('CatBoost not installed, skipping.')

from src.features.engineer import engineer_credit_features, preprocess_credit_data

print('=== Credit Scoring Model Training (Advanced) ===')

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

# ── SMOTEENN: SMOTE + Edited Nearest Neighbors (cleans boundary noise) ─────────
print('Applying SMOTEENN (removes noisy boundary samples)...')
smoteenn = SMOTEENN(random_state=42, smote=SMOTE(random_state=42, k_neighbors=5))
X_bal, y_bal = smoteenn.fit_resample(X_train, y_train)
print(f'SMOTEENN: {len(X_train)} → {len(X_bal)} samples  '
      f'(default rate: {y_bal.mean():.3f})')

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
        mlflow.sklearn.log_model(sk_model=model, name=name)
    print(f'  {name:35s}  AUC={auc:.4f}  AP={ap:.4f}')
    return model, auc

# ── Baseline models ─────────────────────────────────────────────────────────────
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

# ── Optuna tuning: XGBoost ──────────────────────────────────────────────────────
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

# ── Optuna tuning: LightGBM ─────────────────────────────────────────────────────
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

# ── CatBoost with Optuna tuning ─────────────────────────────────────────────────
if HAS_CATBOOST:
    print('\nOptuna tuning CatBoost (40 trials)...')

    def catboost_objective(trial):
        p = {
            'iterations':        trial.suggest_int('iterations', 300, 700),
            'depth':             trial.suggest_int('depth', 4, 8),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'l2_leaf_reg':       trial.suggest_float('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'border_count':      trial.suggest_int('border_count', 32, 128),
        }
        m = CatBoostClassifier(**p, auto_class_weights='Balanced',
                               random_seed=42, verbose=0, thread_count=-1)
        m.fit(Xtr, y_bal)
        return roc_auc_score(y_val, m.predict_proba(Xv)[:, 1])

    cat_study = optuna.create_study(direction='maximize')
    cat_study.optimize(catboost_objective, n_trials=40, show_progress_bar=False)
    best_cat = CatBoostClassifier(**cat_study.best_params, auto_class_weights='Balanced',
                                   random_seed=42, verbose=0, thread_count=-1)
    tm, auc = run('CatBoost_Tuned', best_cat)
    results['CatBoost_Tuned'] = auc
    trained['CatBoost_Tuned'] = tm
    print(f'  Best CatBoost params: {cat_study.best_params}')

# ── Deep MLP (PyTorch) ──────────────────────────────────────────────────────────
print('\nTraining Deep MLP (PyTorch)...')

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TabularMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, depth=4, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.head   = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x).squeeze(1)


class MLPWrapper(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper around TabularMLP."""

    def __init__(self, in_dim, hidden=256, depth=4, dropout=0.3,
                 epochs=40, lr=3e-4, batch=512):
        self.in_dim  = in_dim
        self.hidden  = hidden
        self.depth   = depth
        self.dropout = dropout
        self.epochs  = epochs
        self.lr      = lr
        self.batch   = batch
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Xt = torch.FloatTensor(np.array(X)).to(device)
        yt = torch.FloatTensor(np.array(y)).to(device)
        pos_weight = torch.tensor([(yt == 0).sum() / (yt == 1).sum()]).to(device)
        ds     = TensorDataset(Xt, yt)
        loader = DataLoader(ds, batch_size=self.batch, shuffle=True)
        self.model_ = TabularMLP(self.in_dim, self.hidden, self.depth, self.dropout).to(device)
        opt  = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr * 5, steps_per_epoch=len(loader), epochs=self.epochs)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self.model_(xb), yb).backward()
                opt.step()
                sched.step()
        self.model_.eval()
        self.device_ = device
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            Xt = torch.FloatTensor(np.array(X)).to(self.device_)
            logits = self.model_(Xt).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


mlp = MLPWrapper(in_dim=Xtr.shape[1], hidden=256, depth=4, dropout=0.3,
                 epochs=40, lr=3e-4, batch=512)
mlp.fit(Xtr, y_bal)
p_mlp = mlp.predict_proba(Xv)[:, 1]
auc_mlp = roc_auc_score(y_val, p_mlp)
ap_mlp  = average_precision_score(y_val, p_mlp)
print(f'  {"DeepMLP":35s}  AUC={auc_mlp:.4f}  AP={ap_mlp:.4f}')
results['DeepMLP'] = auc_mlp
trained['DeepMLP'] = mlp

# ── Stacking Ensemble ───────────────────────────────────────────────────────────
print('\nTraining Stacking Ensemble (tuned base learners)...')
est_stack = [
    ('xgb',  best_xgb),
    ('lgbm', best_lgbm),
    ('rf',   RandomForestClassifier(n_estimators=200, max_depth=8,
                                     class_weight='balanced', random_state=42, n_jobs=-1)),
]
if HAS_CATBOOST:
    est_stack.append(('cat', best_cat))
stack = StackingClassifier(estimators=est_stack,
                             final_estimator=LogisticRegression(C=1.0, max_iter=500),
                             cv=5, n_jobs=-1, passthrough=True)
tm, auc = run('StackingEnsemble', stack)
results['StackingEnsemble'] = auc
trained['StackingEnsemble'] = tm

# ── Soft Voting Ensemble (manual average — avoids VotingClassifier clone issues) ─
print('\nBuilding Soft Voting Ensemble...')

base_models = [best_xgb, best_lgbm, mlp]
if HAS_CATBOOST:
    base_models.append(best_cat)


class ManualSoftVoter(BaseEstimator, ClassifierMixin):
    """Average predicted probabilities from a list of already-trained models."""
    _estimator_type = 'classifier'

    def __init__(self, models):
        self.models   = models
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        probs = np.mean([m.predict_proba(X) for m in self.models], axis=0)
        return probs

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


voter = ManualSoftVoter(base_models)
p_v   = voter.predict_proba(Xv)[:, 1]
auc_v = roc_auc_score(y_val, p_v)
ap_v  = average_precision_score(y_val, p_v)
with mlflow.start_run(run_name='SoftVotingEnsemble'):
    mlflow.log_metrics({'val_roc_auc': auc_v, 'val_avg_precision': ap_v})
print(f'  {"SoftVotingEnsemble":35s}  AUC={auc_v:.4f}  AP={ap_v:.4f}')
results['SoftVotingEnsemble'] = auc_v
trained['SoftVotingEnsemble'] = voter

# ── Select best model ───────────────────────────────────────────────────────────
best_name  = max(results, key=results.get)
best_model = trained[best_name]
print(f'\nBest: {best_name}  val_AUC={results[best_name]:.4f}')

# ── Threshold optimisation for accuracy ────────────────────────────────────────
probs_val = best_model.predict_proba(Xv)[:, 1]
best_thresh, best_acc = 0.5, 0.0
for t in np.arange(0.25, 0.75, 0.005):
    acc = accuracy_score(y_val, (probs_val > t).astype(int))
    if acc > best_acc:
        best_acc, best_thresh = acc, t
print(f'Optimal threshold: {best_thresh:.3f}  (val accuracy={best_acc:.4f})')
joblib.dump(best_thresh, 'data/feature_store/best_threshold.pkl')

# ── Test set evaluation ─────────────────────────────────────────────────────────
yp    = best_model.predict_proba(Xte)[:, 1]
ypred = (yp > best_thresh).astype(int)
print(f'Test AUC:      {roc_auc_score(y_test, yp):.4f}')
print(f'Test Accuracy: {accuracy_score(y_test, ypred):.4f}')
print(classification_report(y_test, ypred, target_names=['No Default', 'Default']))

# ── All model AUC summary ───────────────────────────────────────────────────────
print('\n--- Model Leaderboard ---')
for n, a in sorted(results.items(), key=lambda x: -x[1]):
    print(f'  {n:35s}  val_AUC={a:.4f}')

# ── ROC curves ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
for name, m in trained.items():
    RocCurveDisplay.from_predictions(y_test, m.predict_proba(Xte)[:, 1], name=name, ax=ax)
ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax.set_title('ROC Curves — All Models (Advanced Branch)', fontsize=14)
plt.tight_layout()
plt.savefig('data/processed/roc_curves.png', dpi=150, bbox_inches='tight')
print('ROC curve saved.')

joblib.dump(best_model, 'data/feature_store/best_credit_model.pkl')
print(f'Model saved: data/feature_store/best_credit_model.pkl')
