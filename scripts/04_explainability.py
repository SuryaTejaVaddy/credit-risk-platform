import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs('data/processed', exist_ok=True)

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from fairlearn.metrics import (MetricFrame, demographic_parity_difference,
                                equalized_odds_difference)
from sklearn.metrics import accuracy_score, precision_score, recall_score

print('=== Explainability & Fairness ===')

df     = pd.read_csv('data/processed/credit_features.csv')
scaler = joblib.load('data/feature_store/scaler.pkl')
model  = joblib.load('data/feature_store/best_credit_model.pkl')

X = df.drop(columns=['ID', 'target'], errors='ignore')
y = df['target']
_, Xte, _, yte = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
Xte_sc = scaler.transform(Xte)
Xte_df = pd.DataFrame(Xte_sc, columns=X.columns)

print('Computing SHAP values (2-3 min)...')
try:
    exp = shap.TreeExplainer(model)
    sv  = exp.shap_values(Xte_df.head(500))
    if isinstance(sv, list):
        sv = sv[1]
    ev = exp.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = ev[1]
    print(f'SHAP values shape: {sv.shape}')
except Exception as e:
    print(f'TreeExplainer failed ({e}), using KernelExplainer...')
    bg  = shap.sample(Xte_df, 100)
    exp = shap.KernelExplainer(model.predict_proba, bg)
    sv  = exp.shap_values(Xte_df.head(100))
    sv  = sv[1] if isinstance(sv, list) else sv
    ev  = exp.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = ev[1]

Xp = Xte_df.head(len(sv))

plt.figure(figsize=(10, 8))
shap.summary_plot(sv, Xp, plot_type='bar', show=False)
plt.title('Global Feature Importance (SHAP)', fontsize=14)
plt.tight_layout()
plt.savefig('data/processed/shap_global.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: shap_global.png')

plt.figure(figsize=(10, 8))
shap.summary_plot(sv, Xp, show=False)
plt.title('SHAP Beeswarm', fontsize=14)
plt.tight_layout()
plt.savefig('data/processed/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: shap_beeswarm.png')

shap_exp = shap.Explanation(
    values=sv[5],
    base_values=float(ev),
    data=Xp.iloc[5].values,
    feature_names=Xp.columns.tolist()
)
plt.figure(figsize=(12, 5))
shap.plots.waterfall(shap_exp, show=False)
plt.tight_layout()
plt.savefig('data/processed/shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: shap_waterfall.png')

print('\n=== Fairness Audit ===')
ypred = (model.predict_proba(Xte_sc)[:, 1] > 0.5).astype(int)
sens  = Xte['SEX'].values
mf = MetricFrame(
    metrics={'accuracy': accuracy_score, 'precision': precision_score, 'recall': recall_score},
    y_true=yte.values,
    y_pred=ypred,
    sensitive_features=sens
)
print(mf.by_group.to_string())
print(f'Demographic Parity Diff: {demographic_parity_difference(yte, ypred, sensitive_features=sens):.4f}')
print(f'Equalized Odds Diff:     {equalized_odds_difference(yte, ypred, sensitive_features=sens):.4f}')
print('Values close to 0 = fair model.')
