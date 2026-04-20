import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

print('=== Fraud Detection ===')

df = pd.read_csv('data/raw/fraud_transactions.csv')
print(f'Data: {df.shape}  fraud_rate={df["isFraud"].mean():.4f}')

le = LabelEncoder()
df['type_enc']          = le.fit_transform(df['type'])
df['balance_diff_orig'] = (df['newbalanceOrig'] - df['oldbalanceOrg'] + df['amount']).abs()
df['balance_diff_dest'] = (df['newbalanceDest'] - df['oldbalanceDest'] - df['amount']).abs()
df['zero_orig_after']   = (df['newbalanceOrig'] == 0).astype(int)
df['zero_dest_before']  = (df['oldbalanceDest'] == 0).astype(int)
df['amount_log']        = np.log1p(df['amount'])
df['hour']              = df['step'] % 24

FEAT = ['type_enc', 'amount_log', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'balance_diff_orig',
        'balance_diff_dest', 'zero_orig_after', 'zero_dest_before', 'hour']

X, y = df[FEAT], df['isFraud']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print('Training Isolation Forest...')
iso = IsolationForest(n_estimators=200, contamination=0.013, random_state=42, n_jobs=-1)
iso.fit(Xtr)
print(f'  AUC: {roc_auc_score(yte, -iso.score_samples(Xte)):.4f}')

print('Training Random Forest (supervised)...')
Xb, yb = SMOTE(random_state=42, k_neighbors=3).fit_resample(Xtr, ytr)
rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                              class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(Xb, yb)
print(f'  AUC: {roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]):.4f}')

print('Training Autoencoder...')
sc     = StandardScaler()
Xn_sc  = sc.fit_transform(Xtr[ytr == 0])
Xte_sc = sc.transform(Xte)
Xn_t   = torch.FloatTensor(Xn_sc)
Xte_t  = torch.FloatTensor(Xte_sc)


class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,32), nn.ReLU(),
                                  nn.Linear(32,16), nn.ReLU(), nn.Linear(16,8))
        self.dec = nn.Sequential(nn.Linear(8,16), nn.ReLU(),
                                  nn.Linear(16,32), nn.ReLU(), nn.Linear(32,d))
    def forward(self, x): return self.dec(self.enc(x))


ae  = AE(Xn_sc.shape[1])
opt = torch.optim.Adam(ae.parameters(), lr=0.001)
for ep in range(60):
    ae.train()
    opt.zero_grad()
    loss = nn.MSELoss()(ae(Xn_t), Xn_t)
    loss.backward()
    opt.step()
    if ep % 20 == 0:
        print(f'  epoch {ep}  loss={loss.item():.5f}')

ae.eval()
with torch.no_grad():
    scores = ((Xte_t - ae(Xte_t)) ** 2).mean(dim=1).numpy()
print(f'  Autoencoder AUC: {roc_auc_score(yte, scores):.4f}')

joblib.dump(rf, 'data/feature_store/fraud_model.pkl')
joblib.dump(sc, 'data/feature_store/fraud_scaler.pkl')
torch.save(ae.state_dict(), 'data/feature_store/autoencoder.pt')
print('Fraud models saved.')
