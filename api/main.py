import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app  = FastAPI(title='Credit Risk Intelligence API', version='1.0.0')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

scaler       = joblib.load(os.path.join(BASE, 'data/feature_store/scaler.pkl'))
credit_model = joblib.load(os.path.join(BASE, 'data/feature_store/best_credit_model.pkl'))

FEATS = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'max_delay', 'mean_delay', 'times_delayed', 'times_paid_early',
    'total_bill', 'mean_bill', 'bill_trend', 'bill_volatility',
    'total_paid', 'mean_paid', 'utilization', 'pay_ratio', 'limit_per_age'
]


class CreditReq(BaseModel):
    LIMIT_BAL: float = 200000
    SEX: int = 1
    EDUCATION: int = 2
    MARRIAGE: int = 1
    AGE: int = 35
    PAY_0: int = 0
    BILL_AMT1: float = 50000
    PAY_AMT1: float = 20000


@app.get('/')
def root():
    return {'message': 'Credit Risk Intelligence API', 'status': 'running', 'version': '1.0.0'}


@app.get('/health')
def health():
    return {'status': 'healthy', 'models': ['credit_scorer']}


@app.post('/predict/credit-score')
def predict(req: CreditReq):
    d     = req.dict()
    bills = [d.get('BILL_AMT1', 0)] + [0] * 5
    pmts  = [d.get('PAY_AMT1', 0)]  + [0] * 5
    pays  = [d.get('PAY_0', 0)]     + [0] * 5

    row = {
        **d,
        'PAY_2': 0, 'PAY_3': 0, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,
        'BILL_AMT2': 0, 'BILL_AMT3': 0, 'BILL_AMT4': 0, 'BILL_AMT5': 0, 'BILL_AMT6': 0,
        'PAY_AMT2':  0, 'PAY_AMT3':  0, 'PAY_AMT4':  0, 'PAY_AMT5':  0, 'PAY_AMT6':  0,
        'max_delay':        max(pays),
        'mean_delay':       float(np.mean(pays)),
        'times_delayed':    sum(p > 0 for p in pays),
        'times_paid_early': sum(p < 0 for p in pays),
        'total_bill':       sum(bills),
        'mean_bill':        float(np.mean(bills)),
        'bill_trend':       bills[0] - bills[-1],
        'bill_volatility':  float(np.std(bills)),
        'total_paid':       sum(pmts),
        'mean_paid':        float(np.mean(pmts)),
        'utilization':      min(bills[0] / (d['LIMIT_BAL'] + 1), 5.0),
        'pay_ratio':        min(pmts[0] / (bills[0] + 1), 10.0),
        'limit_per_age':    d['LIMIT_BAL'] / (d['AGE'] + 1),
    }

    X    = pd.DataFrame([[row.get(f, 0) for f in FEATS]], columns=FEATS)
    prob = float(credit_model.predict_proba(scaler.transform(X))[0, 1])

    return {
        'default_probability': round(prob, 4),
        'risk_level':     'HIGH'    if prob > 0.6 else 'MEDIUM' if prob > 0.3 else 'LOW',
        'recommendation': 'DECLINE' if prob > 0.6 else 'REVIEW' if prob > 0.3 else 'APPROVE',
    }
