import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
import joblib
from src.features.engineer import engineer_credit_features


@pytest.fixture
def sample_row():
    return pd.DataFrame([{
        'LIMIT_BAL': 200000, 'SEX': 1, 'EDUCATION': 2, 'MARRIAGE': 1, 'AGE': 35,
        'PAY_0': 0, 'PAY_2': 0, 'PAY_3': 0, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,
        'BILL_AMT1': 50000, 'BILL_AMT2': 45000, 'BILL_AMT3': 40000,
        'BILL_AMT4': 35000, 'BILL_AMT5': 30000, 'BILL_AMT6': 25000,
        'PAY_AMT1': 20000, 'PAY_AMT2': 18000, 'PAY_AMT3': 15000,
        'PAY_AMT4': 12000, 'PAY_AMT5': 10000, 'PAY_AMT6': 8000,
        'default payment next month': 0
    }])


def test_engineering_adds_columns(sample_row):
    assert engineer_credit_features(sample_row).shape[1] > sample_row.shape[1]


def test_no_nulls_after_engineering(sample_row):
    assert engineer_credit_features(sample_row).isnull().sum().sum() == 0


def test_utilization_is_clipped(sample_row):
    result = engineer_credit_features(sample_row)
    assert result['utilization'].iloc[0] <= 5.0


def test_model_files_exist():
    assert os.path.exists('data/feature_store/scaler.pkl'),            'Scaler missing'
    assert os.path.exists('data/feature_store/best_credit_model.pkl'), 'Credit model missing'
    assert os.path.exists('data/feature_store/fraud_model.pkl'),       'Fraud model missing'


def test_credit_model_probabilities_valid():
    sc    = joblib.load('data/feature_store/scaler.pkl')
    model = joblib.load('data/feature_store/best_credit_model.pkl')
    df    = pd.read_csv('data/processed/credit_features.csv')
    X     = df.drop(columns=['ID', 'target'], errors='ignore').head(10)
    probs = model.predict_proba(sc.transform(X))[:, 1]
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_fraud_model_probabilities_valid():
    fm    = joblib.load('data/feature_store/fraud_model.pkl')
    dummy = pd.DataFrame(np.zeros((5, 11)), columns=[
        'type_enc', 'amount_log', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'balance_diff_orig',
        'balance_diff_dest', 'zero_orig_after', 'zero_dest_before', 'hour'
    ])
    probs = fm.predict_proba(dummy)[:, 1]
    assert all(0.0 <= p <= 1.0 for p in probs)
