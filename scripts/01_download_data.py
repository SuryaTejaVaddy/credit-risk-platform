import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs('data/raw', exist_ok=True)

print('Generating credit default dataset...')
np.random.seed(42)
n = 30000
df = pd.DataFrame({
    'ID': range(1, n+1),
    'LIMIT_BAL': np.random.choice([10000,20000,30000,50000,80000,100000,150000,200000,300000,500000], n),
    'SEX': np.random.choice([1,2], n),
    'EDUCATION': np.random.choice([1,2,3,4], n, p=[0.35,0.46,0.16,0.03]),
    'MARRIAGE': np.random.choice([1,2,3], n, p=[0.45,0.46,0.09]),
    'AGE': np.random.randint(21, 75, n),
    'PAY_0': np.random.choice([-2,-1,0,1,2,3], n, p=[0.12,0.28,0.38,0.10,0.08,0.04]),
    'PAY_2': np.random.choice([-2,-1,0,1,2,3], n, p=[0.13,0.30,0.38,0.09,0.08,0.02]),
    'PAY_3': np.random.choice([-2,-1,0,1,2,3], n, p=[0.14,0.31,0.37,0.08,0.08,0.02]),
    'PAY_4': np.random.choice([-2,-1,0,1,2,3], n, p=[0.15,0.32,0.36,0.07,0.08,0.02]),
    'PAY_5': np.random.choice([-2,-1,0,1,2,3], n, p=[0.15,0.33,0.36,0.07,0.07,0.02]),
    'PAY_6': np.random.choice([-2,-1,0,1,2,3], n, p=[0.15,0.34,0.35,0.07,0.07,0.02]),
    'BILL_AMT1': np.random.exponential(50000, n).clip(0, 800000).round(0),
    'BILL_AMT2': np.random.exponential(48000, n).clip(0, 800000).round(0),
    'BILL_AMT3': np.random.exponential(46000, n).clip(0, 800000).round(0),
    'BILL_AMT4': np.random.exponential(44000, n).clip(0, 800000).round(0),
    'BILL_AMT5': np.random.exponential(42000, n).clip(0, 800000).round(0),
    'BILL_AMT6': np.random.exponential(40000, n).clip(0, 800000).round(0),
    'PAY_AMT1': np.random.exponential(20000, n).clip(0, 400000).round(0),
    'PAY_AMT2': np.random.exponential(19000, n).clip(0, 400000).round(0),
    'PAY_AMT3': np.random.exponential(18000, n).clip(0, 400000).round(0),
    'PAY_AMT4': np.random.exponential(17000, n).clip(0, 400000).round(0),
    'PAY_AMT5': np.random.exponential(16000, n).clip(0, 400000).round(0),
    'PAY_AMT6': np.random.exponential(15000, n).clip(0, 400000).round(0),
    'default payment next month': np.random.choice([0,1], n, p=[0.778, 0.222])
})
df.to_csv('data/raw/credit_default.csv', index=False)
print(f'  Saved credit_default.csv  shape={df.shape}')

print('Generating fraud dataset...')
np.random.seed(42)
n2 = 100000
fl = np.random.choice([0,1], n2, p=[0.9987, 0.0013])
df2 = pd.DataFrame({
    'step': np.random.randint(1, 744, n2),
    'type': np.random.choice(['PAYMENT','TRANSFER','CASH_OUT','DEBIT','CASH_IN'], n2,
                              p=[0.34,0.25,0.22,0.13,0.06]),
    'amount': np.where(fl==1,
                       np.random.exponential(300000, n2),
                       np.random.exponential(80000, n2)).clip(1).round(2),
    'oldbalanceOrg':  np.random.exponential(250000, n2).clip(0).round(2),
    'newbalanceOrig': np.random.exponential(200000, n2).clip(0).round(2),
    'oldbalanceDest': np.random.exponential(200000, n2).clip(0).round(2),
    'newbalanceDest': np.random.exponential(150000, n2).clip(0).round(2),
    'isFraud': fl
})
df2.to_csv('data/raw/fraud_transactions.csv', index=False)
print(f'  Saved fraud_transactions.csv  shape={df2.shape}  fraud_rate={fl.mean():.4f}')

print('Generating sentiment dataset...')
np.random.seed(42)
sentences = [
    'The company reported strong earnings this quarter.',
    'Revenue declined sharply due to market conditions.',
    'The stock remained flat amid uncertainty.',
    'Profits surged following the acquisition.',
    'The firm faces significant debt challenges.',
    'Quarterly results exceeded analyst expectations.',
    'The company announced major layoffs.',
    'Stock price hit an all-time high today.',
    'Operating margins improved significantly.',
    'The board approved a dividend cut.',
] * 300
labels = np.random.choice([0,1,2], 3000, p=[0.24, 0.09, 0.67])
pd.DataFrame({'sentence': sentences[:3000], 'label': labels}).to_csv(
    'data/raw/financial_sentiment.csv', index=False)
print('  Saved financial_sentiment.csv')
print('\nAll datasets ready.')
