import pandas as pd
import numpy as np


def engineer_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'default payment next month' in df.columns:
        df.rename(columns={'default payment next month': 'target'}, inplace=True)

    pay_cols  = [c for c in ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'] if c in df.columns]
    bill_cols = [c for c in ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'] if c in df.columns]
    amt_cols  = [c for c in ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'] if c in df.columns]

    if pay_cols:
        df['max_delay']        = df[pay_cols].max(axis=1)
        df['mean_delay']       = df[pay_cols].mean(axis=1)
        df['times_delayed']    = (df[pay_cols] > 0).sum(axis=1)
        df['times_paid_early'] = (df[pay_cols] < 0).sum(axis=1)

    if bill_cols:
        df['total_bill']      = df[bill_cols].sum(axis=1)
        df['mean_bill']       = df[bill_cols].mean(axis=1)
        df['bill_trend']      = df[bill_cols[0]] - df[bill_cols[-1]]
        df['bill_volatility'] = df[bill_cols].std(axis=1)

    if amt_cols:
        df['total_paid'] = df[amt_cols].sum(axis=1)
        df['mean_paid']  = df[amt_cols].mean(axis=1)

    if 'LIMIT_BAL' in df.columns and bill_cols:
        df['utilization'] = (df[bill_cols[0]] / (df['LIMIT_BAL'] + 1)).clip(0, 5)

    if bill_cols and amt_cols:
        df['pay_ratio'] = (df[amt_cols[0]] / (df[bill_cols[0]] + 1)).clip(0, 10)

    if 'LIMIT_BAL' in df.columns and 'AGE' in df.columns:
        df['limit_per_age'] = df['LIMIT_BAL'] / (df['AGE'] + 1)

    df.fillna(0, inplace=True)
    return df
