import pandas as pd
import numpy as np
import itertools


def preprocess_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean known noise issues in the UCI Credit Default dataset before
    feature engineering.
    """
    df = df.copy()
    if 'default payment next month' in df.columns:
        df.rename(columns={'default payment next month': 'target'}, inplace=True)

    # ── 1. Fix undocumented categorical values ────────────────────────────────
    # EDUCATION: 0, 5, 6 are not in the codebook — collapse to 4 (Other)
    if 'EDUCATION' in df.columns:
        df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE: 0 is undocumented — collapse to 3 (Other)
    if 'MARRIAGE' in df.columns:
        df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})

    # ── 2. Distinguish PAY=-2 (no consumption) from PAY=-1 (paid duly) ───────
    # Many papers treat both as "good" but they signal different risk levels
    pay_cols = [c for c in ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
                if c in df.columns]
    for col in pay_cols:
        df[col + '_no_consumption'] = (df[col] == -2).astype(int)
        df[col + '_paid_duly']      = (df[col] == -1).astype(int)
        # Re-encode: treat -2 and -1 both as 0 (no delay) for the raw column
        df[col] = df[col].clip(lower=0)

    # ── 3. Winsorize bill amounts (1st–99th percentile) ───────────────────────
    bill_cols = [c for c in ['BILL_AMT1','BILL_AMT2','BILL_AMT3',
                              'BILL_AMT4','BILL_AMT5','BILL_AMT6'] if c in df.columns]
    for col in bill_cols:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    # ── 4. Winsorize payment amounts (0–99th percentile) ─────────────────────
    amt_cols = [c for c in ['PAY_AMT1','PAY_AMT2','PAY_AMT3',
                             'PAY_AMT4','PAY_AMT5','PAY_AMT6'] if c in df.columns]
    for col in amt_cols:
        hi = df[col].quantile(0.99)
        df[col] = df[col].clip(0, hi)

    # ── 5. Log-transform payment amounts (right-skewed distributions) ─────────
    for col in amt_cols:
        df[col + '_log'] = np.log1p(df[col])

    # ── 6. Log-transform credit limit ─────────────────────────────────────────
    if 'LIMIT_BAL' in df.columns:
        p99 = df['LIMIT_BAL'].quantile(0.99)
        df['LIMIT_BAL'] = df['LIMIT_BAL'].clip(0, p99)
        df['LIMIT_BAL_log'] = np.log1p(df['LIMIT_BAL'])

    df.fillna(0, inplace=True)
    return df


def engineer_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'default payment next month' in df.columns:
        df.rename(columns={'default payment next month': 'target'}, inplace=True)

    pay_cols  = [c for c in ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'] if c in df.columns]
    bill_cols = [c for c in ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'] if c in df.columns]
    amt_cols  = [c for c in ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'] if c in df.columns]

    # ── Payment status features ────────────────────────────────────────────────
    if pay_cols:
        df['max_delay']           = df[pay_cols].max(axis=1)
        df['mean_delay']          = df[pay_cols].mean(axis=1)
        df['times_delayed']       = (df[pay_cols] > 0).sum(axis=1)
        df['times_paid_early']    = (df[pay_cols] < 0).sum(axis=1)
        df['consecutive_delays']  = df[pay_cols].apply(
            lambda r: max((sum(1 for _ in g)
                           for k, g in itertools.groupby(r > 0) if k), default=0), axis=1)
        df['delay_trend']         = df[pay_cols[0]] - df[pay_cols[-1]]
        df['pay_status_std']      = df[pay_cols].std(axis=1)
        df['ever_severely_late']  = (df[pay_cols] >= 3).any(axis=1).astype(int)
        df['recent_delay_weight'] = (df[pay_cols[0]] * 3 + df[pay_cols[1]] * 2 +
                                     df[pay_cols[2]] * 1).clip(lower=0)

    # ── Bill amount features ───────────────────────────────────────────────────
    if bill_cols:
        df['total_bill']          = df[bill_cols].sum(axis=1)
        df['mean_bill']           = df[bill_cols].mean(axis=1)
        df['bill_trend']          = df[bill_cols[0]] - df[bill_cols[-1]]
        df['bill_volatility']     = df[bill_cols].std(axis=1)
        df['bill_max']            = df[bill_cols].max(axis=1)
        df['bill_increasing']     = (df[bill_cols[0]] > df[bill_cols[-1]]).astype(int)
        df['zero_bill_months']    = (df[bill_cols] == 0).sum(axis=1)

    # ── Payment amount features ────────────────────────────────────────────────
    if amt_cols:
        df['total_paid']          = df[amt_cols].sum(axis=1)
        df['mean_paid']           = df[amt_cols].mean(axis=1)
        df['pay_amt_trend']       = df[amt_cols[0]] - df[amt_cols[-1]]
        df['pay_amt_volatility']  = df[amt_cols].std(axis=1)
        df['zero_pay_months']     = (df[amt_cols] == 0).sum(axis=1)
        df['pay_amt_max']         = df[amt_cols].max(axis=1)

    # ── Ratio / interaction features ───────────────────────────────────────────
    if 'LIMIT_BAL' in df.columns and bill_cols:
        df['utilization']         = (df[bill_cols[0]] / (df['LIMIT_BAL'] + 1)).clip(0, 5)
        df['mean_utilization']    = (df['mean_bill']  / (df['LIMIT_BAL'] + 1)).clip(0, 5) \
                                     if 'mean_bill' in df.columns else 0

    if bill_cols and amt_cols:
        df['pay_ratio']           = (df[amt_cols[0]] / (df[bill_cols[0]] + 1)).clip(0, 10)
        df['total_pay_ratio']     = (df['total_paid'] / (df['total_bill'] + 1)).clip(0, 10) \
                                     if 'total_paid' in df.columns else 0
        df['pay_bill_gap']        = (df[bill_cols[0]] - df[amt_cols[0]]).clip(lower=0)
        df['cumulative_gap']      = sum(
            (df[b] - df[a]).clip(lower=0) for b, a in zip(bill_cols, amt_cols))

    if 'LIMIT_BAL' in df.columns and 'AGE' in df.columns:
        df['limit_per_age']       = df['LIMIT_BAL'] / (df['AGE'] + 1)

    # ── Interaction / non-linear features ─────────────────────────────────────
    if 'max_delay' in df.columns and 'utilization' in df.columns:
        df['delay_x_utilization'] = df['max_delay'] * df['utilization']

    if 'times_delayed' in df.columns and 'total_pay_ratio' in df.columns:
        df['risk_score']          = df['times_delayed'] * (1 - df['total_pay_ratio'].clip(0, 1))

    if 'EDUCATION' in df.columns and 'LIMIT_BAL' in df.columns:
        df['edu_limit']           = df['EDUCATION'] * np.log1p(df['LIMIT_BAL'])

    if 'zero_pay_months' in df.columns and 'times_delayed' in df.columns:
        df['nonpayment_score']    = df['zero_pay_months'] + df['times_delayed'] * 2

    df.fillna(0, inplace=True)
    return df
