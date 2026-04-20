import pandas as pd
import numpy as np
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping


def generate_drift_report(ref, cur, target_col, out='data/processed/drift_report.html'):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur,
               column_mapping=ColumnMapping(target=target_col))
    report.save_html(out)
    print(f'Drift report saved: {out}')
    return report


if __name__ == '__main__':
    df  = pd.read_csv('data/processed/credit_features.csv')
    mid = len(df) // 2
    ref = df.iloc[:mid].copy()
    cur = df.iloc[mid:].copy()
    nc  = [c for c in cur.select_dtypes(include=np.number).columns if c != 'target']
    cur[nc[:5]] = cur[nc[:5]] * 1.12
    generate_drift_report(ref, cur, 'target')
