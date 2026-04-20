import pandas as pd
import numpy as np
import os


def generate_drift_report(ref, cur, target_col, out='data/processed/drift_report.html'):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently.pipeline.column_mapping import ColumnMapping
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur,
                   column_mapping=ColumnMapping(target=target_col))
        report.save_html(out)
    except (ModuleNotFoundError, ImportError):
        # evidently >= 0.6 uses a different API
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref, current_data=cur)
            report.save_html(out)
        except Exception as e:
            _fallback_drift_report(ref, cur, target_col, out, str(e))
            return
    print(f'Drift report saved: {out}')


def _fallback_drift_report(ref, cur, target_col, out, reason):
    """Simple HTML drift summary when evidently API is unavailable."""
    num_cols = ref.select_dtypes(include=np.number).columns.tolist()
    rows = []
    for col in num_cols:
        ref_mean = ref[col].mean()
        cur_mean = cur[col].mean()
        shift = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-9)
        drifted = 'YES' if shift > 0.1 else 'no'
        rows.append(f'<tr><td>{col}</td><td>{ref_mean:.4f}</td>'
                    f'<td>{cur_mean:.4f}</td><td>{shift:.2%}</td>'
                    f'<td style="color:{"red" if drifted=="YES" else "green"}">{drifted}</td></tr>')
    html = f"""<html><body>
<h2>Data Drift Report (fallback — evidently API mismatch: {reason})</h2>
<table border="1" cellpadding="4">
<tr><th>Feature</th><th>Ref Mean</th><th>Cur Mean</th><th>Shift</th><th>Drifted?</th></tr>
{''.join(rows)}
</table></body></html>"""
    with open(out, 'w') as f:
        f.write(html)
    print(f'Drift report saved (fallback): {out}')


if __name__ == '__main__':
    df  = pd.read_csv('data/processed/credit_features.csv')
    mid = len(df) // 2
    ref = df.iloc[:mid].copy()
    cur = df.iloc[mid:].copy()
    nc  = [c for c in cur.select_dtypes(include=np.number).columns if c != 'target']
    cur[nc[:5]] = cur[nc[:5]] * 1.12
    generate_drift_report(ref, cur, 'target')
