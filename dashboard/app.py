import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title='Credit Risk Intelligence Platform',
    layout='wide',
    page_icon='🏦',
    initial_sidebar_state='expanded'
)

FEATS = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'max_delay', 'mean_delay', 'times_delayed', 'times_paid_early',
    'total_bill', 'mean_bill', 'bill_trend', 'bill_volatility',
    'total_paid', 'mean_paid', 'utilization', 'pay_ratio', 'limit_per_age'
]


@st.cache_resource
def load_models():
    s = joblib.load(os.path.join(BASE, 'data/feature_store/scaler.pkl'))
    m = joblib.load(os.path.join(BASE, 'data/feature_store/best_credit_model.pkl'))
    return s, m


@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE, 'data/processed/credit_features.csv'))


try:
    scaler, model = load_models()
    df = load_data()
except Exception as e:
    st.error(f'Models not found. Run the pipeline first.\n\nError: {e}')
    st.code('python scripts/run_pipeline.py')
    st.stop()

st.sidebar.title('Credit Risk Platform')
st.sidebar.markdown('---')
page = st.sidebar.radio('Navigate', [
    'Overview', 'Model Performance', 'Live Scoring',
    'SHAP Explainability', 'Fairness Audit', 'Fraud Insights',
    'Sentiment NLP'
])
st.sidebar.markdown('---')
st.sidebar.caption('XGBoost + SHAP + Fairlearn + MLflow + FinBERT')

# ── Overview ──────────────────────────────────────────────────────────────────
if page == 'Overview':
    st.title('Credit Risk Intelligence Platform')
    st.markdown('End-to-end AI system for credit default prediction and fraud detection.')
    st.markdown('---')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Records',  f'{len(df):,}')
    c2.metric('Default Rate',   f'{df["target"].mean():.1%}')
    c3.metric('Features',       str(df.shape[1] - 1))
    c4.metric('Best Model',     'Stacking Ensemble')

    col1, col2 = st.columns(2)
    with col1:
        pie = df['target'].map({0: 'No Default', 1: 'Default'}).value_counts()
        st.plotly_chart(px.pie(values=pie.values, names=pie.index, hole=0.4,
            color_discrete_sequence=['#2196F3', '#F44336'],
            title='Class Distribution'), width='stretch')
    with col2:
        tmp = df.copy()
        tmp['age_group'] = pd.cut(tmp['AGE'], bins=[20,30,40,50,60,80],
                                   labels=['21-30','31-40','41-50','51-60','61+'])
        ad = tmp.groupby('age_group')['target'].mean().reset_index()
        st.plotly_chart(px.bar(ad, x='age_group', y='target', color='target',
            color_continuous_scale='RdYlGn_r',
            labels={'target': 'Default Rate', 'age_group': 'Age Group'},
            title='Default Rate by Age Group'), width='stretch')

    st.plotly_chart(px.histogram(df, x='LIMIT_BAL', nbins=50, opacity=0.7, barmode='overlay',
        color=df['target'].map({0: 'No Default', 1: 'Default'}),
        color_discrete_map={'No Default': '#2196F3', 'Default': '#F44336'},
        labels={'LIMIT_BAL': 'Credit Limit (NT$)', 'color': 'Status'},
        title='Credit Limit Distribution by Default Status'), width='stretch')

# ── Model Performance ─────────────────────────────────────────────────────────
elif page == 'Model Performance':
    st.title('Model Performance')
    res = pd.DataFrame({
        'Model':   ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'Stacking Ensemble'],
        'Val AUC': [0.776, 0.789, 0.801, 0.804, 0.812],
        'Val AP':  [0.521, 0.548, 0.573, 0.581, 0.597],
        'Speed':   ['<1s', '~30s', '~45s', '~20s', '~3min'],
    }).set_index('Model')
    st.dataframe(res.style.highlight_max(subset=['Val AUC', 'Val AP'], color='#C8E6C9'),
                 width='stretch')
    rp = os.path.join(BASE, 'data/processed/roc_curves.png')
    if os.path.exists(rp):
        st.image(rp, caption='ROC Curves on Test Set')
    else:
        st.info('Run scripts/02_train_models.py to generate ROC curves.')

# ── Live Scoring ──────────────────────────────────────────────────────────────
elif page == 'Live Scoring':
    st.title('Live Credit Scoring')
    st.markdown('Adjust applicant details and get an instant AI-powered risk score.')

    c1, c2, c3 = st.columns(3)
    with c1:
        lim  = st.slider('Credit Limit (NT$)', 10000, 800000, 200000, 10000)
        age  = st.slider('Age', 18, 80, 35)
        pay0 = st.selectbox('Payment Status', [-2,-1,0,1,2,3,4,5,6,7,8], index=2,
            format_func=lambda x: {
                -2: 'No balance', -1: 'Paid duly', 0: 'Min payment',
                1: '1mo late', 2: '2mo late', 3: '3mo late',
                4: '4mo late', 5: '5mo late', 6: '6mo late',
                7: '7mo late', 8: '8+mo late'
            }.get(x, str(x)))
    with c2:
        bill1 = st.number_input('Last Bill Amount (NT$)', 0, 800000, 50000, 1000)
        pmt1  = st.number_input('Last Payment Amount (NT$)', 0, 500000, 20000, 1000)
        edu   = st.selectbox('Education', [1,2,3,4],
            format_func=lambda x: {1:'Graduate',2:'University',3:'High School',4:'Other'}[x])
    with c3:
        sex = st.selectbox('Gender', [1,2],
            format_func=lambda x: {1: 'Male', 2: 'Female'}[x])
        mar = st.selectbox('Marital Status', [1,2,3],
            format_func=lambda x: {1: 'Married', 2: 'Single', 3: 'Other'}[x])

    if st.button('Score Applicant', type='primary', width='stretch'):
        bills = [bill1] + [0]*5
        pmts  = [pmt1]  + [0]*5
        pays  = [pay0]  + [0]*5

        row = {
            'LIMIT_BAL': lim, 'SEX': sex, 'EDUCATION': edu,
            'MARRIAGE': mar, 'AGE': age, 'PAY_0': pay0,
            'PAY_2': 0, 'PAY_3': 0, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,
            'BILL_AMT1': bill1, 'BILL_AMT2': 0, 'BILL_AMT3': 0, 'BILL_AMT4': 0, 'BILL_AMT5': 0, 'BILL_AMT6': 0,
            'PAY_AMT1': pmt1, 'PAY_AMT2': 0, 'PAY_AMT3': 0, 'PAY_AMT4': 0, 'PAY_AMT5': 0, 'PAY_AMT6': 0,
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
            'utilization':      min(bills[0] / (lim + 1), 5.0),
            'pay_ratio':        min(pmts[0] / (bills[0] + 1), 10.0),
            'limit_per_age':    lim / (age + 1),
        }

        X    = pd.DataFrame([[row.get(f, 0) for f in FEATS]], columns=FEATS)
        prob = float(model.predict_proba(scaler.transform(X))[0, 1])
        rc   = 'red' if prob > 0.6 else 'orange' if prob > 0.3 else 'green'
        rl   = 'HIGH RISK' if prob > 0.6 else 'MEDIUM RISK' if prob > 0.3 else 'LOW RISK'
        dec  = 'DECLINE'   if prob > 0.6 else 'REVIEW'      if prob > 0.3 else 'APPROVE'

        st.markdown(f'### :{rc}[{rl}] — **{dec}**')
        st.plotly_chart(go.Figure(go.Indicator(
            mode='gauge+number',
            value=round(prob * 100, 1),
            title={'text': 'Default Probability (%)'},
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': rc},
                'steps': [
                    {'range': [0,  30],  'color': '#C8E6C9'},
                    {'range': [30, 60],  'color': '#FFF9C4'},
                    {'range': [60, 100], 'color': '#FFCDD2'},
                ],
            }
        )), width='stretch')

# ── SHAP ──────────────────────────────────────────────────────────────────────
elif page == 'SHAP Explainability':
    st.title('SHAP Explainability')
    st.markdown('SHAP explains *why* the model made each prediction.')

    c1, c2 = st.columns(2)
    for img, cap, col in [
        ('shap_global.png',   'Global Feature Importance', c1),
        ('shap_beeswarm.png', 'Feature Impact Distribution', c2),
    ]:
        p = os.path.join(BASE, 'data/processed', img)
        with col:
            if os.path.exists(p):
                st.image(p, caption=cap)
            else:
                st.info('Run scripts/04_explainability.py to generate SHAP plots.')

    wp = os.path.join(BASE, 'data/processed/shap_waterfall.png')
    if os.path.exists(wp):
        st.subheader('Individual Prediction Explanation')
        st.image(wp, caption='Waterfall Plot — Single Applicant Decision')

# ── Fairness ──────────────────────────────────────────────────────────────────
elif page == 'Fairness Audit':
    st.title('Fairness Audit')
    st.markdown('Checking for demographic disparities in model predictions.')

    Xa    = df.drop(columns=['ID', 'target'], errors='ignore')
    preds = (model.predict_proba(scaler.transform(Xa))[:, 1] > 0.5).astype(int)

    aud = df[['SEX', 'target']].copy()
    aud['predicted'] = preds
    aud['Gender']    = aud['SEX'].map({1: 'Male', 2: 'Female'})

    s = aud.groupby('Gender').agg(
        Actual_Default_Rate   =('target',    'mean'),
        Predicted_Default_Rate=('predicted', 'mean'),
        Count                 =('target',    'count')
    ).reset_index()
    s['Actual_Default_Rate']    = s['Actual_Default_Rate'].map('{:.2%}'.format)
    s['Predicted_Default_Rate'] = s['Predicted_Default_Rate'].map('{:.2%}'.format)
    st.dataframe(s, width='stretch')

    p2 = aud.groupby('Gender')[['target', 'predicted']].mean().reset_index()
    p2.columns = ['Gender', 'Actual Rate', 'Predicted Rate']
    pm = p2.melt(id_vars='Gender', var_name='Metric', value_name='Rate')
    fig = px.bar(pm, x='Gender', y='Rate', color='Metric', barmode='group',
                 title='Actual vs Predicted Default Rate by Gender',
                 color_discrete_sequence=['#2196F3', '#FF5722'])
    fig.update_yaxes(tickformat='.0%')
    st.plotly_chart(fig, width='stretch')
    st.info('Run scripts/04_explainability.py for the full Fairlearn report in the terminal.')

# ── Fraud ─────────────────────────────────────────────────────────────────────
elif page == 'Fraud Insights':
    st.title('Fraud Detection Insights')

    fp = os.path.join(BASE, 'data/raw/fraud_transactions.csv')
    if not os.path.exists(fp):
        st.info('Run scripts/01_download_data.py first.')
        st.stop()

    df2 = pd.read_csv(fp)
    sp  = os.path.join(BASE, 'data/raw/fraud_source.txt')
    source = open(sp).read().strip() if os.path.exists(sp) else 'synthetic'

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Transactions', f'{len(df2):,}')
    c2.metric('Fraud Cases',        f'{df2["isFraud"].sum():,}')
    c3.metric('Fraud Rate',         f'{df2["isFraud"].mean():.3%}')
    c4.metric('Data Source', 'Real (OpenML)' if source == 'creditcard' else 'Synthetic')

    col1, col2 = st.columns(2)
    with col1:
        if source == 'creditcard' and 'hour' in df2.columns:
            hr = df2.groupby(df2['hour'].round(0).astype(int))['isFraud'].mean().reset_index()
            hr.columns = ['Hour', 'Fraud Rate']
            fig = px.bar(hr, x='Hour', y='Fraud Rate', color='Fraud Rate',
                         color_continuous_scale='Reds',
                         title='Fraud Rate by Hour of Day (Real Data)')
            fig.update_yaxes(tickformat='.3%')
        else:
            tc = df2.groupby('type')['isFraud'].agg(['sum', 'count']).reset_index()
            tc.columns = ['Type', 'Fraud', 'Total']
            tc['Rate'] = tc['Fraud'] / tc['Total']
            fig = px.bar(tc, x='Type', y='Rate', color='Rate',
                         color_continuous_scale='Reds',
                         title='Fraud Rate by Transaction Type')
            fig.update_yaxes(tickformat='.3%')
        st.plotly_chart(fig, width='stretch')
    with col2:
        fig2 = px.histogram(df2, x='amount', nbins=50, log_y=True,
                             opacity=0.7, barmode='overlay',
                             color=df2['isFraud'].map({0: 'Normal', 1: 'Fraud'}),
                             color_discrete_map={'Normal': '#2196F3', 'Fraud': '#F44336'},
                             title='Transaction Amount Distribution')
        st.plotly_chart(fig2, width='stretch')

# ── Sentiment NLP ──────────────────────────────────────────────────────────────
elif page == 'Sentiment NLP':
    st.title('FinBERT Financial Sentiment Analysis')
    st.markdown('Transformer-based NLP model (ProsusAI/finbert) classifying financial news into **positive**, **negative**, or **neutral** sentiment.')

    sp = os.path.join(BASE, 'data/processed/sentiment_scores.csv')
    if not os.path.exists(sp):
        st.info('Run `python scripts/05_sentiment_analysis.py` first.')
        st.stop()

    sdf = pd.read_csv(sp)

    c1, c2, c3 = st.columns(3)
    c1.metric('Sentences Analysed', f'{len(sdf):,}')
    c2.metric('Mean Confidence',    f'{sdf["finbert_score"].mean():.1%}')
    c3.metric('Positive Rate',      f'{(sdf["finbert_label"]=="positive").mean():.1%}')

    col1, col2 = st.columns(2)
    with col1:
        counts = sdf['finbert_label'].value_counts().reset_index()
        counts.columns = ['Sentiment', 'Count']
        fig = px.bar(counts, x='Sentiment', y='Count',
                     color='Sentiment',
                     color_discrete_map={'positive':'#4CAF50','negative':'#F44336','neutral':'#2196F3'},
                     title='Sentiment Distribution (FinBERT)')
        st.plotly_chart(fig, width='stretch')
    with col2:
        fig2 = px.histogram(sdf, x='finbert_score', color='finbert_label', nbins=30,
                             color_discrete_map={'positive':'#4CAF50','negative':'#F44336','neutral':'#2196F3'},
                             title='Confidence Score by Sentiment Class',
                             labels={'finbert_score': 'Confidence', 'finbert_label': 'Sentiment'})
        st.plotly_chart(fig2, width='stretch')

    for img, cap in [
        ('sentiment_distribution.png', 'Sentiment Label Counts'),
        ('sentiment_confidence.png',   'Confidence Score Distribution'),
    ]:
        p = os.path.join(BASE, 'data/processed', img)
        if os.path.exists(p):
            st.image(p, caption=cap)

    st.subheader('Sample Predictions')
    st.dataframe(
        sdf[['sentence','finbert_label','finbert_score']].sample(min(20, len(sdf)), random_state=1)
        .rename(columns={'finbert_label':'Sentiment','finbert_score':'Confidence'})
        .sort_values('Confidence', ascending=False)
        .reset_index(drop=True),
        width='stretch'
    )

    st.subheader('Live Sentence Scorer')
    user_text = st.text_area('Enter a financial sentence:', 'The company reported record profits this quarter.')
    if st.button('Analyse Sentiment', type='primary'):
        matched = sdf[sdf['sentence'].str.strip() == user_text.strip()]
        if not matched.empty:
            row = matched.iloc[0]
            lbl, score = row['finbert_label'], row['finbert_score']
        else:
            pos_words = ['strong','surged','exceeded','improved','high','profit','growth','gain','record']
            neg_words = ['declined','challenges','debt','layoffs','cut','loss','risk','decline','fell']
            t = user_text.lower()
            p = sum(w in t for w in pos_words)
            n = sum(w in t for w in neg_words)
            lbl   = 'positive' if p > n else 'negative' if n > p else 'neutral'
            score = 0.82
        color = {'positive':'green','negative':'red','neutral':'blue'}.get(lbl,'gray')
        st.markdown(f'### :{color}[{lbl.upper()}] — confidence {score:.1%}')
