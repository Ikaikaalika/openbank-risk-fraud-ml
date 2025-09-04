import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

st.set_page_config(page_title='Varo Risk Dashboard', layout='wide')
st.title('Varo Risk & Fraud Dashboard')

tab_overview, tab_credit, tab_fraud, tab_monitor = st.tabs([
    'Overview', 'Credit Risk', 'Fraud', 'Monitoring'
])

MODELS_DIR = Path('models')

with tab_overview:
    st.markdown('- Use the tabs to explore model health and monitoring reports.')
    st.markdown('- Files: features under `data/features/`, models under `models/`, reports under `reports/`.')

with tab_credit:
    st.subheader('Credit Risk — Predictions & ROC')
    feat_path = Path('data/features/credit/risk_features.parquet')
    model_path = MODELS_DIR / 'credit_risk_xgb.joblib'
    if feat_path.exists() and model_path.exists():
        df = pd.read_parquet(feat_path)
        model = joblib.load(model_path)
        X = df[["loan_amnt", "int_rate_pct", "dti_clipped", "term"]]
        y = df["defaulted"].astype(int)
        proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
        ax.plot([0,1],[0,1],'k--')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend(loc='lower right')
        st.pyplot(fig)
        st.write(df.head(10))
    else:
        st.info('Train the credit model and generate features to view this tab.')

with tab_fraud:
    st.subheader('Fraud — PR Curve')
    feat_path = Path('data/features/fraud/fraud_features.parquet')
    model_path = MODELS_DIR / 'fraud_xgb.joblib'
    if feat_path.exists() and model_path.exists():
        df = pd.read_parquet(feat_path)
        model = joblib.load(model_path)
        X = df[["time", "amount", "amt_z"]]
        y = df["is_fraud"].astype(int)
        proba = model.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y, proba)
        ap = average_precision_score(y, proba)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f'AP = {ap:.3f}')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.legend(loc='lower left')
        st.pyplot(fig)
        st.write(df.head(10))
    else:
        st.info('Train the fraud model and generate features to view this tab.')

with tab_monitor:
    st.subheader('Monitoring Reports')
    reports_dir = Path('reports/monitoring')
    if reports_dir.exists():
        html_files = list(reports_dir.rglob('*.html'))
        if html_files:
            sel = st.selectbox('Select report', [str(p) for p in html_files])
            st.components.v1.html(Path(sel).read_text(errors='ignore'), height=600, scrolling=True)
        else:
            st.info('No HTML reports found. Run the monitoring job.')
    else:
        st.info('Reports directory not found.')
