import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import subprocess
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import shap
from mlflow.tracking import MlflowClient
import mlflow

st.set_page_config(page_title='Varo Risk Dashboard', layout='wide')
st.title('Varo Risk & Fraud Dashboard')

tab_overview, tab_credit, tab_fraud, tab_monitor = st.tabs([
    'Overview', 'Credit Risk', 'Fraud', 'Monitoring'
])

MODELS_DIR = Path('models')

with tab_overview:
    st.markdown('- Use the tabs to explore model health and monitoring reports.')
    st.markdown('- Files: features under `data/features/`, models under `models/`, reports under `reports/`.')
    st.divider()
    st.subheader('Ops Agent')
    st.caption('Wire up an agent to orchestrate your pipelines. Requires `make agent-serve` running on :8090.')
    import requests as _rq
    col1, col2, col3 = st.columns([3,1,1])
    with col1:
        goal = st.text_area('Goal', placeholder='e.g., Download lendingclub, spark etl, build features, train credit with calibration, evaluate reports')
    with col2:
        dry = st.checkbox('Dry run', value=True)
        use_llm = st.checkbox('Use LLM (Ollama)', value=True)
    with col3:
        model = st.text_input('Model', value='llama3.1')
        if st.button('Execute Plan', type='primary'):
            try:
                resp = _rq.post('http://localhost:8090/agent/execute', json={"goal": goal, "dry_run": dry, "use_llm": use_llm, "model": model}, timeout=60)
                if resp.status_code != 200:
                    st.error(f"Agent error: {resp.status_code} {resp.text}")
                else:
                    data = resp.json()
                    st.write('Plan Execution:')
                    st.json(data)
            except Exception as e:
                st.error(f"Failed to contact agent server: {e}")
    # MLflow recent runs for credit and fraud
    try:
        mlflow.set_tracking_uri(Path('mlruns').absolute().as_uri())
        client = MlflowClient()
        for model_tag in ['credit_risk', 'fraud']:
            st.subheader(f"Latest MLflow runs — {model_tag}")
            runs = client.search_runs(
                experiment_ids=[client.get_experiment_by_name('Default').experiment_id],
                filter_string=f"tags.model_name = '{model_tag}'",
                order_by=["attribute.start_time DESC"],
                max_results=5,
            )
            if not runs:
                st.info('No runs logged yet.')
            else:
                rows = []
                for r in runs:
                    rows.append({
                        'run_id': r.info.run_id,
                        'start_time': r.info.start_time,
                        **r.data.metrics,
                    })
                st.dataframe(pd.DataFrame(rows))
    except Exception as e:
        st.caption(f"MLflow info unavailable: {e}")

with tab_credit:
    st.subheader('Credit Risk — Predictions & ROC')
    eval_dir = Path('reports/evaluation/credit')
    metrics_path = eval_dir / 'metrics.json'
    roc_img = eval_dir / 'roc.png'
    rel_img = eval_dir / 'reliability.png'
    if metrics_path.exists() or roc_img.exists() or rel_img.exists():
        cols = st.columns(2)
        with cols[0]:
            if roc_img.exists():
                st.image(str(roc_img), caption='ROC Curve')
        with cols[1]:
            if rel_img.exists():
                st.image(str(rel_img), caption='Reliability (Calibration)')
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
                st.markdown('Metrics')
                st.json(metrics)
            except Exception:
                st.info('Metrics file present but unreadable.')
        st.divider()
        st.caption('Below is a live preview computed from current features and model (if available).')
    # Controls
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button('Run Evaluation (Credit)'):
            res = subprocess.run(['python','-m','src.jobs.evaluate','--domain','credit'], capture_output=True, text=True)
            st.info(res.stdout or res.stderr)
    with col_btn2:
        if st.button('Calibrated Train (Credit)'):
            res = subprocess.run(['python','-m','src.jobs.train','--domain','credit','--calibrate','true'], capture_output=True, text=True)
            st.info(res.stdout or res.stderr)

    feat_path = Path('data/features/credit/risk_features.parquet')
    # Prefer calibrated model if present
    model_path = MODELS_DIR / 'credit_risk_xgb_isotonic.joblib'
    if not model_path.exists():
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
        # Lift curve
        order = np.argsort(-proba)
        y_sorted = y.iloc[order].to_numpy()
        cum_pos = np.cumsum(y_sorted)
        frac = np.arange(1, len(y_sorted)+1) / len(y_sorted)
        baseline = y.mean() * np.arange(1, len(y_sorted)+1)
        lift = cum_pos / np.maximum(baseline, 1e-6)
        fig2, ax2 = plt.subplots()
        ax2.plot(frac, lift)
        ax2.set_xlabel('Population fraction')
        ax2.set_ylabel('Cumulative lift')
        st.pyplot(fig2)
        # SHAP summary (sampled)
        try:
            def unwrap_estimator(m):
                # Try to get underlying XGB from calibrated wrapper
                base = getattr(m, 'base_estimator', None)
                if base is not None:
                    return base
                ccs = getattr(m, 'calibrated_classifiers_', None)
                if ccs:
                    return ccs[0].estimator
                return m
            est = unwrap_estimator(model)
            explainer = shap.TreeExplainer(est)
            sample_idx = np.random.RandomState(0).choice(len(X), size=min(1000, len(X)), replace=False)
            shap_vals = explainer.shap_values(X.iloc[sample_idx])
            fig3 = plt.figure()
            shap.summary_plot(shap_vals, X.iloc[sample_idx], show=False)
            st.pyplot(fig3)
        except Exception as e:
            st.caption(f'SHAP not available: {e}')
        st.write(df.head(10))
    else:
        st.info('Train the credit model and generate features to view this tab.')

with tab_fraud:
    st.subheader('Fraud — PR Curve')
    eval_dir = Path('reports/evaluation/fraud')
    metrics_path = eval_dir / 'metrics.json'
    pr_img = eval_dir / 'pr.png'
    if pr_img.exists():
        st.image(str(pr_img), caption='Precision-Recall Curve')
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            st.markdown('Metrics')
            st.json(metrics)
        except Exception:
            st.info('Metrics file present but unreadable.')
    st.divider()
    st.caption('Below is a live preview computed from current features and model (if available).')
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button('Run Evaluation (Fraud)'):
            res = subprocess.run(['python','-m','src.jobs.evaluate','--domain','fraud'], capture_output=True, text=True)
            st.info(res.stdout or res.stderr)
    with col_btn2:
        if st.button('Calibrated Train (Fraud)'):
            res = subprocess.run(['python','-m','src.jobs.train','--domain','fraud','--calibrate','true'], capture_output=True, text=True)
            st.info(res.stdout or res.stderr)

    feat_path = Path('data/features/fraud/fraud_features.parquet')
    model_path = MODELS_DIR / 'fraud_xgb_isotonic.joblib'
    if not model_path.exists():
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
        # Lift curve style for fraud (capture at top-k)
        order = np.argsort(-proba)
        y_sorted = y.iloc[order].to_numpy()
        cum_pos = np.cumsum(y_sorted)
        frac = np.arange(1, len(y_sorted)+1) / len(y_sorted)
        baseline = y.mean() * np.arange(1, len(y_sorted)+1)
        lift = cum_pos / np.maximum(baseline, 1e-6)
        fig2, ax2 = plt.subplots()
        ax2.plot(frac, lift)
        ax2.set_xlabel('Population fraction')
        ax2.set_ylabel('Cumulative lift')
        st.pyplot(fig2)
        # SHAP summary (sampled)
        try:
            def unwrap_estimator(m):
                base = getattr(m, 'base_estimator', None)
                if base is not None:
                    return base
                ccs = getattr(m, 'calibrated_classifiers_', None)
                if ccs:
                    return ccs[0].estimator
                return m
            est = unwrap_estimator(model)
            explainer = shap.TreeExplainer(est)
            sample_idx = np.random.RandomState(0).choice(len(X), size=min(1000, len(X)), replace=False)
            shap_vals = explainer.shap_values(X.iloc[sample_idx])
            fig3 = plt.figure()
            shap.summary_plot(shap_vals, X.iloc[sample_idx], show=False)
            st.pyplot(fig3)
        except Exception as e:
            st.caption(f'SHAP not available: {e}')
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
