# Open Banking Risk & Fraud Detection Platform — Build Spec (Codex-Ready)

**Owner:** Tyler  
**Target Role:** Sr. Staff Data Scientist, Machine Learning @ Varo Bank  
**Goal:** Produce a production-grade, end‑to‑end ML platform that models **credit default risk** and **transaction fraud** using public/open datasets, with scalable data engineering, explainable modeling, deployment, and monitoring.

---

## 0) Outcomes & Success Criteria

**Primary outcomes**
1. **Credit Risk Model (PD / default probability)** trained on large, real loan-level data with calibrated probabilities and business-ready scorecards.
2. **Fraud Detection Models** for card/e‑commerce transactions:
   - Supervised classifier (tabular baseline + GBDTs)
   - Sequential model (transformer/RNN) for per-customer streams
   - Optional **graph model** to uncover fraud rings (shared devices/IPs/merchants)
3. **Scalable pipelines** (Spark/Dask) with reproducible ETL and feature stores.
4. **Deployed API service** (FastAPI + Docker) serving risk scores and fraud alerts.
5. **Responsible AI**: explainability (SHAP), fairness audit, drift & stability monitoring (Evidently).
6. **Executive dashboard** (Streamlit) showcasing lift charts, cost/benefit, fraud alerts map, and risk distribution.
7. **CI/CD**: tests, lint, type-check, model registry, and one-command deploy.

**Success metrics (minimum):**
- Credit risk: AUC ≥ 0.78, Brier score ≤ baseline (majority), well-calibrated (ECE ≤ 5%), PSI stability < 0.2 on holdout month.
- Fraud: PR‑AUC / F1 at low false-positive rate; compare at thresholds that achieve ≤ 2% manual review rate; top-decile lift ≥ 3x vs random.
- Serving: p95 latency < 150 ms for single record; batch scoring ≥ 5M rows/hour on 8‑core cloud VM.
- Monitoring: drift alerts on schema/feature/target; and bias slice report by relevant cohorts.

---

## 1) System Architecture (High-Level)

```
+---------------------+       +--------------------+       +-----------------+
| Raw Data Sources    |  -->  | Ingestion/ETL      |  -->  | Feature Store   |
| - LendingClub CSVs  |       | (Spark/Dask jobs)  |       | (Parquet + Hudi)|
| - Fannie/Freddie    |       +--------------------+       +-----------------+
| - Kaggle Fraud CC   |                 |                         |
| - IEEE-CIS Fraud    |                 v                         v
| - Weather (NOAA)    |            Model Training            Batch Scoring
| - Events (optional) |        (PyTorch/XGBoost/Spark)        + Online API
+---------------------+                 |                         |
                                        v                         v
                                 Registry/Artifacts         FastAPI Service
                                 (MLflow + S3)              (Docker + AWS)
                                        |
                                        v
                                 Monitoring/Drift/BI
                               (Evidently + Streamlit)
```

---

## 2) Data Sources (Public & Immediately Acquirable)

> **Note:** Download scripts must be idempotent and resumable; place all raw files under `data/raw/<source>/YYYYMM/` (or by chunk).

### Credit Risk (repayment outcomes)
- **LendingClub Loan Stats** (annual CSVs): https://www.lendingclub.com/investing/summary-statistics (archived mirrors exist on Kaggle)
- **Fannie Mae Single‑Family Loan Performance**: https://www.fanniemae.com/research-and-insights/dataset/single-family-loan-performance-data
- **Freddie Mac Single‑Family**: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

### Fraud Detection
- **Credit Card Fraud (Europe)**: https://www.kaggle.com/mlg-ulb/creditcardfraud
- **IEEE‑CIS Fraud Detection**: https://www.kaggle.com/c/ieee-fraud-detection

### Enrichment (optional)
- **NOAA weather**: https://www.ncei.noaa.gov/cdo-web/webservices/v2
- **Holidays**: https://github.com/dr-prodigy/python-holidays

---

## 3) Tech Stack & Standards

- **Python 3.11**; **Poetry** or `uv` for dependency mgmt; `ruff` (lint), `mypy` (types), `pytest`.
- **Spark 3.x** (PySpark) or **Dask** for ETL at scale.
- **Pandas/Polars** for local.
- **XGBoost/LightGBM/CatBoost** for tabular; **PyTorch** for sequential/graph models.
- **Feature store**: Parquet in S3 with Hudi/Delta (local: Parquet).
- **Tracking**: MLflow (artifacts to `./mlruns` locally; S3 in cloud).
- **Serving**: FastAPI + Uvicorn; **Docker**; **AWS** (S3, ECR, ECS/Fargate or EC2).
- **Monitoring**: Evidently + custom Prometheus metrics; Streamlit dashboard.
- **Infra as Code (optional)**: Terraform for S3, ECR, ECS, IAM.
- **Makefile** orchestration; **pre-commit** hooks.

---

## 4) Repo Layout (authoritative)

```
varo-ml-risk-fraud/
├─ Makefile
├─ pyproject.toml                # or requirements.txt + setup.cfg
├─ README.md
├─ .pre-commit-config.yaml
├─ src/
│  ├─ config/                    # Hydra/YAML configs
│  ├─ common/                    # utils: io, logging, timing, seed, metrics
│  ├─ ingestion/                 # raw data downloaders
│  ├─ etl/                       # Spark/Dask jobs -> clean tables
│  ├─ features/                  # feature pipelines & store IO
│  ├─ models/                    # training code
│  │  ├─ credit_risk/
│  │  └─ fraud/
│  ├─ serving/                   # FastAPI app, pydantic schemas
│  ├─ monitoring/                # drift, stability, fairness reports
│  └─ jobs/                      # CLI entrypoints (typer/click): train/score/etc.
├─ notebooks/
│  ├─ 01_eda_credit.ipynb
│  ├─ 02_eda_fraud.ipynb
│  └─ 03_model_cards.ipynb
├─ data/                         # gitignored
│  ├─ raw/
│  ├─ interim/
│  └─ features/
├─ models/                       # exported models (gitignored)
├─ mlruns/                       # MLflow tracking (gitignored)
├─ infra/                        # docker, terraform, ecs task, ci configs
│  ├─ docker/
│  │  ├─ Dockerfile.api
│  │  └─ Dockerfile.spark
│  ├─ terraform/
│  └─ ci/
└─ tests/
   ├─ unit/
   └─ integration/
```

---

## 5) Makefile Targets (one‑liners)

```
setup:            # install deps, pre-commit, create .env
format:           # ruff/mypy/black
test:             # pytest -q
download:         # python -m src.jobs.download_all
etl:              # python -m src.jobs.run_etl --source <name>
features:         # python -m src.jobs.build_features --domain <credit|fraud>
train-risk:       # python -m src.jobs.train --domain credit --model xgb
train-fraud:      # python -m src.jobs.train --domain fraud --model transformer
evaluate:         # python -m src.jobs.evaluate --domain <...>
serve:            # uvicorn src.serving.app:app --reload
docker-api:       # docker build -f infra/docker/Dockerfile.api -t varo-risk-api:latest .
deploy-ecs:       # terraform -chdir=infra/terraform apply
monitor:          # python -m src.jobs.run_monitoring
```

---

## 6) Configuration (Hydra/YAML)

- `src/config/` contains: `env.yaml` (paths, S3), `risk.yaml` (features/model), `fraud.yaml`, `serve.yaml` (ports, threads), `monitor.yaml`.

Example `risk.yaml`:
```yaml
dataset: "lendingclub"
target: "defaulted"
split:
  method: "time"     # train: YYYY-1..YYYY-2, valid: YYYY-3, test: YYYY-4
  train_end: "2019-12-31"
  valid_end: "2020-06-30"
model:
  type: "xgboost"
  params:
    max_depth: 6
    n_estimators: 600
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
calibration: "isotonic"
```

---

## 7) Ingestion & ETL (Spark/Dask)

### 7.1 Ingestion CLI
`python -m src.jobs.download_all --sources lendingclub fannie mae kaggle_cc ieee_cis`  
- Implement source-specific downloaders in `src/ingestion/` with retries and checksum validation.

**Example function signature:**
```python
def download_lendingclub(years: list[int], dest: Path) -> list[Path]:
    ...
```

### 7.2 ETL
Transform raw CSVs into partitioned Parquet with validated schemas.
- Use `pydantic` models to define schemas.
- Add **Great Expectations** checkpoints for data quality (nulls, ranges, unique keys).

`python -m src.jobs.run_etl --source lendingclub`

Outputs under `data/interim/lendingclub/` with partitions by `year=YYYY/month=MM`.

---

## 8) Feature Engineering

### Credit Risk
- **Numerical**: DTI, utilization, income, loan_amt, term, interest, age, credit_age, payment history aggregates (rolling 3/6/12m).
- **Categorical**: purpose, state, employment title (hashed), home ownership.
- **Temporal**: recency, loan vintage, macro signals (monthly unemployment rate proxy if available).
- **Target**: default (60+ DPD or charged-off).

### Fraud
- **Aggregates**: tx velocity per card/device/IP; amount z-scores; hour‑of‑day; merchant category frequency; distance between successive swipes (haversine).
- **Sequences**: per‑card ordered events (timestamps → transformer inputs).
- **Graph**: bipartite (card ↔ device/IP), project to card‑card edges via shared attributes.

Store engineered features in `data/features/<domain>/` with a **feature manifest** JSON for reproducibility.

---

## 9) Modeling

### 9.1 Baselines
- Credit risk: Logistic Regression, XGBoost/LightGBM.
- Fraud: XGBoost + class weights / focal loss; thresholding tuned for target review rate.

### 9.2 Advanced
- **Risk**: TabTransformer or FT-Transformer; calibrate with **isotonic** or **Platt**; scorecard export (WOE/IV optional).
- **Fraud (sequential)**: Transformer encoder over per‑card sequences (masking, peaky attention); contrastive pretraining on next‑tx prediction.
- **Fraud (graph)**: PyTorch Geometric (GraphSAGE/GCN) on card‑device graph; label propagation for cold start.

### 9.3 Training CLI
```
python -m src.jobs.train --domain credit --model xgb --config src/config/risk.yaml
python -m src.jobs.train --domain fraud --model transformer --config src/config/fraud.yaml
```

Log all runs to **MLflow** (params, metrics, artifacts, confusion matrices, PR curves).

---

## 10) Evaluation, Calibration, Fairness

- **Metrics**: AUC, PR‑AUC, KS, Brier, ECE, lift @ top‑k, cost curves.
- **Backtesting** by month/vintage with **time-based splits**.
- **Calibration**: reliability plots, isotonic/Platt saved to artifact.
- **Fairness**: compute group metrics (TPR/FPR/AUC/ECE) by slices (e.g., state, income bucket, age band) using `fairlearn`.
- **Stability**: PSI/CSI between train/val/test.

CLI:
```
python -m src.jobs.evaluate --domain credit --by-month
python -m src.jobs.evaluate --domain fraud --threshold-grid
```

Artifacts exported to `reports/` (HTML + PNG).

---

## 11) Serving (FastAPI)

- Endpoints:
  - `POST /score/credit` → returns `pd_default`, `reason_codes` (top SHAP features).
  - `POST /score/fraud` → returns `fraud_prob`, `action` (allow/review/decline), `explanations`.
- Input validation with **pydantic** schemas.
- Load latest registered models from MLflow model registry.

Run locally:
```
make serve
# or
uvicorn src.serving.app:app --host 0.0.0.0 --port 8080
```

---

## 12) Monitoring & Drift

- Nightly job computes:
  - **Data drift** (KS/JS) and **target drift** (if labels arrive).
  - **Performance** on delayed labels (where applicable).
  - **Bias** slice report monthly.
- Generate **Evidently** dashboards → `reports/monitoring/YYYY-MM/*.html`.
- Alerting stub (print/Slack webhook) when thresholds breached.

CLI:
```
python -m src.jobs.run_monitoring --domain fraud --window 7d
```

---

## 13) Dashboard (Streamlit)

- Tabs: **Overview, Credit Risk, Fraud, Monitoring**.
- Visuals: ROC/PR, lift, calibration, PSI; per‑customer SHAP waterfall; fraud geo heatmap; threshold dial targeting fixed review rate.
```
streamlit run dashboard/app.py
```

---

## 14) Reproducibility, Testing, and CI/CD

- **Determinism**: global seed; record package versions; `requirements.lock`.
- **Unit tests** for feature builders and metrics; **integration tests** for ETL → features → train happy path.
- **Pre-commit**: ruff, black, mypy, nbstripout.
- **GitHub Actions**:
  - Lint/type/test on PR.
  - Build & push Docker image (api) to ECR (if AWS creds).
  - Optional terraform plan/apply with manual approval.

---

## 15) Security & Privacy (Open Data Context)

- No PII beyond public datasets. Ensure configs do not log secrets.
- `.env` for creds; never commit. Provide `.env.example`.

---

## 16) Getting Started (Local, No-Cloud Path)

```
# 1) Bootstrap
python -m venv .venv && source .venv/bin/activate
pip install -U pip uv
uv pip install -r requirements.txt
pre-commit install

# 2) Download small samples for quick dev
python -m src.jobs.download_all --sources lendingclub kaggle_cc --sample

# 3) ETL + features
python -m src.jobs.run_etl --source lendingclub
python -m src.jobs.build_features --domain credit

# 4) Train
python -m src.jobs.train --domain credit --model xgb
python -m src.jobs.train --domain fraud --model xgb

# 5) Evaluate + reports
python -m src.jobs.evaluate --domain credit --by-month
python -m src.jobs.evaluate --domain fraud --threshold-grid

# 6) Serve API
uvicorn src.serving.app:app --port 8080
```

---

## 17) Stretch Goals (Show Senior Scope)

- **Causal uplift modeling** for acquisition or retention offers (T‑Learner/DR‑Learner).
- **Active learning** loop to optimize fraud review labeling.
- **Counterfactual explanations** (DiCE) for credit denials.
- **Online learning** sketch (streaming with Kafka + incremental models).

---

## 18) Deliverables Checklist (for PRD sign‑off)

- [ ] `Makefile` and all CLIs runnable end‑to‑end on a laptop (sample mode).
- [ ] Spark ETL path runs on a small EMR cluster (README walkthrough).
- [ ] Two registered models in MLflow with model cards.
- [ ] Deployed Docker image runs FastAPI; example cURL requests documented.
- [ ] Streamlit dashboard renders reports from latest artifacts.
- [ ] Monitoring job outputs monthly drift + fairness HTML reports.
- [ ] CI green on lint/type/tests; protected main branch.
- [ ] README with **business framing** (Varo‑style outcomes) + demo GIFs/screenshots.
