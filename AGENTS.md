# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Python source code
  - `src/jobs/`: batch jobs (e.g., Spark ETL) — `run_etl_spark.py`
  - `src/monitoring/`: ML monitoring utilities — `drift.py`, `fairness.py`
  - `src/mlflow_utils.py`: MLflow helpers (stub)
- `dashboard/`: Streamlit UI — `app.py`
- `infra/docker/`: Dockerfiles — `Dockerfile.spark`, `Dockerfile.api`
- `docs/`: Ops/EMR notes — `EMR_walkthrough.md`
- `reports/`: Generated outputs and artifacts
- `requirements.txt`: Python dependencies

## Build, Test, and Development Commands
- Install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run Spark ETL: `python src/jobs/run_etl_spark.py --input data/raw/lendingclub.csv --out data/interim/lendingclub_spark`
- Launch dashboard: `streamlit run dashboard/app.py`
- Docker (Spark job): `docker build -f infra/docker/Dockerfile.spark -t risk-spark . && docker run --rm -v "$PWD":/app risk-spark`
- Docker (API): `docker build -f infra/docker/Dockerfile.api -t risk-api .` (serving app path is a placeholder; add `src/serving/app.py` before running)

## Coding Style & Naming Conventions
- Python 3.11, PEP 8, 4-space indentation, descriptive names.
- Modules: lowercase with underscores (e.g., `data_utils.py`); classes: `CamelCase`; functions/vars: `snake_case`.
- Prefer type hints; validate external I/O with `pydantic` where practical.
- No required formatter in repo; recommended: `pip install black ruff` then `black . && ruff .`.

## Testing Guidelines
- No test suite yet. Add `pytest` tests under `tests/` mirroring `src/` (e.g., `tests/monitoring/test_drift.py`).
- Naming: `test_*.py`, functions `test_*` with clear, behavior-focused names.
- Run tests: `pytest -q`; aim to cover data paths and failure modes; target >80% coverage if feasible (`pytest --cov=src`).

## Commit & Pull Request Guidelines
- Conventional history not established. Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- PRs: include purpose, scope, screenshots for UI (Streamlit), and steps to reproduce. Link issues and note any config/data requirements.
- CI-friendly changes: keep diffs focused; update docs in `docs/` and sample commands in this file when behavior changes.

## Security & Configuration Tips
- Do not commit credentials or raw PII. Use environment variables or a `.env` (excluded) for secrets.
- Large data: store outside the repo; reference via relative paths (see ETL defaults).
- For EMR/Spark, follow `docs/EMR_walkthrough.md`; validate IAM and network policies before running jobs.
