PY=python

.PHONY: setup format test download etl features train-risk train-fraud evaluate serve docker-api monitor

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
	pre-commit install || true

format:
	black . || true
	ruff . || true
	mypy src || true

test:
	pytest -q

download:
	$(PY) -m src.jobs.download_all --sources lendingclub kaggle_cc --sample

download-noaa-holidays:
	$(PY) -m src.jobs.download_all --sources lendingclub --noaa-station GHCND:USW00023174 --noaa-start 2018-01-01 --noaa-end 2018-01-31 --holidays-country US --holidays-years 2018,2019

etl:
	$(PY) -m src.jobs.run_etl --source lendingclub

etl-spark:
	$(PY) -m src.jobs.run_etl_spark --input "data/raw/lendingclub/*.csv" --out data/interim/lendingclub

etl-dask:
	$(PY) -m src.jobs.run_etl_dask --source lendingclub --input "data/raw/lendingclub/*.csv" --out data/interim/lendingclub

features:
	$(PY) -m src.jobs.build_features --domain credit
	$(PY) -m src.jobs.build_features --domain fraud

train-risk:
	$(PY) -m src.jobs.train --domain credit

train-fraud:
	$(PY) -m src.jobs.train --domain fraud

train-risk-cal:
	$(PY) -m src.jobs.train --domain credit --calibrate true

train-fraud-cal:
	$(PY) -m src.jobs.train --domain fraud --calibrate true

evaluate:
	$(PY) -m src.jobs.evaluate --domain credit
	$(PY) -m src.jobs.evaluate --domain fraud

serve:
	uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8080

docker-api:
	docker build -f infra/docker/Dockerfile.api -t varo-risk-api:latest .

monitor:
	$(PY) -m src.jobs.run_monitoring --domain credit
	$(PY) -m src.jobs.run_monitoring --domain fraud

validate:
	$(PY) -m src.jobs.validate_data --suite lendingclub_clean --data-glob "data/interim/lendingclub/**/*.parquet"
	$(PY) -m src.jobs.validate_data --suite kaggle_cc_clean --data-glob "data/interim/kaggle_cc/**/*.parquet"
