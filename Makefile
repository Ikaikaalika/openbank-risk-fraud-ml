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

etl:
	$(PY) -m src.jobs.run_etl --source lendingclub

features:
	$(PY) -m src.jobs.build_features --domain credit
	$(PY) -m src.jobs.build_features --domain fraud

train-risk:
	$(PY) -m src.jobs.train --domain credit

train-fraud:
	$(PY) -m src.jobs.train --domain fraud

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

