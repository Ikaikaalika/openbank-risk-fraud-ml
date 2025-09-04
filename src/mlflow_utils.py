from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import os


def set_local_tracking(uri: str = "./mlruns") -> None:
    mlflow.set_tracking_uri(Path(uri).absolute().as_uri())


def start_run(model_name: str):
    return mlflow.start_run(tags={"model_name": model_name})


def log_artifacts_and_metrics(params: dict, metrics: dict, artifact_path: Path) -> None:
    if params:
        mlflow.log_params(params)
    if metrics:
        mlflow.log_metrics(metrics)
    if artifact_path.exists():
        mlflow.log_artifact(str(artifact_path))


def find_latest_artifact(model_name: str, filename: str, experiment_name: str = "Default") -> Optional[Path]:
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id if exp else "0"
    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.model_name = '{model_name}'",
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    run = runs[0]
    art_uri = run.info.artifact_uri  # e.g., file:///.../mlruns/0/<run_id>/artifacts
    if art_uri.startswith("file://"):
        art_path = Path(art_uri.replace("file://", ""))
    else:
        # fallback: treat as local path
        art_path = Path(art_uri)
    target = art_path / filename
    return target if target.exists() else None


def register_model(artifact_path: Path, model_name: str, await_registration: bool = False) -> Optional[str]:
    """Attempt to register a model in MLflow Model Registry.

    Returns the registered model version URI if successful; None otherwise.
    Requires MLflow tracking server with SQL backend.
    """
    try:
        result = mlflow.register_model(model_uri=str(artifact_path), name=model_name)
        if await_registration:
            client = MlflowClient()
            client.transition_model_version_stage(name=model_name, version=result.version, stage="Staging", archive_existing_versions=False)
        return f"models:/{model_name}/{getattr(result,'version','1')}"
    except Exception:
        return None


def load_model_from_uri(uri: str):
    """Load a model using MLflow's pyfunc given a URI (models:/, runs:/, file://, etc.)."""
    try:
        return mlflow.pyfunc.load_model(uri)
    except Exception:
        return None


def env_model_uri(key: str) -> Optional[str]:
    uri = os.getenv(key)
    return uri if uri else None
