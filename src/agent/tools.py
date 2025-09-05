from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.jobs import download_all as dl
from src.jobs import run_etl as etl
from src.jobs import run_etl_spark as etl_spark
from src.jobs import run_etl_dask as etl_dask
from src.jobs import build_features as features
from src.jobs import train as train_job
from src.jobs import evaluate as eval_job
from src.jobs import run_monitoring as monitoring
from src.jobs import batch_score as batch
from src.jobs import validate_data as validate


@dataclass
class ToolResult:
    ok: bool
    message: str


def _wrap(func: Callable[..., Any], **kwargs) -> ToolResult:
    try:
        func.callback = None  # in case Typer app
    except Exception:
        pass
    try:
        func(**kwargs)
        return ToolResult(True, f"Ran {func.__module__}.{func.__name__} with {kwargs}")
    except Exception as e:
        return ToolResult(False, f"Error in {func.__name__}: {e}")


def tool_download(sources: list[str], lc_files: Optional[str] = None, kcc_files: Optional[str] = None) -> ToolResult:
    return _wrap(dl.main, data_root="data/raw", sources=sources, sample=True, lc_files=lc_files, kcc_files=kcc_files, ieee_files=None, fannie_files=None, freddie_files=None, noaa_station=None, noaa_start=None, noaa_end=None, holidays_country=None, holidays_years=None)


def tool_etl(source: str, engine: str = "pandas") -> ToolResult:
    if engine == "spark":
        return _wrap(etl_spark.main, source=source, input=f"data/raw/{source}/*.csv", out=f"data/interim/{source}")
    if engine == "dask":
        return _wrap(etl_dask.main, source=source, input=f"data/raw/{source}/*.csv", out=f"data/interim/{source}")
    return _wrap(etl.main, source=source, raw_root="data/raw", out_root=f"data/interim/{source}", raw_path="")


def tool_features(domain: str) -> ToolResult:
    return _wrap(features.main, domain=domain, interim_root="data/interim/lendingclub", raw_root="data/raw/kaggle_cc/2018/01", out_root="data/features")


def tool_train(domain: str, calibrate: bool = False, register: bool = False) -> ToolResult:
    return _wrap(train_job.main, domain=domain, features_root="data/features", models_dir="models", calibrate=calibrate, register=register)


def tool_evaluate(domain: str) -> ToolResult:
    return _wrap(eval_job.main, domain=domain, reports_root="reports/evaluation")


def tool_monitor(domain: str) -> ToolResult:
    return _wrap(monitoring.main, domain=domain, reports_root="reports/monitoring")


def tool_batch_score(domain: str) -> ToolResult:
    if domain == "credit":
        return _wrap(batch.main, domain=domain, input_path="data/features/credit/risk_features.parquet", output_path="reports/predictions/credit_scores.parquet", batch_size=50000)
    else:
        return _wrap(batch.main, domain=domain, input_path="data/features/fraud/fraud_features.parquet", output_path="reports/predictions/fraud_scores.parquet", batch_size=50000)


def tool_validate(suite: str, data_glob: str) -> ToolResult:
    return _wrap(validate.main, suite=suite, data_glob=data_glob, out_dir="validation/results", sample_rows=100000)


def tool_read_file(path: str, max_bytes: int = 20000) -> ToolResult:
    p = Path(path)
    if not p.exists():
        return ToolResult(False, f"File not found: {path}")
    data = p.read_bytes()[:max_bytes]
    return ToolResult(True, data.decode(errors="ignore"))

