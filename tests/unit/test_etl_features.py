from pathlib import Path

from src.ingestion.downloaders import download_lendingclub
from src.etl.etl import etl_lendingclub
from src.features.risk import build_risk_features


def test_etl_and_features_tmp(tmp_path: Path):
    raw = tmp_path / "raw/lendingclub"
    interim = tmp_path / "interim/lendingclub"
    features = tmp_path / "features/credit"
    paths = download_lendingclub([2020], raw, sample=True)
    out_parts = etl_lendingclub(paths, interim)
    assert out_parts, "No ETL outputs created"
    feat_path = build_risk_features(out_parts, features)
    assert feat_path.exists(), "Features parquet not created"

