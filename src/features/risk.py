from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.common.io import ensure_dir


def build_risk_features(parquet_paths: list[Path], out_dir: Path) -> Path:
    ensure_dir(out_dir)
    dfs = [pd.read_parquet(p) for p in parquet_paths]
    df = pd.concat(dfs, ignore_index=True)
    # simple engineered ratios
    df["int_rate_pct"] = df["int_rate"] / 100.0
    df["dti_clipped"] = df["dti"].clip(0, 60)
    features = df[[
        "loan_amnt",
        "int_rate_pct",
        "dti_clipped",
        "term",
        "defaulted",
        "year",
        "month",
    ]].copy()
    out_path = out_dir / "risk_features.parquet"
    features.to_parquet(out_path, index=False)
    return out_path

