from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.common.io import ensure_dir


def build_fraud_features(csv_paths: list[Path], out_dir: Path) -> Path:
    ensure_dir(out_dir)
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    # basic aggregates
    df["amt_z"] = (df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-6)
    features = df[["time", "amount", "amt_z", "is_fraud"]].copy()
    out_path = out_dir / "fraud_features.parquet"
    features.to_parquet(out_path, index=False)
    return out_path

