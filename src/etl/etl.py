from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.io import ensure_dir


def etl_lendingclub(raw_paths: list[Path], out_root: Path) -> list[Path]:
    out_root = ensure_dir(out_root)
    outputs: list[Path] = []
    for p in raw_paths:
        df = pd.read_csv(p, parse_dates=["issue_d"])  # basic parse
        df["year"] = df["issue_d"].dt.year
        df["month"] = df["issue_d"].dt.month
        part_dir = out_root / f"year={df['year'].iloc[0]}" / f"month={df['month'].iloc[0]:02d}"
        ensure_dir(part_dir)
        out_file = part_dir / "lendingclub.parquet"
        df.to_parquet(out_file, index=False)
        outputs.append(out_file)
    return outputs

