from __future__ import annotations

from pathlib import Path
import pandas as pd
from src.common.io import ensure_dir
try:
    from great_expectations.dataset import PandasDataset
except Exception:
    PandasDataset = None


def etl_kaggle_cc(raw_paths: list[Path], out_root: Path) -> list[Path]:
    out_root = ensure_dir(out_root)
    outputs: list[Path] = []
    for p in raw_paths:
        df = pd.read_csv(p)
        # Standard Kaggle CC has 'Time', 'Amount', 'Class'
        # Construct a pseudo date starting 2018-01-01 + seconds
        if 'Time' in df.columns:
            base = pd.Timestamp('2018-01-01')
            df['issue_d'] = base + pd.to_timedelta(df['Time'], unit='s')
        else:
            df['issue_d'] = pd.Timestamp('2018-01-01')
        df['amount'] = df['Amount'] if 'Amount' in df.columns else df.get('amount', 0)
        df['is_fraud'] = df['Class'] if 'Class' in df.columns else df.get('is_fraud', 0)
        # GE checks
        if PandasDataset is not None:
            ds = PandasDataset(df)
            if not ds.expect_column_values_to_be_between('amount', min_value=0).success:
                raise ValueError('Amount must be non-negative')
        df['year'] = df['issue_d'].dt.year
        df['month'] = df['issue_d'].dt.month
        part_dir = out_root / f"year={df['year'].iloc[0]}" / f"month={int(df['month'].iloc[0]):02d}"
        ensure_dir(part_dir)
        out_file = part_dir / 'kaggle_cc.parquet'
        df[['issue_d','amount','is_fraud','year','month']].to_parquet(out_file, index=False)
        outputs.append(out_file)
    return outputs

