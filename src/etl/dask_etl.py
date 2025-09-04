from __future__ import annotations

from pathlib import Path
import dask.dataframe as dd


def dask_etl_lendingclub(input_glob: str, out_root: Path) -> None:
    df = dd.read_csv(input_glob, assume_missing=True)
    # Parse and clean
    if 'issue_d' in df.columns:
        df['issue_d'] = dd.to_datetime(df['issue_d'], errors='coerce')
    if 'addr_state' in df.columns:
        df['state'] = df['addr_state'].astype('object')
    if 'int_rate' in df.columns:
        df['int_rate'] = df['int_rate'].astype('object').str.replace('%','').astype('float64')
    if 'term' in df.columns:
        df['term'] = df['term'].astype('object').str.extract(r'(\d+)', expand=False).astype('float64').fillna(36).astype('int64')
    if 'loan_status' in df.columns:
        bad = df['loan_status'].astype('object').str.lower().str.contains('charged off|default|late (31-120 days)', regex=True)
        df['defaulted'] = bad.fillna(False).astype('int64')
    # Essential columns
    keep = [c for c in ['issue_d','loan_amnt','int_rate','dti','term','state','defaulted'] if c in df.columns]
    df = df[keep]
    df = df.dropna(subset=['issue_d'])
    df['year'] = df['issue_d'].dt.year
    df['month'] = df['issue_d'].dt.month
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_root), engine='pyarrow', write_index=False, partition_on=['year','month'], overwrite=True)


def dask_etl_kaggle_cc(input_glob: str, out_root: Path) -> None:
    df = dd.read_csv(input_glob, assume_missing=True)
    base = dd.to_datetime('2018-01-01')
    if 'Time' in df.columns:
        df['issue_d'] = dd.to_datetime('2018-01-01') + dd.to_timedelta(df['Time'], unit='s')
    else:
        df['issue_d'] = dd.to_datetime('2018-01-01')
    df['amount'] = df['Amount'] if 'Amount' in df.columns else df.get('amount', 0)
    df['is_fraud'] = df['Class'] if 'Class' in df.columns else df.get('is_fraud', 0)
    df = df[['issue_d','amount','is_fraud']]
    df['year'] = df['issue_d'].dt.year
    df['month'] = df['issue_d'].dt.month
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_root), engine='pyarrow', write_index=False, partition_on=['year','month'], overwrite=True)

