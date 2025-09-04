from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.io import ensure_dir
try:
    from great_expectations.dataset import PandasDataset  # lightweight usage
except Exception:  # pragma: no cover
    PandasDataset = None


def etl_lendingclub(raw_paths: list[Path], out_root: Path) -> list[Path]:
    out_root = ensure_dir(out_root)
    outputs: list[Path] = []
    for p in raw_paths:
        # Robust parse for LendingClub accepted CSVs
        df = pd.read_csv(p)
        # Parse dates that look like 'Dec-2015' or ISO
        if "issue_d" in df.columns:
            df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
        # Harmonize state
        if "addr_state" in df.columns and "state" not in df.columns:
            df["state"] = df["addr_state"].astype(str)
        # Clean interest rate to numeric percent
        if "int_rate" in df.columns:
            df["int_rate"] = (
                df["int_rate"].astype(str).str.replace("%", "", regex=False).astype(float)
            )
        # Clean term to months integer
        if "term" in df.columns:
            term_num = df["term"].astype(str).str.extract(r"(\d+)")[0]
            term_num = pd.to_numeric(term_num, errors="coerce").fillna(36).astype(int)
            df["term"] = term_num
        # Create defaulted target from loan_status if present
        if "defaulted" not in df.columns and "loan_status" in df.columns:
            bad_status = (
                df["loan_status"].astype(str).str.contains(
                    r"Charged Off|Default|Late \(31-120 days\)", regex=True, case=False
                )
            )
            df["defaulted"] = bad_status.astype(int)
        # retain essential columns to avoid mixed dtypes
        keep = [
            "issue_d",
            "loan_amnt",
            "int_rate",
            "dti",
            "term",
            "state",
            "defaulted",
        ]
        df = df[[c for c in keep if c in df.columns]].copy()
        # basic partitions
        df = df.dropna(subset=["issue_d"])  # ensure partitionable
        df["year"] = df["issue_d"].dt.year
        df["month"] = df["issue_d"].dt.month
        # Data quality checks (Great Expectations - minimal)
        if PandasDataset is not None:
            ds = PandasDataset(df)
            res = ds.expect_column_values_to_not_be_null("loan_amnt")
            res2 = ds.expect_column_values_to_be_between("int_rate", min_value=0, max_value=100)
            if not (res.success and res2.success):
                raise ValueError("Data validation failed for LendingClub ETL")
        part_dir = out_root / f"year={df['year'].iloc[0]}" / f"month={df['month'].iloc[0]:02d}"
        ensure_dir(part_dir)
        out_file = part_dir / "lendingclub.parquet"
        df.to_parquet(out_file, index=False)
        outputs.append(out_file)
    return outputs
