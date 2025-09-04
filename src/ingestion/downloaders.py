from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from src.common.io import ensure_dir


def download_lendingclub(years: List[int] | None, dest: Path, sample: bool = False) -> list[Path]:
    """Create a tiny sample CSV for LendingClub.

    In a real implementation, this would download official CSVs.
    """
    dest = ensure_dir(dest)
    out_paths: list[Path] = []
    year = years[0] if years else 2020
    d = ensure_dir(dest / str(year) / "01")
    p = d / "lendingclub_sample.csv"
    if not p.exists() or sample:
        p.write_text(
            "issue_d,loan_amnt,int_rate,dti,term,purpose,state,defaulted\n"
            "2020-01-15,10000,13.5,18.2,36,credit_card,CA,0\n"
            "2020-01-20,5000,8.7,12.5,36,car,TX,0\n"
            "2020-01-25,15000,19.9,35.1,60,small_business,NY,1\n"
        )
    out_paths.append(p)
    return out_paths


def download_kaggle_cc(dest: Path, sample: bool = False) -> list[Path]:
    dest = ensure_dir(dest / "kaggle_cc" / "2020" / "01")
    p = dest / "creditcard_sample.csv"
    if not p.exists() or sample:
        p.write_text(
            "time,amount,device_ip,merchant,is_fraud\n"
            "1,100.5,10.0.0.1,AMZN,0\n"
            "2,999.0,10.0.0.2,EBAY,1\n"
            "3,12.3,10.0.0.1,STARBUCKS,0\n"
        )
    return [p]


def clean_raw(root: Path) -> None:
    """Remove raw data directory (helper for tests)."""
    if root.exists():
        shutil.rmtree(root)

