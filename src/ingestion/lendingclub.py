from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from src.common.io import ensure_dir


def expected_files_for_years(years: Iterable[int]) -> List[str]:
    # Historical naming varies; we return canonical targets for organization.
    return [f"accepted_{y}.csv" for y in years]


def organize_existing(files: List[Path], dest_root: Path) -> list[Path]:
    out: list[Path] = []
    for f in files:
        # Place under data/raw/lendingclub/YYYY/01/
        try:
            # crude year detection from filename
            y = int(''.join([c for c in f.name if c.isdigit()])[:4]) if any(c.isdigit() for c in f.name) else 2018
        except Exception:
            y = 2018
        d = ensure_dir(dest_root / str(y) / '01')
        target = d / f.name
        if not target.exists():
            target.symlink_to(f)
        out.append(target)
    return out

