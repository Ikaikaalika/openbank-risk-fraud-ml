from __future__ import annotations

from pathlib import Path
from typing import List

from src.common.io import ensure_dir


def organize_existing(files: List[Path], dest_root: Path) -> list[Path]:
    # Place under <dataset>/<year>/<month>
    out: list[Path] = []
    for f in files:
        y = '2018'
        m = '01'
        d = ensure_dir(dest_root / y / m)
        tgt = d / f.name
        if not tgt.exists():
            tgt.symlink_to(f)
        out.append(tgt)
    return out

