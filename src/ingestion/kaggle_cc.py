from __future__ import annotations

from pathlib import Path
from typing import List

from src.common.io import ensure_dir


def organize_existing(files: List[Path], dest_root: Path) -> list[Path]:
    # Place under data/raw/kaggle_cc/2018/01/
    d = ensure_dir(dest_root / '2018' / '01')
    out: list[Path] = []
    for f in files:
        tgt = d / f.name
        if not tgt.exists():
            tgt.symlink_to(f)
        out.append(tgt)
    return out

