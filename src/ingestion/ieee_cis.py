from __future__ import annotations

from pathlib import Path
from typing import List

from src.common.io import ensure_dir


def organize_existing(files: List[Path], dest_root: Path) -> list[Path]:
    # IEEE-CIS has train_transaction.csv etc.; keep under 2018/11
    d = ensure_dir(dest_root / '2018' / '11')
    out: list[Path] = []
    for f in files:
        tgt = d / f.name
        if not tgt.exists():
            tgt.symlink_to(f)
        out.append(tgt)
    return out

