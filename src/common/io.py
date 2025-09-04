from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_path(root: Path | str, *parts: str) -> Path:
    return Path(root).joinpath(*parts)

