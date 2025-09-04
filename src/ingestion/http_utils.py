from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.common.io import ensure_dir


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _stream_download(url: str, dest: Path, timeout: int = 60) -> None:
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + '.part')
        with tmp.open('wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)


def download_file(url: str, dest: Path, expected_sha256: Optional[str] = None, timeout: int = 60) -> Path:
    ensure_dir(dest.parent)
    _stream_download(url, dest, timeout=timeout)
    if expected_sha256:
        actual = sha256sum(dest)
        if actual.lower() != expected_sha256.lower():
            dest.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {dest.name}: {actual} != {expected_sha256}")
    return dest

