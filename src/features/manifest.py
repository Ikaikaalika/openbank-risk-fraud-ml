from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


@dataclass
class FeatureManifest:
    domain: str
    features_path: str
    created_at: str
    source_inputs: List[str]
    columns: List[str]
    git_commit: str

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))


def write_manifest(domain: str, features_path: Path, source_inputs: List[Path], columns: List[str], out_dir: Path) -> Path:
    manifest = FeatureManifest(
        domain=domain,
        features_path=str(features_path),
        created_at=datetime.utcnow().isoformat() + "Z",
        source_inputs=[str(p) for p in source_inputs],
        columns=columns,
        git_commit=_git_commit(),
    )
    out_path = out_dir / "manifest.json"
    manifest.write(out_path)
    return out_path

