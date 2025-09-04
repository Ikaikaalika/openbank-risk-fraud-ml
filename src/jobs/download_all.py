from __future__ import annotations

from pathlib import Path
import typer

from src.ingestion.downloaders import download_lendingclub, download_kaggle_cc
from pathlib import Path
from typing import Optional
from src.ingestion.lendingclub import organize_existing as lc_organize
from src.ingestion.kaggle_cc import organize_existing as kcc_organize
from src.ingestion.ieee_cis import organize_existing as ieee_organize
from src.ingestion.fannie_freddie import organize_existing as ff_organize

app = typer.Typer()


@app.command()
def main(
    data_root: str = typer.Option("data/raw", help="Raw data root"),
    sources: list[str] = typer.Option(["lendingclub", "kaggle_cc"], help="Sources"),
    sample: bool = typer.Option(True, help="Generate small sample files"),
    lc_files: Optional[str] = typer.Option(None, help="Comma-separated paths to existing LendingClub CSVs"),
    kcc_files: Optional[str] = typer.Option(None, help="Comma-separated paths to Kaggle CC CSVs"),
    ieee_files: Optional[str] = typer.Option(None, help="Comma-separated paths to IEEE-CIS CSVs"),
    fannie_files: Optional[str] = typer.Option(None, help="Comma-separated paths to Fannie Mae files"),
    freddie_files: Optional[str] = typer.Option(None, help="Comma-separated paths to Freddie Mac files"),
):
    root = Path(data_root)
    if "lendingclub" in sources:
        if lc_files:
            paths = [Path(p.strip()) for p in lc_files.split(",")]
            lc_organize(paths, root / "lendingclub")
        else:
            download_lendingclub(years=[2020], dest=root / "lendingclub", sample=sample)
    if "kaggle_cc" in sources:
        if kcc_files:
            paths = [Path(p.strip()) for p in kcc_files.split(",")]
            kcc_organize(paths, root / "kaggle_cc")
        else:
            download_kaggle_cc(dest=root, sample=sample)
    if "ieee_cis" in sources and ieee_files:
        paths = [Path(p.strip()) for p in ieee_files.split(",")]
        ieee_organize(paths, root / "ieee_cis")
    if "fannie" in sources and fannie_files:
        paths = [Path(p.strip()) for p in fannie_files.split(",")]
        ff_organize(paths, root / "fannie")
    if "freddie" in sources and freddie_files:
        paths = [Path(p.strip()) for p in freddie_files.split(",")]
        ff_organize(paths, root / "freddie")
    typer.echo("Downloads/organization complete.")


if __name__ == "__main__":
    app()
