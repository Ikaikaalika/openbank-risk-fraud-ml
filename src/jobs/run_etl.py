from __future__ import annotations

from pathlib import Path
import typer

from src.ingestion.downloaders import download_lendingclub
from src.etl.etl import etl_lendingclub

app = typer.Typer()


@app.command()
def main(
    source: str = typer.Option("lendingclub", help="Data source"),
    raw_root: str = typer.Option("data/raw", help="Raw data root"),
    out_root: str = typer.Option("data/interim/lendingclub", help="ETL output root"),
    raw_path: str = typer.Option("", help="Optional direct path to a raw CSV file"),
):
    if source != "lendingclub":
        raise typer.BadParameter("Only 'lendingclub' is implemented in sample")
    if raw_path:
        raw_paths = [Path(raw_path)]
    else:
        raw_paths = download_lendingclub(years=[2020], dest=Path(raw_root) / source, sample=True)
    outputs = etl_lendingclub(raw_paths, Path(out_root))
    typer.echo(f"ETL wrote {len(outputs)} partitions to {out_root}")


if __name__ == "__main__":
    app()
