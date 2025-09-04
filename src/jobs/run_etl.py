from __future__ import annotations

from pathlib import Path
import typer

from src.ingestion.downloaders import download_lendingclub
from src.etl.etl import etl_lendingclub
from src.etl.kaggle_cc_etl import etl_kaggle_cc

app = typer.Typer()


@app.command()
def main(
    source: str = typer.Option("lendingclub", help="Data source"),
    raw_root: str = typer.Option("data/raw", help="Raw data root"),
    out_root: str = typer.Option("data/interim/lendingclub", help="ETL output root"),
    raw_path: str = typer.Option("", help="Optional direct path to a raw CSV file"),
):
    if source == "lendingclub":
        raw_paths = [Path(raw_path)] if raw_path else download_lendingclub(years=[2020], dest=Path(raw_root) / source, sample=True)
        outputs = etl_lendingclub(raw_paths, Path(out_root))
    elif source == "kaggle_cc":
        raw_paths = [Path(raw_path)] if raw_path else list((Path(raw_root) / source).rglob("*.csv"))
        if not raw_paths:
            raise typer.BadParameter("No Kaggle CC CSVs found; provide --raw-path or place files under data/raw/kaggle_cc/")
        outputs = etl_kaggle_cc(raw_paths, Path(out_root).with_name('kaggle_cc'))
    else:
        raise typer.BadParameter("Unsupported source; choose 'lendingclub' or 'kaggle_cc'")
    typer.echo(f"ETL wrote {len(outputs)} partitions to {out_root}")


if __name__ == "__main__":
    app()
