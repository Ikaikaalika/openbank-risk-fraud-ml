from __future__ import annotations

from pathlib import Path
import typer

from src.ingestion.downloaders import download_lendingclub, download_kaggle_cc

app = typer.Typer()


@app.command()
def main(
    data_root: str = typer.Option("data/raw", help="Raw data root"),
    sources: list[str] = typer.Option(["lendingclub", "kaggle_cc"], help="Sources"),
    sample: bool = typer.Option(True, help="Generate small sample files"),
):
    root = Path(data_root)
    if "lendingclub" in sources:
        download_lendingclub(years=[2020], dest=root / "lendingclub", sample=sample)
    if "kaggle_cc" in sources:
        download_kaggle_cc(dest=root, sample=sample)
    typer.echo("Downloads complete.")


if __name__ == "__main__":
    app()

