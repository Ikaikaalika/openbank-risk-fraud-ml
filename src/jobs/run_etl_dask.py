from __future__ import annotations

from pathlib import Path
import typer

from src.etl.dask_etl import dask_etl_lendingclub, dask_etl_kaggle_cc

app = typer.Typer()


@app.command()
def main(
    source: str = typer.Option("lendingclub", help="lendingclub|kaggle_cc"),
    input: str = typer.Option("data/raw/lendingclub/*.csv", help="Input glob"),
    out: str = typer.Option("data/interim/lendingclub", help="Output root"),
):
    out_root = Path(out)
    if source == "lendingclub":
        dask_etl_lendingclub(input, out_root)
    elif source == "kaggle_cc":
        if "lendingclub" in out:
            out_root = Path(out.replace("lendingclub","kaggle_cc"))
        dask_etl_kaggle_cc(input, out_root)
    else:
        raise typer.BadParameter("Unsupported source for dask ETL")
    typer.echo(f"Dask ETL wrote to {out_root}")


if __name__ == "__main__":
    app()

