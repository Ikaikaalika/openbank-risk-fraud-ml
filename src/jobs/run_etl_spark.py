
from __future__ import annotations

import typer

from src.etl.spark_etl import main as spark_etl_main

app = typer.Typer()


@app.command()
def main(
    input: str = typer.Option("data/raw/lendingclub/*.csv", help="Input CSV glob or path"),
    out: str = typer.Option("data/interim/lendingclub", help="Output root directory"),
):
    spark_etl_main.callback = None
    spark_etl_main(input=input, out=out)


if __name__ == "__main__":
    app()
