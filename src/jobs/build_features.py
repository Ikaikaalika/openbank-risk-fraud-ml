from __future__ import annotations

from pathlib import Path
import typer

from src.features.risk import build_risk_features
from src.features.fraud import build_fraud_features
from src.features.manifest import write_manifest

app = typer.Typer()


@app.command()
def main(
    domain: str = typer.Option("credit", help="credit or fraud"),
    interim_root: str = typer.Option("data/interim/lendingclub", help="Interim root for credit"),
    raw_root: str = typer.Option("data/raw/kaggle_cc/2020/01", help="Raw root for fraud"),
    out_root: str = typer.Option("data/features", help="Features root"),
):
    out_dir = Path(out_root) / ("credit" if domain == "credit" else "fraud")
    out_dir.mkdir(parents=True, exist_ok=True)
    if domain == "credit":
        parts = list(Path(interim_root).rglob("*.parquet"))
        path = build_risk_features(parts, out_dir)
        write_manifest("credit", path, parts, ["loan_amnt","int_rate_pct","dti_clipped","term","defaulted","state","year","month"], out_dir)
    elif domain == "fraud":
        csvs = list(Path(raw_root).rglob("*.csv"))
        path = build_fraud_features(csvs, out_dir)
        write_manifest("fraud", path, csvs, ["time","amount","amt_z","is_fraud"], out_dir)
    else:
        raise typer.BadParameter("domain must be 'credit' or 'fraud'")
    typer.echo(f"Features saved: {path}")


if __name__ == "__main__":
    app()
