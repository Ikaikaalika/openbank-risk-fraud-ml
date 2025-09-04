from __future__ import annotations

from pathlib import Path
import typer

from src.models.credit_risk import train_credit_risk
from src.models.fraud import train_fraud

app = typer.Typer()


@app.command()
def main(
    domain: str = typer.Option("credit", help="credit or fraud"),
    features_root: str = typer.Option("data/features", help="Features root"),
    models_dir: str = typer.Option("models", help="Model output directory"),
    calibrate: bool = typer.Option(False, help="Apply isotonic calibration on validation split"),
):
    features_path = Path(features_root) / ("credit/risk_features.parquet" if domain == "credit" else "fraud/fraud_features.parquet")
    models_out = Path(models_dir)
    models_out.mkdir(parents=True, exist_ok=True)
    if domain == "credit":
        metrics = train_credit_risk(features_path, models_out, calibrate=calibrate)
    elif domain == "fraud":
        metrics = train_fraud(features_path, models_out, calibrate=calibrate)
    else:
        raise typer.BadParameter("domain must be 'credit' or 'fraud'")
    typer.echo(f"Trained {domain} model. Metrics: {metrics}")


if __name__ == "__main__":
    app()
