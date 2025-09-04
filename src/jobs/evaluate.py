from __future__ import annotations

from pathlib import Path
import typer
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

app = typer.Typer()


@app.command()
def main(domain: str = typer.Option("credit")):
    if domain == "credit":
        df = pd.read_parquet("data/features/credit/risk_features.parquet")
        auc = roc_auc_score(df["defaulted"], (df["int_rate_pct"] * 0.6 + df["dti_clipped"] / 100.0 * 0.4))
        typer.echo(f"Credit dummy AUC: {auc:.3f}")
    else:
        df = pd.read_parquet("data/features/fraud/fraud_features.parquet")
        ap = average_precision_score(df["is_fraud"], df["amt_z"].abs())
        typer.echo(f"Fraud dummy PR-AUC: {ap:.3f}")


if __name__ == "__main__":
    app()

