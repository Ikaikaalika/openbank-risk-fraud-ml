from __future__ import annotations

from pathlib import Path
import typer
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

app = typer.Typer()


@app.command()
def main(domain: str = typer.Option("credit"), reports_root: str = typer.Option("reports/monitoring")):
    reports_dir = Path(reports_root) / "2020-01"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if domain == "credit":
        df = pd.read_parquet("data/features/credit/risk_features.parquet")
    else:
        df = pd.read_parquet("data/features/fraud/fraud_features.parquet")
    ref = df.sample(frac=0.5, random_state=42)
    cur = df.drop(ref.index)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    out_html = reports_dir / f"{domain}_drift.html"
    report.save_html(str(out_html))
    typer.echo(f"Monitoring report: {out_html}")


if __name__ == "__main__":
    app()

