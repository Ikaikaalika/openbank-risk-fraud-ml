from __future__ import annotations

from pathlib import Path
import typer
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import joblib

from src.mlflow_utils import env_model_uri, load_model_from_uri

app = typer.Typer()


def _load_model(domain: str):
    if domain == "credit":
        uri = env_model_uri("MLFLOW_MODEL_URI_CREDIT")
        default = Path("models/credit_risk_xgb_isotonic.joblib")
    else:
        uri = env_model_uri("MLFLOW_MODEL_URI_FRAUD")
        default = Path("models/fraud_xgb_isotonic.joblib")
    if uri:
        m = load_model_from_uri(uri)
        if m is not None:
            return m
    if default.exists():
        return joblib.load(default)
    # fallback to non-calibrated
    alt = Path(str(default).replace("_isotonic",""))
    return joblib.load(alt) if alt.exists() else None


@app.command()
def main(
    domain: str = typer.Option("credit", help="credit|fraud"),
    input_path: str = typer.Option(..., help="Input Parquet dataset path"),
    output_path: str = typer.Option(..., help="Output Parquet path"),
    batch_size: int = typer.Option(50000, help="Rows per prediction batch"),
):
    model = _load_model(domain)
    if model is None:
        raise typer.BadParameter("No model available for batch scoring.")
    dataset = ds.dataset(input_path, format="parquet")
    scanner = ds.Scanner.from_dataset(dataset, columns=None, batch_size=batch_size)
    out_dir = Path(output_path)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for batch in scanner.to_batches():
        pdf = batch.to_pandas()
        if domain == "credit":
            X = pdf[["loan_amnt","int_rate_pct","dti_clipped","term"]]
        else:
            X = pdf[["time","amount","amt_z"]]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            proba = np.asarray(model.predict(X)).reshape(-1)
        pdf = pdf.copy()
        pdf["score"] = proba
        frames.append(pdf)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(output_path, index=False)
    typer.echo(f"Wrote predictions to {output_path} ({len(out)} rows)")


if __name__ == "__main__":
    app()

