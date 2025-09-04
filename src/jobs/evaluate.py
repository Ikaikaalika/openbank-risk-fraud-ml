from __future__ import annotations

from pathlib import Path
import numpy as np
import typer
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from fairlearn.metrics import MetricFrame, selection_rate

app = typer.Typer()


@app.command()
def psi(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    cts, brks = np.histogram(a, bins=bins)
    cts_b, _ = np.histogram(b, bins=brks)
    p = np.maximum(cts / max(cts.sum(), 1), 1e-6)
    q = np.maximum(cts_b / max(cts_b.sum(), 1), 1e-6)
    return float(np.sum((p - q) * np.log(p / q)))


def main(domain: str = typer.Option("credit")):
    if domain == "credit":
        df = pd.read_parquet("data/features/credit/risk_features.parquet")
        score = (df["int_rate_pct"] * 0.6 + df["dti_clipped"] / 100.0 * 0.4)
        auc = roc_auc_score(df["defaulted"], score)
        # Fairness by state (if available)
        if "state" in df.columns:
            mf = MetricFrame(metrics={"selection_rate": selection_rate}, y_true=df["defaulted"], y_pred=score > score.median(), sensitive_features=df["state"])
            fairness_summary = mf.by_group.to_dict()["selection_rate"]
        else:
            fairness_summary = {}
        # PSI stability between halves
        s1 = score.iloc[: len(score) // 2].to_numpy()
        s2 = score.iloc[len(score) // 2 :].to_numpy()
        stability_psi = psi(s1, s2)
        typer.echo(f"Credit AUC={auc:.3f} PSI={stability_psi:.3f} Fairness(slices)={fairness_summary}")
    else:
        df = pd.read_parquet("data/features/fraud/fraud_features.parquet")
        ap = average_precision_score(df["is_fraud"], df["amt_z"].abs())
        s1 = df["amt_z"].abs().iloc[: len(df) // 2].to_numpy()
        s2 = df["amt_z"].abs().iloc[len(df) // 2 :].to_numpy()
        stability_psi = psi(s1, s2)
        typer.echo(f"Fraud PR-AUC={ap:.3f} PSI={stability_psi:.3f}")


if __name__ == "__main__":
    app()
