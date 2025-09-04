from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import typer
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from fairlearn.metrics import MetricFrame, selection_rate
import matplotlib.pyplot as plt

app = typer.Typer()


@app.command()
def psi(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    cts, brks = np.histogram(a, bins=bins)
    cts_b, _ = np.histogram(b, bins=brks)
    p = np.maximum(cts / max(cts.sum(), 1), 1e-6)
    q = np.maximum(cts_b / max(cts_b.sum(), 1), 1e-6)
    return float(np.sum((p - q) * np.log(p / q)))


def _save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(
    domain: str = typer.Option("credit"),
    reports_root: str = typer.Option("reports/evaluation", help="Output reports directory"),
):
    if domain == "credit":
        df = pd.read_parquet("data/features/credit/risk_features.parquet")
        score = df["int_rate_pct"] * 0.6 + df["dti_clipped"] / 100.0 * 0.4
        y = df["defaulted"].astype(int)
        auc = roc_auc_score(y, score)
        # Fairness by state (if available)
        if "state" in df.columns:
            mf = MetricFrame(metrics={"selection_rate": selection_rate}, y_true=y, y_pred=score > score.median(), sensitive_features=df["state"])
            fairness_summary = mf.by_group.to_dict()["selection_rate"]
        else:
            fairness_summary = {}
        # PSI stability between halves
        s1 = score.iloc[: len(score) // 2].to_numpy()
        s2 = score.iloc[len(score) // 2 :].to_numpy()
        stability_psi = psi(s1, s2)
        # Plots: ROC, calibration/reliability, lift
        fpr, tpr, _ = roc_curve(y, score)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(loc="lower right")
        _save_fig(fig, Path(reports_root) / "credit" / "roc.png")
        # Reliability
        bins = np.linspace(0, 1, 11)
        bin_ids = np.digitize(score, bins) - 1
        prob_bin = [score[bin_ids == i].mean() if np.any(bin_ids == i) else np.nan for i in range(10)]
        avg_outcome = [y[bin_ids == i].mean() if np.any(bin_ids == i) else np.nan for i in range(10)]
        fig, ax = plt.subplots()
        ax.plot(prob_bin, avg_outcome, marker="o")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("Predicted prob"); ax.set_ylabel("Observed rate")
        _save_fig(fig, Path(reports_root) / "credit" / "reliability.png")
        # Lift at top 10%
        n = max(1, int(0.1 * len(score)))
        idx = np.argsort(-score)[:n]
        lift = float(y.iloc[idx].mean() / y.mean()) if y.mean() > 0 else float("nan")
        # Metrics JSON
        out = {
            "auc": float(auc),
            "psi_half_split": float(stability_psi),
            "lift_top10": lift,
            "fairness_selection_rate": fairness_summary,
        }
        outp = Path(reports_root) / "credit" / "metrics.json"
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, indent=2))
        typer.echo(f"Credit AUC={auc:.3f} PSI={stability_psi:.3f} Lift@10={lift:.2f}")
    else:
        df = pd.read_parquet("data/features/fraud/fraud_features.parquet")
        y = df["is_fraud"].astype(int)
        score = df["amt_z"].abs()
        ap = average_precision_score(y, score)
        s1 = score.iloc[: len(df) // 2].to_numpy()
        s2 = score.iloc[len(df) // 2 :].to_numpy()
        stability_psi = psi(s1, s2)
        # PR plot
        precision, recall, _ = precision_recall_curve(y, score)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f"AP={ap:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(loc="lower left")
        _save_fig(fig, Path(reports_root) / "fraud" / "pr.png")
        Path(reports_root).mkdir(parents=True, exist_ok=True)
        (Path(reports_root) / "fraud" / "metrics.json").write_text(json.dumps({"pr_auc": float(ap), "psi_half_split": float(stability_psi)}, indent=2))
        typer.echo(f"Fraud PR-AUC={ap:.3f} PSI={stability_psi:.3f}")


if __name__ == "__main__":
    app()
