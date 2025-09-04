from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.mlflow_utils import set_local_tracking, find_latest_artifact

MODELS_DIR = Path("models")

app = FastAPI(title="Varo Risk & Fraud API")


class CreditInput(BaseModel):
    loan_amnt: float
    int_rate: float = Field(..., description="Interest rate as percent, e.g., 13.5")
    dti: float
    term: int


class CreditOutput(BaseModel):
    pd_default: float
    reason_codes: list[str]


class FraudInput(BaseModel):
    time: float
    amount: float
    amt_z: Optional[float] = None


class FraudOutput(BaseModel):
    fraud_prob: float
    action: str
    explanations: list[str]


def load_model(name: str) -> Optional[object]:
    # 1) Load from local models dir
    local = MODELS_DIR / name
    if local.exists():
        return joblib.load(local)
    # 2) Try MLflow latest artifact (local mlruns)
    try:
        set_local_tracking()
        mlflow_path = find_latest_artifact(
            model_name="credit_risk" if name.startswith("credit") else "fraud",
            filename=name,
        )
        if mlflow_path and mlflow_path.exists():
            return joblib.load(mlflow_path)
    except Exception:
        pass
    return None


@app.post("/score/credit", response_model=CreditOutput)
def score_credit(inp: CreditInput):
    model = load_model("credit_risk_xgb.joblib")
    int_rate_pct = inp.int_rate / 100.0
    X = [[inp.loan_amnt, int_rate_pct, min(max(inp.dti, 0.0), 60.0), inp.term]]
    if model is None:
        # heuristic fallback
        pd_default = float(min(0.99, max(0.01, 0.02 + 0.02 * int_rate_pct + 0.005 * (inp.dti / 10.0))))
    else:
        pd_default = float(model.predict_proba(X)[0, 1])
    reasons = ["int_rate", "dti"]
    return CreditOutput(pd_default=pd_default, reason_codes=reasons)


@app.post("/score/fraud", response_model=FraudOutput)
def score_fraud(inp: FraudInput):
    model = load_model("fraud_xgb.joblib")
    amt_z = inp.amt_z if inp.amt_z is not None else 0.0
    X = [[inp.time, inp.amount, amt_z]]
    if model is None:
        fraud_prob = float(min(0.99, max(0.01, 0.01 + 0.001 * max(inp.amount - 500, 0))))
    else:
        fraud_prob = float(model.predict_proba(X)[0, 1])
    action = "review" if fraud_prob >= 0.5 else "allow"
    return FraudOutput(fraud_prob=fraud_prob, action=action, explanations=["amount"])
