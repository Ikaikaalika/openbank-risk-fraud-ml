from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import os

from fastapi import FastAPI
from pydantic import BaseModel

from src.agent.agent import plan as make_plan, execute

app = FastAPI(title="Varo Ops Agent")


class ExecuteRequest(BaseModel):
    goal: str
    dry_run: bool = True
    use_llm: bool = False
    model: str | None = None


class StepResult(BaseModel):
    action: str
    params: Dict[str, Any]
    ok: bool
    message: str


class ExecuteResponse(BaseModel):
    goal: str
    dry_run: bool
    steps: List[StepResult]


@app.post("/agent/execute", response_model=ExecuteResponse)
def agent_execute(req: ExecuteRequest):
    steps = make_plan(req.goal, use_llm=req.use_llm, model=req.model)
    steps = execute(steps, dry_run=req.dry_run)
    out_steps: List[StepResult] = []
    for s in steps:
        out_steps.append(
            StepResult(action=s.action, params=s.params, ok=bool(s.result and s.result.ok), message=(s.result.message if s.result else "no result"))
        )
    resp = ExecuteResponse(goal=req.goal, dry_run=req.dry_run, steps=out_steps)
    # Append to agent log (JSONL)
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "goal": req.goal,
            "dry_run": req.dry_run,
            "use_llm": req.use_llm,
            "model": req.model,
            "steps": [
                {
                    "action": s.action,
                    "params": s.params,
                    "ok": bool(s.result and s.result.ok),
                    "message": (s.result.message if s.result else "no result"),
                }
                for s in steps
            ],
        }
        with (log_dir / "agent.log").open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
    return resp
