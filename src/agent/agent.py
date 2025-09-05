from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.agent import tools


@dataclass
class Step:
    action: str
    params: dict
    result: tools.ToolResult | None = None


def plan(goal: str) -> List[Step]:
    g = goal.lower()
    steps: List[Step] = []
    # Simple keyword routing; extend as needed.
    if "download" in g:
        srcs = []
        if "lending" in g:
            srcs.append("lendingclub")
        if "kaggle" in g or "credit card" in g:
            srcs.append("kaggle_cc")
        if not srcs:
            srcs = ["lendingclub", "kaggle_cc"]
        steps.append(Step("download", {"sources": srcs}))
    if "etl" in g:
        engine = "spark" if "spark" in g else ("dask" if "dask" in g else "pandas")
        if "kaggle" in g:
            steps.append(Step("etl", {"source": "kaggle_cc", "engine": engine}))
        else:
            steps.append(Step("etl", {"source": "lendingclub", "engine": engine}))
    if "feature" in g:
        if "fraud" in g:
            steps.append(Step("features", {"domain": "fraud"}))
        else:
            steps.append(Step("features", {"domain": "credit"}))
    if "train" in g:
        cal = "calibrat" in g
        if "fraud" in g:
            steps.append(Step("train", {"domain": "fraud", "calibrate": cal}))
        else:
            steps.append(Step("train", {"domain": "credit", "calibrate": cal}))
    if "evaluate" in g or "report" in g:
        if "fraud" in g:
            steps.append(Step("evaluate", {"domain": "fraud"}))
        else:
            steps.append(Step("evaluate", {"domain": "credit"}))
    if "monitor" in g or "drift" in g:
        steps.append(Step("monitor", {"domain": "credit"}))
        steps.append(Step("monitor", {"domain": "fraud"}))
    if "batch" in g or "score" in g:
        if "fraud" in g:
            steps.append(Step("batch_score", {"domain": "fraud"}))
        else:
            steps.append(Step("batch_score", {"domain": "credit"}))
    if not steps:
        # Default: features -> train -> evaluate (credit)
        steps = [
            Step("features", {"domain": "credit"}),
            Step("train", {"domain": "credit", "calibrate": True}),
            Step("evaluate", {"domain": "credit"}),
        ]
    return steps


def execute(steps: List[Step], dry_run: bool = True) -> List[Step]:
    for s in steps:
        if dry_run:
            s.result = tools.ToolResult(True, f"DRY RUN: {s.action} {s.params}")
            continue
        if s.action == "download":
            s.result = tools.tool_download(**s.params)
        elif s.action == "etl":
            s.result = tools.tool_etl(**s.params)
        elif s.action == "features":
            s.result = tools.tool_features(**s.params)
        elif s.action == "train":
            s.result = tools.tool_train(**s.params)
        elif s.action == "evaluate":
            s.result = tools.tool_evaluate(**s.params)
        elif s.action == "monitor":
            s.result = tools.tool_monitor(**s.params)
        elif s.action == "batch_score":
            s.result = tools.tool_batch_score(**s.params)
        else:
            s.result = tools.ToolResult(False, f"Unknown action: {s.action}")
    return steps

