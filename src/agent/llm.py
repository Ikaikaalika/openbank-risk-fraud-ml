from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests


SYSTEM_PROMPT = (
    "You are an Ops Planner for an ML repo. Plan a short sequence of steps "
    "to achieve the user goal using ONLY these actions: download, etl, features, train, "
    "evaluate, monitor, batch_score. Output STRICT JSON: {\"steps\":[{\"action\":<name>,\"params\":{...}}, ...]}. "
    "Do not add explanations. Param rules: \n"
    "- download: sources (list[str]), lc_files? (str), kcc_files? (str)\n"
    "- etl: source (\"lendingclub\"|\"kaggle_cc\"), engine (\"pandas\"|\"spark\"|\"dask\")\n"
    "- features: domain (\"credit\"|\"fraud\")\n"
    "- train: domain (\"credit\"|\"fraud\"), calibrate (bool), register (bool)\n"
    "- evaluate: domain (\"credit\"|\"fraud\")\n"
    "- monitor: domain (\"credit\"|\"fraud\")\n"
    "- batch_score: domain (\"credit\"|\"fraud\")\n"
)


def plan_with_ollama(goal: str, model: Optional[str] = None, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
    model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
    base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    prompt = f"Goal: {goal}\nReturn JSON now."
    body = {"model": model, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], "stream": False}
    try:
        resp = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        # Attempt to find JSON in content
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            obj = json.loads(content[start : end + 1])
            steps = obj.get("steps", [])
            # Basic sanitization
            cleaned: List[Dict[str, Any]] = []
            for s in steps:
                a = str(s.get("action", "")).strip()
                p = s.get("params", {}) or {}
                if a in {"download","etl","features","train","evaluate","monitor","batch_score"}:
                    cleaned.append({"action": a, "params": p})
            return cleaned
    except Exception:
        pass
    return []

