from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.common.io import ensure_dir


BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"


def _headers(token: str) -> dict:
    return {"token": token}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _get(url: str, headers: dict, params: dict) -> requests.Response:
    r = requests.get(url, headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r


def fetch_noaa_daily(
    station_id: str,
    start: date,
    end: date,
    dest_dir: Path,
    token: Optional[str] = None,
    dataset_id: str = "GHCND",
) -> Path:
    token = token or os.getenv("NOAA_TOKEN")
    if not token:
        raise RuntimeError("NOAA_TOKEN not provided. Set env or pass token.")
    ensure_dir(dest_dir)
    out = dest_dir / f"{station_id}_{start.isoformat()}_{end.isoformat()}.jsonl"
    # Paginate
    limit = 1000
    offset = 1
    with out.open("w") as f:
        while True:
            params = {
                "datasetid": dataset_id,
                "stationid": station_id,
                "startdate": start.isoformat(),
                "enddate": end.isoformat(),
                "units": "standard",
                "limit": limit,
                "offset": offset,
            }
            resp = _get(f"{BASE}/data", headers=_headers(token), params=params)
            js = resp.json()
            results = js.get("results", [])
            if not results:
                break
            for row in results:
                f.write(f"{row}\n")
            if len(results) < limit:
                break
            offset += limit
    return out

