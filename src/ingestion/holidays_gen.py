from __future__ import annotations

from pathlib import Path
from typing import Iterable

import holidays
import pandas as pd

from src.common.io import ensure_dir


def generate_holidays(country: str, years: Iterable[int], out_dir: Path) -> Path:
    ensure_dir(out_dir)
    hol = holidays.country_holidays(country=country, years=list(years))
    rows = [(d.isoformat(), name) for d, name in hol.items()]
    df = pd.DataFrame(rows, columns=["date", "holiday"]).sort_values("date")
    out = out_dir / f"holidays_{country}_{min(years)}_{max(years)}.csv"
    df.to_csv(out, index=False)
    return out

