from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import typer
import pandas as pd

try:
    from great_expectations.dataset import PandasDataset
except Exception:  # pragma: no cover
    PandasDataset = None

app = typer.Typer()


def run_suite(df: pd.DataFrame, suite: dict) -> dict:
    if PandasDataset is None:
        raise RuntimeError("great_expectations PandasDataset not available")
    ds = PandasDataset(df)
    results = []
    success = True
    for exp in suite.get("expectations", []):
        etype = exp["expectation_type"]
        kwargs = exp.get("kwargs", {})
        func = getattr(ds, etype)
        res = func(**kwargs)
        results.append({"type": etype, "kwargs": kwargs, "success": bool(res.success)})
        success = success and bool(res.success)
    return {"suite_name": suite.get("suite_name"), "success": success, "results": results}


@app.command()
def main(
    suite: str = typer.Option("lendingclub_clean", help="Suite name: lendingclub_clean|kaggle_cc_clean"),
    data_glob: str = typer.Option("data/interim/lendingclub/**/*.parquet", help="Parquet glob to validate"),
    out_dir: str = typer.Option("validation/results", help="Output directory"),
    sample_rows: int = typer.Option(100000, help="Max rows to validate (sample if larger)"),
):
    suite_path = Path("validation/suites") / f"{suite}.json"
    if not suite_path.exists():
        raise typer.BadParameter(f"Suite not found: {suite_path}")
    data_files = list(Path().glob(data_glob))
    if not data_files:
        raise typer.BadParameter(f"No data files for glob: {data_glob}")
    # Load a sample of up to sample_rows
    dfs = []
    rows = 0
    for p in data_files:
        df = pd.read_parquet(p)
        dfs.append(df)
        rows += len(df)
        if rows >= sample_rows:
            break
    df_all = pd.concat(dfs, ignore_index=True)
    if len(df_all) > sample_rows:
        df_all = df_all.sample(n=sample_rows, random_state=42)
    suite_obj = json.loads(suite_path.read_text())
    result = run_suite(df_all, suite_obj)
    out_root = Path(out_dir) / suite
    out_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = out_root / f"validation_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2))
    status = "PASSED" if result["success"] else "FAILED"
    typer.echo(f"Validation {status}. Results: {out_path}")


if __name__ == "__main__":
    app()

