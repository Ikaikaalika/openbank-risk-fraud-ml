from __future__ import annotations

import re
from pathlib import Path
from typing import List

import typer
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

app = typer.Typer()


def parse_lendingclub_columns(df: DataFrame) -> DataFrame:
    # issue_d to timestamp
    if "issue_d" in df.columns:
        df = df.withColumn("issue_d", F.to_timestamp("issue_d"))
    # addr_state to state
    if "addr_state" in df.columns and "state" not in df.columns:
        df = df.withColumn("state", F.col("addr_state").cast("string"))
    # int_rate like "13.56%" -> double percent
    if "int_rate" in df.columns:
        df = df.withColumn("int_rate", F.regexp_replace("int_rate", "%", "").cast("double"))
    # term like " 36 months" -> integer
    if "term" in df.columns:
        df = df.withColumn("term", F.regexp_extract("term", r"(\\d+)", 1).cast("int")).fillna(36, subset=["term"])
    # defaulted from loan_status if exists
    if "defaulted" not in df.columns and "loan_status" in df.columns:
        bad = F.regexp_extract(F.lower(F.col("loan_status")), r"(charged off|default|late \(31-120 days\))", 1) != ""
        df = df.withColumn("defaulted", bad.cast("int"))
    # essential columns only
    keep = [c for c in ["issue_d", "loan_amnt", "int_rate", "dti", "term", "state", "defaulted"] if c in df.columns]
    df = df.select(*keep)
    # partitions
    df = df.withColumn("year", F.year("issue_d")).withColumn("month", F.month("issue_d"))
    return df


@app.command()
def main(
    input: str = typer.Option("data/raw/lendingclub/*.csv", help="Input CSV glob or path"),
    out: str = typer.Option("data/interim/lendingclub", help="Output root directory"),
):
    spark = SparkSession.builder.appName("ETL_LendingClub").getOrCreate()
    try:
        df = spark.read.option("header", True).csv(input, inferSchema=True)
        df = parse_lendingclub_columns(df).dropna(subset=["issue_d"])  # type: ignore[arg-type]
        (
            df.write.mode("overwrite")
            .partitionBy("year", "month")
            .parquet(out)
        )
        print(f"Wrote Spark ETL to {out}")
    finally:
        spark.stop()


if __name__ == "__main__":
    app()

