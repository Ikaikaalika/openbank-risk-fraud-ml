
import typer
from pyspark.sql import SparkSession

app = typer.Typer()

@app.command()
def main(input: str = "data/raw/lendingclub.csv", out: str = "data/interim/lendingclub_spark"):
    spark = SparkSession.builder.appName("ETL").getOrCreate()
    df = spark.read.option("header", True).csv(input, inferSchema=True)
    df.write.mode("overwrite").parquet(out)
    spark.stop()
    print(f"Saved to {out}")

if __name__ == "__main__":
    app()
