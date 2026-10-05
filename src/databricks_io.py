"""Thin Unity Catalog <-> pandas bridge for Databricks job tasks.

Pattern: Delta table -> toPandas() -> existing pandas function -> createDataFrame() -> Delta table.
Fine for this dataset (~240k rows). Spark/Databricks Connect is imported lazily so the local
CSV pipeline never needs it.
"""
import os

import pandas as pd

DEFAULT_CATALOG = "workspace"
LANDING_VOLUME = "raw.landing"  # <schema>.<volume>


def get_spark():
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()
    from databricks.connect import DatabricksSession  # local VS Code via Databricks Connect
    return DatabricksSession.builder.getOrCreate()


def volume_path(catalog: str, filename: str) -> str:
    schema, volume = LANDING_VOLUME.split(".")
    return f"/Volumes/{catalog}/{schema}/{volume}/{filename}"


def fqn(catalog: str, schema: str, table: str) -> str:
    return f"`{catalog}`.`{schema}`.`{table}`"


def read_table(catalog: str, schema: str, table: str) -> pd.DataFrame:
    """Read a Delta table into pandas; ingestion metadata columns (leading '_') are dropped."""
    pdf = get_spark().table(fqn(catalog, schema, table)).toPandas()
    return pdf.drop(columns=[c for c in pdf.columns if c.startswith("_")])


def write_table(pdf: pd.DataFrame, catalog: str, schema: str, table: str) -> None:
    """Overwrite a managed Delta table. Stages are idempotent full refreshes."""
    from pyspark.sql import functions as F
    pdf = pdf.copy()
    for c in pdf.columns:
        if isinstance(pdf[c].dtype, pd.CategoricalDtype):
            pdf[c] = pdf[c].astype(object)
    sdf = get_spark().createDataFrame(pdf)
    for name, dtype in sdf.dtypes:  # pandas NaN -> SQL NULL so AVG/COUNT behave in SQL
        if dtype == "double":
            sdf = sdf.withColumn(name, F.when(F.isnan(name), None).otherwise(F.col(name)))
    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fqn(catalog, schema, table))
    print(f"wrote {catalog}.{schema}.{table}: {len(pdf):,} rows")
