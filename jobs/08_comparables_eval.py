"""Task 8: comparables engine historical evaluation + selection of ONE canonical methodology (CV only), holdout scored
once. Writes comparables_evaluation / comparables_config / comparables_quality_calibration and tracks the experiments in MLflow.
Usage: 08_comparables_eval.py --catalog workspace --experiment /Users/<you>/flatfair-comparables"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# __file__ is undefined when Databricks exec()s a script task; argv[0] holds the script path
sys.path.insert(0, str(Path(globals().get("__file__") or sys.argv[0]).resolve().parents[1] / "src"))
import databricks_io as dio
import fair_value_data as fv

import comparables as cp
import run_comparables_evaluation as rce

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=dio.DEFAULT_CATALOG)
    ap.add_argument("--experiment", required=True)
    a = ap.parse_args()
    hdb = dio.read_table(a.catalog, "features", "hdb_transactions")
    acc = dio.read_table(a.catalog, "features", "property_accessibility")
    d, idx, info = fv.build_dataset(hdb, acc)
    pool = cp.Pool(fv.prepare_transactions(hdb), acc)
    r = rce.run(d, pool, with_test=True)
    print("CHOICE:", r["choice"], r["quality"])
    tables, cfg = rce.build_tables(r, d, pool)
    for name, df in tables.items():
        dio.write_table(df, a.catalog, "features", name)
    print("MLflow parent run:", rce.log_mlflow(a.experiment, r, tables, cfg))
