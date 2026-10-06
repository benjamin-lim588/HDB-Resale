"""Task 7: fair-value estimator. Baseline vs hedonic Ridge vs Random Forest vs XGBoost, chronological CV + untouched
holdout (scored once), MLflow tracking, champion registration (only if the champion is not the baseline).
Usage: 07_fair_value.py --catalog workspace --experiment /Users/<you>/flatfair-fair-value"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# __file__ is undefined when Databricks exec()s a script task; argv[0] holds the script path
sys.path.insert(0, str(Path(globals().get("__file__") or sys.argv[0]).resolve().parents[1] / "src"))
import databricks_io as dio
import fair_value_data as fv

import fair_value_models as fm
import fair_value_mlflow as fmf
import run_fair_value as rfv

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=dio.DEFAULT_CATALOG)
    ap.add_argument("--experiment", required=True)
    a = ap.parse_args()
    hdb = dio.read_table(a.catalog, "features", "hdb_transactions")
    acc = dio.read_table(a.catalog, "features", "property_accessibility")
    d, idx, info = fv.build_dataset(hdb, acc)
    audit = fv.audit_market_features(fv.prepare_transactions(hdb), d)
    print(info, "| leakage audit:", audit)
    res = rfv.run_all(d, with_test=True)
    print("DECISION:", res["decision"])
    tables = rfv.build_tables(res, d, idx)
    tables["fair_value_hedonic_effects"] = fm.hedonic_effects(res["test_models"]["hedonic_ridge"])
    for name, df in tables.items():       # persist first, so a logging problem never forces a second look at the holdout
        dio.write_table(df, a.catalog, "features", name)
    print("MLflow parent run:", fmf.log_experiment(a.experiment, a.catalog, res, d, tables, audit))
