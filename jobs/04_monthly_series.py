"""Task 4: features.hdb_transactions -> features.hdb_monthly_series + features.series_coverage.
All series are kept; `eligible_for_forecast` marks the modelling cohort. Usage: 04_monthly_series.py --catalog workspace"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# __file__ is undefined when Databricks exec()s a script task; argv[0] holds the script path
sys.path.insert(0, str(Path(globals().get("__file__") or sys.argv[0]).resolve().parents[1] / "src"))
import databricks_io as dio
import forecast_data as fd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=dio.DEFAULT_CATALOG)
    catalog = ap.parse_args().catalog
    hdb = dio.read_table(catalog, "features", "hdb_transactions")
    months = fd.complete_months(hdb)  # drops a trailing partial month
    monthly = fd.monthly_series(hdb, months)
    coverage = fd.series_coverage(monthly, months)
    monthly = monthly.merge(coverage[fd.KEYS + ["eligible_for_forecast"]], on=fd.KEYS)
    dio.write_table(monthly, catalog, "features", "hdb_monthly_series")
    dio.write_table(coverage, catalog, "features", "series_coverage")
    print(f"{len(months)} complete months, last {months[-1].date()}; eligible series {int(coverage['eligible_for_forecast'].sum())}/{len(coverage)}")
