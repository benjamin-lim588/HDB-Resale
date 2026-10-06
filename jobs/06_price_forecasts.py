"""Task 6: production forecast (pooled 3-month median) -> features.price_forecasts. No prediction intervals.
Usage: 06_price_forecasts.py --catalog workspace"""
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
    monthly = dio.read_table(catalog, "features", "hdb_monthly_series")
    coverage = dio.read_table(catalog, "features", "series_coverage")
    panel = fd.build_panel(monthly, coverage, sorted(monthly["month"].unique()))
    dio.write_table(fd.production_forecast(panel), catalog, "features", "price_forecasts")
