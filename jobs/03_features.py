"""Task 3: clean Delta tables -> feature / analytics Delta tables, via run_pipeline.features_stage.
Usage: 03_features.py --catalog workspace"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# __file__ is undefined when Databricks exec()s a script task; argv[0] holds the script path
sys.path.insert(0, str(Path(globals().get("__file__") or sys.argv[0]).resolve().parents[1] / "src"))
import databricks_io as dio
from run_pipeline import features_stage

CLEAN_INPUTS = ["hdb_resale", "population_planning_area", "household_income_wide", "town_planning_area_mapping"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=dio.DEFAULT_CATALOG)
    catalog = ap.parse_args().catalog
    clean = {t: dio.read_table(catalog, "clean", t) for t in CLEAN_INPUTS}
    events = dio.read_table(catalog, "raw", "policy_events")
    events["effective_date"] = pd.to_datetime(events["effective_date"])
    raw = lambda t: dio.read_table(catalog, "raw", t)
    for name, df in features_stage(clean, events, geocode=raw("geocode_cache"),
                                   mrt=raw("mrt_exits"), bus=raw("bus_stops")).items():
        dio.write_table(df, catalog, "features", name)
