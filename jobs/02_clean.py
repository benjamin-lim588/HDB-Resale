"""Task 2: raw Delta tables -> clean Delta tables, via run_pipeline.clean_stage. Usage: 02_clean.py --catalog workspace"""
import argparse
import sys
from pathlib import Path

# __file__ is undefined when Databricks exec()s a script task; argv[0] holds the script path
sys.path.insert(0, str(Path(globals().get("__file__") or sys.argv[0]).resolve().parents[1] / "src"))
import databricks_io as dio
from run_pipeline import clean_stage

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=dio.DEFAULT_CATALOG)
    catalog = ap.parse_args().catalog
    raw = {k: dio.read_table(catalog, "raw", t)
           for k, t in [("hdb", "hdb_resale"), ("population", "population_2020"), ("income", "household_income_2020")]}
    for name, df in clean_stage(raw).items():
        dio.write_table(df, catalog, "clean", name)
