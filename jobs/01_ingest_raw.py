"""Task 1: landing volume files -> raw Delta tables (no cleaning). Usage: 01_ingest_raw.py --catalog workspace"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# __file__ is undefined when Databricks exec()s a script task; argv[0] holds the script path
sys.path.insert(0, str(Path(globals().get("__file__") or sys.argv[0]).resolve().parents[1] / "src"))
import databricks_io as dio
import geographic_features as gf

# raw table -> (source file in the landing volume, reader)
SOURCES = {
    "hdb_resale": ("Resaleflat_prices_based_on_registration_datefrom_Jan2017_onwards.csv", pd.read_csv),
    "population_2020": ("Total_Population_by_Planning_Area_2020.csv", pd.read_csv),
    "household_income_2020": ("ResidentHouseholdsbyPlanningAreaofResidenceandMonthlyHouseholdIncomefromWorkCensusofPopulation2020.csv", pd.read_csv),
    "policy_events": ("policy_events.csv", pd.read_csv),
    "geocode_cache": ("onemap_geocode_cache.csv", pd.read_csv),  # pre-computed OneMap result; never re-queried here
    "mrt_exits": ("LTAMRTStationExitGEOJSON.geojson", gf.load_mrt_stations),
    "bus_stops": ("LTABusStop.geojson", gf.load_bus_stops),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=dio.DEFAULT_CATALOG)
    catalog = ap.parse_args().catalog
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for table, (fname, reader) in SOURCES.items():
        df = reader(dio.volume_path(catalog, fname))
        df["_source_file"], df["_ingested_at"] = fname, now
        dio.write_table(df, catalog, "raw", table)
