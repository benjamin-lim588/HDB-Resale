"""Central project paths (relative to the repo root, no absolute paths)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"

HDB_RAW = RAW / "Resaleflat_prices_based_on_registration_datefrom_Jan2017_onwards.csv"
POP_RAW = RAW / "Total_Population_by_Planning_Area_2020.csv"
# Optional layers -- loaders tolerate these being absent.
INCOME_RAW = RAW / "ResidentHouseholdsbyPlanningAreaofResidenceandMonthlyHouseholdIncomefromWorkCensusofPopulation2020.csv"
MRT_RAW = RAW / "LTAMRTStationExitGEOJSON.geojson"
BUS_RAW = RAW / "LTABusStop.geojson"

# Local convenience only: Databricks jobs read/write Unity Catalog, not repo folders.
if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    for _p in (PROCESSED, CACHE, FIGURES, TABLES):
        _p.mkdir(parents=True, exist_ok=True)
