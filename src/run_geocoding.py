"""Geocode every unique HDB block once (cached, resumable). Usage: python src/run_geocoding.py [max_new]
Optional: export ONEMAP_TOKEN=... if OneMap starts requiring authentication."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geographic_features import geocode_properties
from paths import PROCESSED

props = pd.read_csv(PROCESSED / "hdb_property_table.csv")
cache = geocode_properties(props["property_id"].tolist(), max_new=int(sys.argv[1]) if len(sys.argv) > 1 else None)
print(cache["status"].value_counts().to_dict())
