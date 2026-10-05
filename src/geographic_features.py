"""Block-level geocoding + transit accessibility (OPTIONAL layer; core EDA never depends on it).

Design: geocode each unique property_id once via OneMap, cache to data/cache, compute distances once per
block with the haversine formula, then join back to transactions.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from paths import CACHE, MRT_RAW, BUS_RAW

GEOCODE_CACHE = CACHE / "onemap_geocode_cache.csv"
ONEMAP = "https://www.onemap.gov.sg/api/common/elastic/search?searchVal={q}&returnGeom=Y&getAddrDetails=Y&pageNum=1"


ABBREV = {"AVE": "AVENUE", "ST": "STREET", "RD": "ROAD", "DR": "DRIVE", "BT": "BUKIT", "CRES": "CRESCENT",
          "NTH": "NORTH", "STH": "SOUTH", "CTRL": "CENTRAL", "LOR": "LORONG", "CL": "CLOSE", "PL": "PLACE",
          "GDNS": "GARDENS", "TER": "TERRACE", "UPP": "UPPER", "TG": "TANJONG", "JLN": "JALAN", "KG": "KAMPONG",
          "HTS": "HEIGHTS", "PK": "PARK", "MKT": "MARKET", "CTR": "CENTRE", "LK": "LINK", "WAY": "WAY"}


def expand_street(street: str) -> str:
    """HDB abbreviates street names ('ANG MO KIO AVE 10'); OneMap stores full words."""
    return " ".join(ABBREV.get(w, w) for w in street.split())


def _geocode_one(pid):
    """Query OneMap for 'BLOCK STREET'. Accept a result only if its BLK_NO equals our block
    (SEARCHVAL is often a building name, so it cannot be used). Status records road agreement."""
    block, street = pid.split(" ", 1)
    street_full = expand_street(street)
    headers = {"Authorization": os.environ["ONEMAP_TOKEN"]} if os.environ.get("ONEMAP_TOKEN") else {}
    for attempt in range(4):
        try:
            req = urllib.request.Request(ONEMAP.format(q=urllib.parse.quote(f"{block} {street_full}")), headers=headers)
            with urllib.request.urlopen(req, timeout=20) as r:
                res = json.load(r).get("results", [])
            hits = [x for x in res if x.get("BLK_NO", "").upper() == block]
            for x in hits:  # prefer same road name
                if x.get("ROAD_NAME", "").upper() == street_full:
                    return pid, float(x["LATITUDE"]), float(x["LONGITUDE"]), "ok"
            if hits:
                return pid, float(hits[0]["LATITUDE"]), float(hits[0]["LONGITUDE"]), "ok_road_differs"
            return pid, np.nan, np.nan, "no_matching_result"
        except Exception:
            time.sleep(2 * (attempt + 1))
    return pid, np.nan, np.nan, "request_failed"


def geocode_properties(property_ids, max_new=None, workers=3) -> pd.DataFrame:
    """Geocode unique property_ids once. Resumable: results are appended to the cache every 200 blocks.
    'request_failed' rows are retried on the next run; 'no_matching_result' rows are not."""
    cols = ["property_id", "latitude", "longitude", "status"]
    cache = pd.read_csv(GEOCODE_CACHE) if GEOCODE_CACHE.exists() else pd.DataFrame(columns=cols)
    done = set(cache.loc[cache["status"] != "request_failed", "property_id"])  # failures are retried
    todo = [p for p in property_ids if p not in done][:max_new]
    cache = cache[cache["property_id"].isin(done)]
    new = []
    with ThreadPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(_geocode_one, todo), 1):
            new.append(row)
            if i % 200 == 0 or i == len(todo):
                cache = pd.concat([cache, pd.DataFrame(new, columns=cols)], ignore_index=True)
                cache.to_csv(GEOCODE_CACHE, index=False)
                new = []
                print(f"geocoded {i}/{len(todo)}", flush=True)
    return cache


def load_mrt_stations(path=MRT_RAW) -> pd.DataFrame:
    """LTA MRT/LRT *exits* (GeoJSON points). Kept at exit level; `accessibility` collapses to stations."""
    g = json.load(open(path))
    return pd.DataFrame({
        "station_name": [f["properties"]["STATION_NA"].title() for f in g["features"]],
        "longitude": [f["geometry"]["coordinates"][0] for f in g["features"]],
        "latitude": [f["geometry"]["coordinates"][1] for f in g["features"]]})


def load_bus_stops(path=BUS_RAW) -> pd.DataFrame:
    """LTA bus stops (GeoJSON points); only the stop number is available, no name."""
    g = json.load(open(path))
    return pd.DataFrame({
        "bus_stop_name": [f["properties"]["BUS_STOP_NUM"] for f in g["features"]],
        "longitude": [f["geometry"]["coordinates"][0] for f in g["features"]],
        "latitude": [f["geometry"]["coordinates"][1] for f in g["features"]]})


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres; broadcasts over numpy arrays."""
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def accessibility(props: pd.DataFrame, poi: pd.DataFrame, name_col, prefix, radii) -> pd.DataFrame:
    """props: property_id, latitude, longitude. poi: name_col, latitude, longitude.
    Brute-force distance matrix (~9k blocks x ~5k stops) is fine, done once per block."""
    p = props.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    d = haversine_m(p["latitude"].values[:, None], p["longitude"].values[:, None],
                    poi["latitude"].values[None, :], poi["longitude"].values[None, :])
    out = pd.DataFrame({"property_id": p["property_id"]})
    out[f"nearest_{prefix}"] = poi[name_col].values[d.argmin(axis=1)]
    out[f"distance_to_nearest_{prefix}_m"] = d.min(axis=1)
    # count distinct named points (several MRT exits belong to one station)
    codes = pd.factorize(poi[name_col])[0]
    for r in radii:
        within = d <= r
        out[f"{prefix}_within_{r}m"] = [len(np.unique(codes[row])) for row in within]
    return out


def build_accessibility(props: pd.DataFrame):
    """Returns None when transit files are absent so callers can skip gracefully."""
    parts = []
    if MRT_RAW.exists():
        mrt = load_mrt_stations()
        parts.append(accessibility(props, mrt, "station_name", "mrt", (500, 1000)))
    if BUS_RAW.exists():
        bus = load_bus_stops()
        parts.append(accessibility(props, bus, "bus_stop_name", "bus_stop", (300, 500)))
    if not parts:
        return None
    out = parts[0]
    for q in parts[1:]:
        out = out.merge(q, on="property_id", how="outer")
    return out
