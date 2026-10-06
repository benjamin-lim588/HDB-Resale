"""Properties the comparables engine must keep: no price in matching, no future sales, insufficient -> explicit message."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import comparables as cp

CFG = {"k": 5, "quality": {"k": 5, "high_min_candidates": 20, "high_max_dist": 0.02, "medium_max_dist": 0.04},
       "engine": cp.make_config("hedonic", "loose", 0.005, False, {"log_area": .1, "storey": 3, "lease_months": 100, "log_mrt": 1},
                                {"log_area": 1.0, "storey": 0.002, "storey_sq": 0.0, "lease_years": 0.004, "lease_years_sq": 0.0,
                                 "model_penalty": 0.03, "log_mrt": 0.0})}


def make_pool(n=60, seed=0, price_seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "txn_id": np.arange(n), "m": 2024 * 12 + rng.integers(0, 24, n), "town": "QUEENSTOWN", "flat_type": "4 ROOM",
        "flat_model": rng.choice(["Model A", "Improved"], n), "floor_area_sqm": rng.uniform(85, 100, n),
        "storey_mid": rng.integers(2, 30, n).astype(float), "remaining_lease_months": rng.uniform(700, 900, n),
        "block": "1", "street_name": "X ST", "storey_range": "10 TO 12", "is_exact_duplicate": False,
        "resale_price": np.random.default_rng(price_seed).uniform(5e5, 9e5, n)})
    return df


def query(pool, v, **kw):
    return cp.find_comparables(pool, CFG, "QUEENSTOWN", "4 ROOM", 93, 11, 864, v, flat_model="Model A", k=5, **kw)


def test_ranking_never_uses_price():
    a, b = cp.Pool(make_pool(price_seed=1)), cp.Pool(make_pool(price_seed=2))   # same flats, completely different prices
    v = 2025 * 12 + 6
    ra, rb = query(a, v), query(b, v)
    assert ra["status"] == "ok"
    assert list(ra["comparables"]["floor_area_sqm"]) == list(rb["comparables"]["floor_area_sqm"])
    assert list(ra["comparables"]["comparable_rank"]) == list(rb["comparables"]["comparable_rank"])
    assert ra["reference"]["median_price"] != rb["reference"]["median_price"]       # price only shows up AFTER selection


def test_only_prior_months_are_eligible():
    pool = cp.Pool(make_pool())
    v = 2024 * 12 + 12
    r = query(pool, v)
    assert (r["comparables"]["transaction_month"] < pd.Timestamp(year=v // 12, month=v % 12 + 1, day=1)).all()
    assert r["search_window_months"] in cp.WINDOWS


def test_insufficient_comparables_is_explicit():
    pool = cp.Pool(make_pool(n=2))
    r = query(pool, 2025 * 12 + 6)
    assert r["status"] == cp.INSUFFICIENT and r["reference"] is None and r["comparables"].empty


def test_exact_duplicates_are_not_double_counted():
    df = make_pool(n=40)
    d2 = pd.concat([df, df.assign(is_exact_duplicate=True)], ignore_index=True)
    assert len(cp.Pool(d2).df) == len(df)
