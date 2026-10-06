import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import valuation_api as va


def tiny_hdb(n=400, seed=0):
    rng = np.random.default_rng(seed)
    m = 2025 * 12 + rng.integers(0, 12, n)
    return pd.DataFrame({
        "transaction_date": pd.to_datetime([f"{x // 12}-{x % 12 + 1:02d}-01" for x in m]), "town": "QUEENSTOWN", "flat_type": "4 ROOM",
        "flat_model": rng.choice(["Model A", "Improved"], n), "floor_area_sqm": rng.uniform(85, 100, n), "storey_mid": rng.integers(2, 30, n).astype(float),
        "remaining_lease_months": rng.uniform(700, 900, n), "block": "1", "street_name": "X ST", "storey_range": "10 TO 12",
        "is_exact_duplicate": False, "lease_commence_date": 1990, "resale_price": rng.uniform(5e5, 9e5, n), "property_id": "1 X ST"})


def test_unknown_town_or_flat_type_is_refused():
    eng = va.ValuationEngine(tiny_hdb())
    assert eng.estimate_fair_value("ATLANTIS", "4 ROOM", 93, 11, 864, "Model A", "2026-10-06")["status"].startswith("unsupported")
    assert eng.estimate_fair_value("QUEENSTOWN", "1 ROOM", 93, 11, 864, "Model A", "2026-10-06")["status"].startswith("unsupported")


def test_valuation_uses_only_prior_months_and_baseline_is_the_anchor():
    hdb = tiny_hdb()
    eng = va.ValuationEngine(hdb)
    r = eng.estimate_fair_value("QUEENSTOWN", "4 ROOM", 93, 11, "72 years 00 months", "Model A", "2026-01-15")
    prior = hdb[(hdb.transaction_date >= "2025-10-01") & (hdb.transaction_date < "2026-01-01")]["resale_price"]
    assert r["status"] == "ok" and r["method"] == "market_median_baseline"
    assert abs(r["market_anchor"]["value"] - prior.median()) < 1e-6                 # months 2025-10..2025-12 only
    assert r["valuation_month"] == "2026-01"


def test_future_valuation_is_capped_with_a_warning():
    eng = va.ValuationEngine(tiny_hdb())
    r = eng.estimate_fair_value("QUEENSTOWN", "4 ROOM", 93, 11, 864, "Model A", "2030-01-01")
    assert r["warning"] and r["valuation_month"] == eng._label(eng.latest_complete_month + 1)


def test_input_parsing():
    assert va.parse_lease_months("72 years 03 months") == 867 and va.parse_lease_months("61 years") == 732
    assert va.parse_lease_months((72, 3)) == 867 and va.parse_storey("10 TO 12") == 11 and va.parse_storey(7) == 7
