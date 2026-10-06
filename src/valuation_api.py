"""Thin product interface for FlatFair valuation. THREE separate answers, never collapsed into one number:

  estimate_fair_value(...)   what is THIS flat likely worth?                      (fair-value estimator)
  find_comparables(...)      which actual recent sales are most similar?          ("Comparable market reference")
  get_market_outlook(...)    where is the town x flat_type market heading?        (existing forecasting engine)

Valuation convention: a flat valued in month V uses only sales from months STRICTLY BEFORE V. The current product values
in month 2026-10 using the latest complete market month 2026-09. Data can come from CSV or Delta (via pandas).

    eng = ValuationEngine(hdb_transactions_df, accessibility_df, comparables_config=load_config())
    eng.estimate_fair_value("QUEENSTOWN", "4 ROOM", 93, 11, "72 years 00 months", "Model A", "2026-10-06")
    eng.find_comparables("QUEENSTOWN", "4 ROOM", 93, 11, 864, "Model A", "2026-10-06", k=5)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import comparables as cp
import fair_value_data as fv
import fair_value_models as fm

CONFIG_PATH = Path(__file__).resolve().parent / "comparables_config.json"


def load_config(path=CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


def parse_lease_months(x) -> float:
    """Accepts months (864), 'years' as 'NN years MM months' / 'NN years', or (years, months) tuple."""
    if isinstance(x, (tuple, list)):
        return float(x[0]) * 12 + float(x[1])
    if isinstance(x, str):
        y = re.search(r"(\d+)\s*year", x)
        m = re.search(r"(\d+)\s*month", x)
        if y or m:
            return float(y.group(1) if y else 0) * 12 + float(m.group(1) if m else 0)
        return float(x)
    return float(x)


def parse_storey(x) -> float:
    """Accepts a floor number or a range string like '10 TO 12' (midpoint)."""
    if isinstance(x, str) and "TO" in x.upper():
        lo, hi = (float(v) for v in re.findall(r"\d+", x))
        return (lo + hi) / 2
    return float(x)


def valuation_month_index(valuation_date) -> int:
    t = pd.Timestamp(valuation_date)
    return t.year * 12 + t.month - 1


class ValuationEngine:
    def __init__(self, hdb: pd.DataFrame, acc: pd.DataFrame | None = None, model=None, model_family: str | None = None,
                 model_version: str | None = None, comparables_config: dict | None = None, feature_set: str = "core"):
        self.df = fv.prepare_transactions(hdb)
        self.idx = fv.MarketIndex(self.df)
        self.pool = cp.Pool(self.df, acc)
        self.acc = acc.drop_duplicates("property_id").set_index("property_id") if acc is not None else None
        self.model, self.family, self.version, self.feature_set = model, model_family or fm.BASELINE, model_version, feature_set
        self.config = comparables_config
        self.latest_complete_month = int(self.df["m"].max())       # month index of the latest complete market month

    # -- helpers
    def _valuation_month(self, valuation_date):
        v = valuation_month_index(valuation_date)
        cap = self.latest_complete_month + 1
        warn = None
        if v > cap:
            warn = (f"valuation month is beyond the market data (latest complete month {self._label(self.latest_complete_month)}); "
                    f"valued as of {self._label(cap)}")
            v = cap
        return v, warn

    @staticmethod
    def _label(m: int) -> str:
        return f"{m // 12}-{m % 12 + 1:02d}"

    # -- 1. fair value
    def estimate_fair_value(self, town, flat_type, floor_area_sqm, storey, remaining_lease, flat_model, valuation_date,
                            accessibility_features: dict | None = None) -> dict:
        v, warn = self._valuation_month(valuation_date)
        if (town, flat_type) not in self.idx.cells:      # never value a town/flat_type with no sales history (no silent fallback)
            return dict(status="unsupported town / flat_type: no sales history", town=town, flat_type=flat_type,
                        valuation_month=self._label(v))
        mk = self.idx.features(town, flat_type, v)
        if not (mk["anchor"] == mk["anchor"]):
            return dict(status="insufficient market data", town=town, flat_type=flat_type, valuation_month=self._label(v))
        lease = parse_lease_months(remaining_lease)
        storey_mid = parse_storey(storey)
        row = pd.DataFrame([dict(town=town, flat_type=flat_type, flat_model=flat_model, floor_area_sqm=float(floor_area_sqm),
                                 storey_mid=storey_mid, remaining_lease_months=lease, m=v, **mk)])
        row = fv.add_row_features(row)
        acc = accessibility_features or {}
        for c in fv.ACCESS_COLS:
            row[c] = acc.get(c, np.nan)
        row = fv.add_access_features(row, None) if not acc else _with_access(row, acc)
        if self.model is None or self.family == fm.BASELINE:
            value, method = mk["anchor"], fm.BASELINE
        else:
            value, method = float(mk["anchor"] * np.exp(self.model.predict(fm.design(row, self.feature_set))[0])), self.family
        return dict(status="ok", estimated_fair_value=round(value, -2), method=method, feature_set=self.feature_set,
                    model_version=self.version, valuation_month=self._label(v),
                    latest_complete_market_month=self._label(self.latest_complete_month),
                    market_anchor=dict(value=mk["anchor"], n_sales=mk["anchor_n"], tier=mk["anchor_tier"],
                                       description="median of town x flat_type sales in the 3 prior complete months "
                                                   "(widened to 6/12 months or island-wide if thin)"),
                    warning=warn)

    # -- 2. comparables
    def find_comparables(self, town, flat_type, floor_area_sqm, storey, remaining_lease, flat_model, valuation_date,
                         k: int = 5, accessibility_features: dict | None = None) -> dict:
        if self.config is None:
            raise ValueError("comparables_config is required (see comparables_config.json)")
        v, warn = self._valuation_month(valuation_date)
        mrt = (accessibility_features or {}).get("distance_to_nearest_mrt_m")
        out = cp.find_comparables(self.pool, self.config, town, flat_type, floor_area_sqm, parse_storey(storey),
                                  parse_lease_months(remaining_lease), v, flat_model=flat_model, k=k, mrt_distance_m=mrt)
        out.update(valuation_month=self._label(v), latest_complete_market_month=self._label(self.latest_complete_month), warning=warn)
        return out

    def accessibility_for(self, block: str, street_name: str) -> dict | None:
        """Convenience: accessibility features of a known block (property_id = 'BLOCK STREET')."""
        if self.acc is None:
            return None
        pid = f"{block} {street_name}".upper().strip()
        return self.acc.loc[pid, fv.ACCESS_COLS].to_dict() if pid in self.acc.index else None


def _with_access(row: pd.DataFrame, acc: dict) -> pd.DataFrame:
    row = row.copy()
    row["acc_missing"] = int(acc.get("distance_to_nearest_mrt_m") is None)
    row["log_mrt_dist"] = np.log1p(acc.get("distance_to_nearest_mrt_m", np.nan))
    row["log_bus_dist"] = np.log1p(acc.get("distance_to_nearest_bus_stop_m", np.nan))
    return row


# -- 3. market outlook (existing forecasting engine; this is just a lookup of workspace.features.price_forecasts)
def get_market_outlook(price_forecasts: pd.DataFrame, town: str, flat_type: str) -> pd.DataFrame:
    """1/3/6-month market forecast for the town x flat_type (pooled 3-month median method). Empty if the series is not
    forecast ('insufficient data for reliable forecast')."""
    f = price_forecasts[(price_forecasts["town"] == town) & (price_forecasts["flat_type"] == flat_type)]
    return f[["forecast_month", "forecast_horizon_months", "predicted_median_price", "forecast_method"]].sort_values(
        "forecast_horizon_months").reset_index(drop=True)


def load_champion_model(catalog: str, family: str, alias: str = "champion"):
    """Load the registered fair-value champion from Unity Catalog (only exists if the champion is not the baseline)."""
    import mlflow
    mlflow.set_registry_uri("databricks-uc")
    uri = f"models:/{catalog}.models.flatfair_fair_value_{family}@{alias}"
    return mlflow.sklearn.load_model(uri)
