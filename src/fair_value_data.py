"""Transaction-level dataset for the FAIR-VALUE estimator (what is THIS flat worth at a valuation date?).

Separate from the market forecasting code. Target: resale_price, modelled as log(resale_price / anchor).
Convention (conservative, month-level): a flat valued in month V may only use sales from months STRICTLY BEFORE V,
because the source has no sale day. `anchor` = pooled median of the town x flat_type sales in V-3..V-1, widened to
6 then 12 months, then to the island-wide flat_type median, when fewer than MIN_ANCHOR_N sales are available.
`_pooled` only ever reads months V-w..V-1, so a market feature cannot see month V or later by construction;
`audit_market_features` proves it by recomputing features from data truncated before V.
"""
import numpy as np
import pandas as pd

from forecast_data import complete_months  # read-only reuse of the partial-month rule

MIN_ANCHOR_N = 10
KEYS = ["town", "flat_type"]
ACCESS_COLS = ["distance_to_nearest_mrt_m", "mrt_within_500m", "mrt_within_1000m",
               "distance_to_nearest_bus_stop_m", "bus_stop_within_300m", "bus_stop_within_500m"]

CORE_CAT = ["town", "flat_type", "flat_model"]
CORE_NUM = ["log_area", "storey_mid", "storey_sq", "lease_years", "lease_years_sq",
            "log_anchor", "anchor_change_3m", "prior6_vs_anchor", "log_prior3_volume", "anchor_tier",
            "month_sin", "month_cos"]
YEAR_NUM = ["valuation_year"]                                   # ablation only
ACCESS_NUM = ["log_mrt_dist", "mrt_within_500m", "mrt_within_1000m", "log_bus_dist",
              "bus_stop_within_300m", "bus_stop_within_500m", "acc_missing"]  # ablation / enrichment only
FORBIDDEN = {"resale_price", "price_per_sqm", "y_rel", "txn_id", "any_flag", "is_exact_duplicate", "policy_events_to_date",
             "months_since_start_of_dataset", "flag_lease_inconsistent", "flag_extreme_price_per_sqm",
             "flag_large_area_for_type"}
FEATURE_SETS = {"core": (CORE_CAT, CORE_NUM), "core_year": (CORE_CAT, CORE_NUM + YEAR_NUM),
                "core_access": (CORE_CAT, CORE_NUM + ACCESS_NUM)}


def month_index(ts: pd.Series) -> pd.Series:
    return ts.dt.year * 12 + ts.dt.month - 1


# ---------------------------------------------------------------- market features (shared by training + serving)
def price_arrays(df: pd.DataFrame):
    """({(town, flat_type): {m: prices}}, {flat_type: {m: prices}}) for fast windowed medians."""
    cells, island = {}, {}
    for (t, f, m), g in df.groupby(["town", "flat_type", "m"], sort=False)["resale_price"]:
        cells.setdefault((t, f), {})[m] = g.to_numpy(float)
    for (f, m), g in df.groupby(["flat_type", "m"], sort=False)["resale_price"]:
        island.setdefault(f, {})[m] = g.to_numpy(float)
    return cells, island


def _pooled(arrs: dict, v: int, w: int) -> np.ndarray:
    """All prices in months v-w .. v-1 (NEVER month v or later)."""
    parts = [arrs[m] for m in range(v - w, v) if m in arrs]
    return np.concatenate(parts) if parts else np.empty(0)


def cell_anchor(cell: dict, island: dict, v: int):
    """(anchor, n, tier): tier 0 = 3m cell, 1 = 6m cell, 2 = 12m cell, 3 = island-wide flat_type 3m."""
    for tier, w in enumerate((3, 6, 12)):
        a = _pooled(cell, v, w)
        if len(a) >= MIN_ANCHOR_N:
            return float(np.median(a)), len(a), tier
    a = _pooled(island, v, 3)
    return (float(np.median(a)), len(a), 3) if len(a) >= MIN_ANCHOR_N else (np.nan, len(a), 3)


def market_features(cell: dict, island: dict, v: int) -> dict:
    anchor, n, tier = cell_anchor(cell, island, v)
    a3, _, _ = cell_anchor(cell, island, v - 3)
    p6 = _pooled(cell, v, 6)
    return {"anchor": anchor, "anchor_n": n, "anchor_tier": tier,
            "anchor_change_3m": np.log(anchor / a3) if anchor == anchor and a3 == a3 else np.nan,
            "prior6_vs_anchor": np.log(np.median(p6) / anchor) if len(p6) >= MIN_ANCHOR_N and anchor == anchor else 0.0,
            "log_prior3_volume": float(np.log1p(len(_pooled(cell, v, 3))))}


class MarketIndex:
    """Windowed price lookup built once from transactions; used for training features AND live valuation."""

    def __init__(self, df: pd.DataFrame):
        self.cells, self.island = price_arrays(df)

    def features(self, town: str, flat_type: str, v: int) -> dict:
        return market_features(self.cells.get((town, flat_type), {}), self.island.get(flat_type, {}), v)

    def table(self, months: range) -> pd.DataFrame:
        rows = [dict(town=t, flat_type=f, m=m, **market_features(c, self.island.get(f, {}), m))
                for (t, f), c in self.cells.items() for m in months]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------- dataset
def prepare_transactions(hdb: pd.DataFrame, months=None) -> pd.DataFrame:
    """Deterministic order + txn_id; keeps complete months only (a trailing partial month is dropped)."""
    df = hdb.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    months = months if months is not None else complete_months(df)
    df = df[df["transaction_date"].isin(months)]
    order = ["transaction_date", "town", "flat_type", "block", "street_name", "storey_range", "floor_area_sqm",
             "resale_price", "flat_model", "lease_commence_date"]
    df = df.sort_values(order, kind="stable").reset_index(drop=True)
    df["txn_id"] = np.arange(len(df))
    df["m"] = month_index(df["transaction_date"])
    return df


def add_row_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["log_area"] = np.log(d["floor_area_sqm"])
    d["storey_sq"] = d["storey_mid"] ** 2
    d["lease_years"] = d["remaining_lease_months"] / 12
    d["lease_years_sq"] = d["lease_years"] ** 2
    d["log_anchor"] = np.log(d["anchor"])
    mo = (d["m"] % 12) / 12 * 2 * np.pi
    d["month_sin"], d["month_cos"] = np.sin(mo), np.cos(mo)
    d["valuation_year"] = d["m"] // 12
    return d


def add_access_features(d: pd.DataFrame, acc: pd.DataFrame | None) -> pd.DataFrame:
    d = d.drop(columns=[c for c in ACCESS_COLS if c in d.columns])
    if acc is not None:
        d = d.merge(acc[["property_id"] + ACCESS_COLS].drop_duplicates("property_id"), on="property_id", how="left")
    else:
        for c in ACCESS_COLS:
            d[c] = np.nan
    d["acc_missing"] = d["distance_to_nearest_mrt_m"].isna().astype(int)
    d["log_mrt_dist"] = np.log1p(d["distance_to_nearest_mrt_m"])
    d["log_bus_dist"] = np.log1p(d["distance_to_nearest_bus_stop_m"])
    return d


def build_dataset(hdb: pd.DataFrame, acc: pd.DataFrame | None = None):
    """Returns (dataset, MarketIndex, info). Rows without a usable anchor / 3-month anchor change (the first months)
    are dropped; `info` records how many."""
    df = prepare_transactions(hdb)
    idx = MarketIndex(df)
    mk = idx.table(range(int(df["m"].min()), int(df["m"].max()) + 1))
    d = df.merge(mk, on=KEYS + ["m"], how="left")
    n0 = len(d)
    d = d[d["anchor"].notna() & d["anchor_change_3m"].notna()].reset_index(drop=True)
    d = add_access_features(add_row_features(d), acc)
    d["y_rel"] = np.log(d["resale_price"] / d["anchor"])
    d["month"] = d["transaction_date"].dt.to_period("M").dt.to_timestamp()
    return d, idx, {"rows_total": n0, "rows_dropped_warmup": n0 - len(d), "first_month": d["month"].min(),
                    "last_month": d["month"].max()}


# ---------------------------------------------------------------- leakage audit
def audit_market_features(df_all: pd.DataFrame, d: pd.DataFrame, n_months: int = 30, per_month: int = 8, seed: int = 0):
    """For sampled rows, recompute market features from data TRUNCATED to months < V and require identical values.
    Also asserts no forbidden column is a predictor."""
    bad = {c for cs in FEATURE_SETS.values() for c in cs[0] + cs[1]} & FORBIDDEN
    assert not bad, f"forbidden predictors: {bad}"
    rng = np.random.default_rng(seed)
    cols = ["anchor", "anchor_n", "anchor_tier", "anchor_change_3m", "prior6_vs_anchor", "log_prior3_volume"]
    ms = rng.choice(np.sort(d["m"].unique())[6:], size=n_months, replace=False)
    checked = mism = 0
    for v in ms:
        trunc = MarketIndex(df_all[df_all["m"] < v])           # nothing from month v onward exists here
        rows = d[d["m"] == v].sample(per_month, random_state=int(v), replace=False)
        for r in rows.itertuples():
            f = trunc.features(r.town, r.flat_type, int(v))
            a = np.array([f[c] for c in cols], float)
            b = np.array([getattr(r, c) for c in cols], float)
            checked += 1
            mism += int(not np.allclose(a, b, equal_nan=True))
    assert mism == 0, f"{mism}/{checked} rows differ when data from the valuation month onward is removed -> leakage"
    return {"rows_checked": checked, "mismatches": mism, "forbidden_predictors": []}
