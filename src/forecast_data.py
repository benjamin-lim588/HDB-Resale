"""Monthly modelling tables and leakage-safe features for town x flat_type price forecasting.

Grain: month x town x flat_type. Target: median resale price `h` months after the forecast origin.
Convention: every feature is computed from data up to AND INCLUDING the origin month t; the target is
month t+h. `price_lag_k` therefore means the price k months before the origin (lag_0 = latest known month).
Pure pandas, no I/O, so the same code runs locally (CSV) and on Databricks (Delta via toPandas).
"""
import numpy as np
import pandas as pd

MIN_ACTIVE_MONTHS = 84      # months with >=1 transaction
MIN_MEDIAN_COUNT = 5        # median transactions per active month
RECENT_MONTHS = 24
KEYS = ["town", "flat_type"]

CATEGORICAL = ["town", "flat_type"]
NUMERIC_FEATURES = [
    "price_lag_0", "price_lag_1", "price_lag_2", "price_lag_5", "price_lag_11",
    "rolling_3m_price", "rolling_6m_price", "rolling_12m_price",
    "txn_count_lag_0", "rolling_3m_txn_count", "rolling_6m_txn_count",
    "mom_price_change", "quarterly_price_change", "yoy_price_change",
    "target_year", "target_month_number", "target_quarter",
]
# Columns that must never appear as predictors (leakage / not known at forecast time).
FORBIDDEN = {"price_per_sqm", "resale_price", "y", "y_rel", "y_count", "target_price", "median_resale_price"}


# ---------------------------------------------------------------- monthly table
def complete_months(hdb: pd.DataFrame) -> list:
    """Calendar months present in the data, minus trailing partial month(s): a trailing month is dropped
    when it has < 50% of the median transaction count of the 6 months before it."""
    n = hdb.groupby(hdb["transaction_date"].dt.to_period("M").dt.to_timestamp()).size().sort_index()
    while len(n) > 7 and n.iloc[-1] < 0.5 * n.iloc[-7:-1].median():
        n = n.iloc[:-1]
    return list(n.index)


def monthly_series(hdb: pd.DataFrame, months=None) -> pd.DataFrame:
    """One row per observed month x town x flat_type. `pooled_median_3m` is the median over all
    transactions of the series in the trailing 3 months (t-2..t): a less noisy level estimate than the
    single-month median. Uses only transactions dated <= the row's month."""
    months = months or complete_months(hdb)
    df = hdb[hdb["transaction_date"].isin(months)].copy()
    df["month"] = df["transaction_date"].dt.to_period("M").dt.to_timestamp()
    g = df.groupby(["month"] + KEYS)
    out = g.agg(median_resale_price=("resale_price", "median"), transaction_count=("resale_price", "size"),
                median_floor_area_sqm=("floor_area_sqm", "median"),
                median_remaining_lease_months=("remaining_lease_months", "median")).reset_index()
    idx = {m: i for i, m in enumerate(sorted(months))}
    out["_i"] = out["month"].map(idx)
    pooled = []
    for _, s in df.assign(_i=df["month"].map(idx)).groupby(KEYS):
        prices = {i: v.to_numpy() for i, v in s.groupby("_i")["resale_price"]}
        for i in sorted(prices):
            w = [prices[j] for j in (i - 2, i - 1, i) if j in prices]
            pooled.append((s["town"].iat[0], s["flat_type"].iat[0], i, float(np.median(np.concatenate(w)))))
    p = pd.DataFrame(pooled, columns=KEYS + ["_i", "pooled_median_3m"])
    out = out.merge(p, on=KEYS + ["_i"], how="left").drop(columns="_i")
    return out.sort_values(["town", "flat_type", "month"]).reset_index(drop=True)


def series_coverage(monthly: pd.DataFrame, months) -> pd.DataFrame:
    """Per series coverage + eligibility. Excluded series are KEPT (historical analytics only)."""
    recent = set(sorted(months)[-RECENT_MONTHS:])
    g = monthly.groupby(KEYS)
    cov = g.agg(active_months=("month", "nunique"), total_transactions=("transaction_count", "sum"),
                median_monthly_count=("transaction_count", "median"), first_month=("month", "min"),
                last_month=("month", "max")).reset_index()
    cov["recent_active_months"] = cov.merge(
        monthly[monthly["month"].isin(recent)].groupby(KEYS).size().rename("r").reset_index(),
        on=KEYS, how="left")["r"].fillna(0).astype(int).to_numpy()
    cov["eligible_for_forecast"] = (cov["active_months"] >= MIN_ACTIVE_MONTHS) & \
                                   (cov["median_monthly_count"] >= MIN_MEDIAN_COUNT)

    def why(r):
        f = []
        if r.active_months < MIN_ACTIVE_MONTHS:
            f.append(f"active_months {r.active_months} < {MIN_ACTIVE_MONTHS}")
        if r.median_monthly_count < MIN_MEDIAN_COUNT:
            f.append(f"median_monthly_count {r.median_monthly_count:g} < {MIN_MEDIAN_COUNT}")
        return "; ".join(f)

    cov["exclusion_reason"] = cov.apply(why, axis=1)
    cov["forecast_status"] = np.where(cov["eligible_for_forecast"], "forecast",
                                      "insufficient data for reliable forecast")
    return cov


# ---------------------------------------------------------------- panel + features
def build_panel(monthly: pd.DataFrame, coverage: pd.DataFrame, months) -> pd.DataFrame:
    """Eligible series reindexed to a contiguous monthly grid. Gap months keep price=NaN / count=0;
    `price_ffill` / `anchor_ffill` carry the last KNOWN value forward (past-only, leak-safe)."""
    months = pd.DatetimeIndex(sorted(months))
    elig = coverage.loc[coverage["eligible_for_forecast"], KEYS]
    m = monthly.merge(elig, on=KEYS)
    full = pd.MultiIndex.from_product([sorted(map(tuple, elig.itertuples(index=False, name=None))), months],
                                      names=["series", "month"])
    m["series"] = list(zip(m["town"], m["flat_type"]))
    p = m.set_index(["series", "month"]).reindex(full).reset_index()
    p["town"] = p["series"].str[0]
    p["flat_type"] = p["series"].str[1]
    p = p.drop(columns="series").sort_values(KEYS + ["month"]).reset_index(drop=True)
    p["price"] = p["median_resale_price"]
    p["has_txn"] = p["transaction_count"].notna()
    p["transaction_count"] = p["transaction_count"].fillna(0)
    g = p.groupby(KEYS, sort=False)
    p["price_ffill"] = g["price"].ffill()
    p["anchor_ffill"] = g["pooled_median_3m"].ffill()
    return p


def origin_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Features known at the end of origin month t, one row per series x month. Horizon-independent."""
    p = panel.sort_values(KEYS + ["month"]).reset_index(drop=True)
    g = p.groupby(KEYS, sort=False)
    f = p[KEYS + ["month", "price_ffill", "anchor_ffill", "transaction_count", "has_txn"]].copy()
    for k in (0, 1, 2, 5, 11):
        f[f"price_lag_{k}"] = g["price_ffill"].shift(k)
    for w in (3, 6, 12):
        f[f"rolling_{w}m_price"] = g["price_ffill"].transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
    f["txn_count_lag_0"] = p["transaction_count"]
    for w in (3, 6):
        f[f"rolling_{w}m_txn_count"] = g["transaction_count"].transform(
            lambda s, w=w: s.rolling(w, min_periods=w).mean())
    f["mom_price_change"] = f["price_lag_0"] / f["price_lag_1"] - 1
    f["quarterly_price_change"] = f["price_lag_0"] / g["price_ffill"].shift(3) - 1
    f["yoy_price_change"] = f["price_lag_0"] / g["price_ffill"].shift(12) - 1
    f["anchor"] = f["anchor_ffill"]
    return f.drop(columns=["price_ffill", "anchor_ffill"])


def make_dataset(panel: pd.DataFrame, h: int, future: bool = False) -> pd.DataFrame:
    """Direct-horizon dataset: origin features at t, target = median price in month t+h.
    future=True returns only the latest origin per series (target unknown) for production forecasts."""
    p = panel.sort_values(KEYS + ["month"]).reset_index(drop=True)
    g = p.groupby(KEYS, sort=False)
    f = origin_features(p)
    f["origin_month"] = f.pop("month")
    f["target_month"] = f["origin_month"] + pd.DateOffset(months=h)
    f["horizon"] = h
    f["y"] = g["price"].shift(-h)                      # actual median (NaN if no sales that month)
    f["y_count"] = g["transaction_count"].shift(-h)
    f["target_year"] = f["target_month"].dt.year
    f["target_month_number"] = f["target_month"].dt.month
    f["target_quarter"] = f["target_month"].dt.quarter
    # baselines, all computable at the origin
    f["base_prev_month"] = f["price_lag_0"]
    f["base_roll3_median"] = f[["price_lag_0", "price_lag_1", "price_lag_2"]].median(axis=1)
    f["base_seasonal_naive"] = g["price_ffill"].shift(12 - h)   # same calendar month last year
    f["base_pooled_3m"] = f["anchor"]
    f["y_rel"] = np.log(f["y"] / f["anchor"])
    last = f["origin_month"].max()
    if future:
        return f[f["origin_month"] == last].reset_index(drop=True)
    need = NUMERIC_FEATURES + ["anchor", "base_seasonal_naive"]
    f = f.dropna(subset=need).copy()                    # drops warm-up months and unusable rows
    return f[f["y"].notna()].reset_index(drop=True)     # score only months with real transactions


# ---------------------------------------------------------------- leakage audit
def audit_features(panel: pd.DataFrame, n_samples: int = 300, seed: int = 0) -> dict:
    """1) No forbidden column in the predictor list.
    2) Truncation test: features at origin t must be IDENTICAL when all data after t is removed
       (and the pooled anchor recomputed from transactions up to t is not needed: it is already trailing)."""
    bad = FORBIDDEN & set(NUMERIC_FEATURES + CATEGORICAL)
    assert not bad, f"forbidden predictors: {bad}"
    full = origin_features(panel)
    rng = np.random.default_rng(seed)
    months = sorted(panel["month"].unique())
    cols = [c for c in full.columns if c not in KEYS + ["month", "has_txn"]]
    checked = mism = 0
    for t in rng.choice(months[13:], size=min(n_samples, len(months) - 13), replace=True):
        trunc = origin_features(panel[panel["month"] <= t])
        a = full[full["month"] == t].set_index(KEYS)[cols].sort_index()
        b = trunc[trunc["month"] == t].set_index(KEYS)[cols].sort_index()
        checked += len(a)
        mism += int((~np.isclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)).any(axis=1).sum())
    assert mism == 0, f"{mism} rows differ after truncation -> look-ahead leakage"
    return {"rows_checked": checked, "mismatches": mism, "forbidden_predictors": sorted(bad)}


# ---------------------------------------------------------------- production forecast
PRODUCTION_METHOD = "pooled_3m_median"


def production_forecast(panel: pd.DataFrame, horizons=(1, 3, 6), generated_at=None) -> pd.DataFrame:
    """Production forecast = the series' pooled median over the 3 months ending at the latest complete month
    (the champion by out-of-sample backtest). It is a smoothed current level, so the same value is
    reported for each horizon. No prediction intervals: none are validated."""
    generated_at = generated_at or pd.Timestamp.now(tz="UTC").tz_localize(None)
    last = panel["month"].max()
    window = panel[panel["month"] > last - pd.DateOffset(months=3)].groupby(KEYS)["transaction_count"].sum()
    rows = []
    for h in horizons:
        f = make_dataset(panel, h, future=True).dropna(subset=["anchor"])
        rows.append(pd.DataFrame({
            "forecast_generated_at": generated_at, "last_observed_month": f["origin_month"],
            "forecast_month": f["target_month"], "forecast_horizon_months": h, "town": f["town"],
            "flat_type": f["flat_type"], "predicted_median_price": f["base_pooled_3m"],
            "forecast_method": PRODUCTION_METHOD,
            "transactions_in_window": f.set_index(KEYS).index.map(window).to_numpy()}))
    return pd.concat(rows, ignore_index=True).sort_values(
        ["town", "flat_type", "forecast_horizon_months"]).reset_index(drop=True)
