"""Reusable feature-engineering functions."""
import numpy as np
import pandas as pd

from data_cleaning import parse_remaining_lease, parse_storey_range


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["transaction_date"]
    df["transaction_year"] = d.dt.year
    df["transaction_month"] = d.dt.month
    df["transaction_quarter"] = d.dt.quarter
    start = d.min()
    df["months_since_start_of_dataset"] = (d.dt.year - start.year) * 12 + (d.dt.month - start.month)
    return df


def add_property_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["remaining_lease_months"] = parse_remaining_lease(df["remaining_lease"])
    df = df.join(parse_storey_range(df["storey_range"]))
    df["flat_age_at_transaction"] = df["transaction_year"] - df["lease_commence_date"]
    # EDA-only metric. LEAKAGE: never use as a predictor of resale_price.
    df["price_per_sqm"] = df["resale_price"] / df["floor_area_sqm"]
    return df


def add_policy_period(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Label each transaction with the number of cooling-measure/policy events on or before its month.
    Descriptive context only -- not a causal identification of policy effects."""
    df = df.copy()
    ev = events.sort_values("effective_date")
    idx = np.searchsorted(ev["effective_date"].values, df["transaction_date"].values, side="right")
    df["policy_events_to_date"] = idx
    return df


def population_features(pa: pd.DataFrame) -> pd.DataFrame:
    """Planning-area demographic context (STATIC Census 2020). Shares use the area's own total."""
    out = pd.DataFrame({"planning_area": pa["planning_area"], "population_total": pa["Total_Total"]})
    for k, col in [("male", "Total_Males"), ("female", "Total_Females"), ("chinese", "Chinese_Total"),
                   ("malay", "Malays_Total"), ("indian", "Indians_Total"), ("others", "Others_Total")]:
        out[f"{k}_share"] = pa[col] / pa["Total_Total"]
    return out.reset_index(drop=True)


def price_to_income(prices: pd.DataFrame, income_col="estimated_median_household_income") -> pd.DataFrame:
    """Housing-price-to-income indicator = median resale price / estimated annual household income.
    NOT mortgage affordability (ignores CPF, grants, rates, household structure, debt)."""
    out = prices.copy()
    out["estimated_annual_household_income"] = out[income_col] * 12
    out["price_to_income_ratio"] = out["median_price"] / out["estimated_annual_household_income"]
    return out
