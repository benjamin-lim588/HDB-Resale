"""Cleaning functions for the HDB resale and Census population datasets.

Every function returns the cleaned frame; rule bookkeeping goes through `log_rule`
so a cleaning-rule table (column, issue, rule, reason, rows_affected) is built as we go.
"""
import numpy as np
import pandas as pd

RULES: list[dict] = []


def log_rule(column, issue, rule, reason, rows_affected):
    RULES.append(dict(column=column, issue=issue, rule=rule, reason=reason,
                      rows_affected=int(rows_affected)))


def rules_table() -> pd.DataFrame:
    return pd.DataFrame(RULES)


# ---------------------------------------------------------------- HDB
def clean_hdb(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw HDB resale transactions. Rows are NOT dropped except for hard
    data-quality failures (non-positive price/area, unparsable month)."""
    df = df.copy()
    n0 = len(df)

    # strip + upper-case categoricals (street/town/flat_type are upper already; flat_model is not)
    text_cols = ["town", "flat_type", "block", "street_name", "storey_range",
                 "flat_model", "remaining_lease"]
    changed = 0
    for c in text_cols:
        new = df[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        changed += int((new != df[c].astype(str)).sum())
        df[c] = new
    log_rule("text columns", "stray/double whitespace", "strip + collapse whitespace",
             "avoid spurious category levels", changed)

    # flat_model casing is inconsistent ('Multi Generation' vs 'MULTI-GENERATION' flat_type etc.)
    before = df["flat_model"].nunique()
    df["flat_model"] = df["flat_model"].str.title().replace({"Dbss": "DBSS", "2-Room": "2-room",
                                                             "3Gen": "3Gen"})
    log_rule("flat_model", "mixed casing", "title-case (DBSS kept upper)",
             f"levels {before} -> {df['flat_model'].nunique()}", (before != df['flat_model'].nunique()) * 0)

    # month -> date
    df["transaction_date"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    bad = df["transaction_date"].isna().sum()
    log_rule("month", "string 'YYYY-MM'", "parse to first-of-month datetime",
             "enable time features", bad)
    df = df[df["transaction_date"].notna()]

    # numeric validity
    for c in ["floor_area_sqm", "resale_price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        bad = ~(df[c] > 0)
        log_rule(c, "must be > 0", "drop rows where value is missing or <= 0",
                 "impossible value", bad.sum())
        df = df[~bad]

    # exact duplicates: FLAGGED, not dropped (two real flats can share every attribute
    # because the file has no unit number or transaction day).
    df["is_exact_duplicate"] = df.duplicated(keep="first")
    log_rule("all columns", "exact duplicate rows", "flag only (is_exact_duplicate); keep rows",
             "no unit id / day in data so identical sales may be genuine", df["is_exact_duplicate"].sum())

    df["property_id"] = make_property_id(df["block"], df["street_name"])
    log_rule("block+street_name", "need a block key", "property_id = upper(block + ' ' + street)",
             "geocode / join once per block", df["property_id"].nunique())
    log_rule("rows", "net row change", "see above", "-", n0 - len(df))
    return df.reset_index(drop=True)


def make_property_id(block: pd.Series, street: pd.Series) -> pd.Series:
    return (block.astype(str) + " " + street.astype(str)).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()


def parse_remaining_lease(s: pd.Series) -> pd.Series:
    """'61 years 04 months' -> 736 ; '61 years' -> 732 ; '1 year 1 month' -> 13."""
    yrs = pd.to_numeric(s.str.extract(r"(\d+)\s*year")[0], errors="coerce").fillna(0)
    mos = pd.to_numeric(s.str.extract(r"(\d+)\s*month")[0], errors="coerce").fillna(0)
    out = yrs * 12 + mos
    return out.where(s.notna())


def parse_storey_range(s: pd.Series) -> pd.DataFrame:
    parts = s.str.extract(r"(\d+)\s*TO\s*(\d+)").astype(float)
    parts.columns = ["storey_lower", "storey_upper"]
    parts["storey_mid"] = (parts["storey_lower"] + parts["storey_upper"]) / 2
    return parts


def flag_unusual_hdb(df: pd.DataFrame) -> pd.DataFrame:
    """Flag (never delete) suspicious records.
    - lease_inconsistent: remaining lease deviates >2y from 99 - age at transaction
    - extreme_price_per_sqm: outside the 0.1-99.9 percentile *within flat_type* (flag only)
    - large_area_for_type: area far above the median of its flat_type (>2x)
    """
    df = df.copy()
    age = df["transaction_date"].dt.year - df["lease_commence_date"]
    expected = 99 * 12 - age * 12
    df["flag_lease_inconsistent"] = (df["remaining_lease_months"] - expected).abs() > 24
    g = df.groupby("flat_type")["price_per_sqm"]
    lo, hi = g.transform(lambda x: x.quantile(.001)), g.transform(lambda x: x.quantile(.999))
    df["flag_extreme_price_per_sqm"] = (df["price_per_sqm"] < lo) | (df["price_per_sqm"] > hi)
    med_area = df.groupby("flat_type")["floor_area_sqm"].transform("median")
    df["flag_large_area_for_type"] = df["floor_area_sqm"] > 2 * med_area
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    df["any_flag"] = df[flag_cols].any(axis=1)
    for c in flag_cols + ["is_exact_duplicate"]:
        log_rule(c, "unusual record", "flag only, not removed", "may be a genuine transaction", df[c].sum())
    return df


# ---------------------------------------------------------------- Population
def clean_population(raw: pd.DataFrame):
    """Return the full tidy table (with hierarchy level) and the planning-area-only table.

    '-' is a suppressed/unavailable value -> NaN (NOT zero).
    """
    df = raw.copy()
    count_cols = [c for c in df.columns if c != "Number"]
    dash = int((df[count_cols] == "-").sum().sum())
    for c in count_cols:
        df[c] = pd.to_numeric(df[c].replace("-", np.nan), errors="coerce")
    log_rule("population counts", "'-' literal string", "replace with NaN, cast to float",
             "suppressed != zero", dash)

    name = df["Number"].str.strip()
    is_total = name.eq("Total")
    # one source typo: 'Changi- Total' (no space before hyphen)
    is_pa = name.str.contains(r"\s?-\s*Total$", regex=True) & ~is_total
    df["level"] = np.where(is_total, "national", np.where(is_pa, "planning_area", "subzone"))
    df["area_name"] = name.str.replace(r"\s?-\s*Total$", "", regex=True).str.strip()
    # subzones inherit their parent planning area (rows are ordered hierarchically)
    df["planning_area"] = df["area_name"].where(df["level"] == "planning_area").ffill()
    df.loc[df["level"] == "national", "planning_area"] = np.nan
    log_rule("Number", "mixed hierarchy in one column", "classify national / planning_area / subzone; "
             "fix 'Changi- Total'", "separate levels", len(df))
    pa = df[df["level"] == "planning_area"].copy()
    return df, pa


def population_reconciliation(full: pd.DataFrame) -> pd.DataFrame:
    """Compare each planning-area total with the sum of its subzones.
    Census figures are rounded to the nearest 10, and '-' subzones are NaN, so small gaps are expected."""
    sub = full[full["level"] == "subzone"].groupby("planning_area")["Total_Total"].agg(
        subzone_sum="sum", n_subzones="size", n_subzone_missing=lambda x: x.isna().sum())
    pa = full[full["level"] == "planning_area"].set_index("planning_area")["Total_Total"].rename("pa_total")
    out = pa.to_frame().join(sub)
    out["diff"] = out["pa_total"] - out["subzone_sum"]
    out["diff_pct"] = out["diff"] / out["pa_total"] * 100
    return out.reset_index()
