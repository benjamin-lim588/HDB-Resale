"""Census 2020 resident households by planning area and monthly household income from work.

Orientation: one row per planning area, one column per income band (wide). Counts are households.
IMPORTANT definitions:
- `NoEmployedPerson` households have no work income and are NOT in any income band; they are kept as a
  separate column. Band shares / medians below use *households with employed persons* as the base
  (documented as `base = employed households`), and `share_no_employed` is reported separately.
- Income is household income FROM WORK only (excludes CPF payouts, rental, investment, transfers), so it
  understates total household resources, especially for retiree-heavy towns.
"""
import numpy as np
import pandas as pd

# (column, lower, upper) in S$/month. Open-ended top band has upper=None.
BANDS = [("Below_1_000", 0, 1000), ("1_000_1_999", 1000, 2000), ("2_000_2_999", 2000, 3000),
         ("3_000_3_999", 3000, 4000), ("4_000_4_999", 4000, 5000), ("5_000_5_999", 5000, 6000),
         ("6_000_6_999", 6000, 7000), ("7_000_7_999", 7000, 8000), ("8_000_8_999", 8000, 9000),
         ("9_000_9_999", 9000, 10000), ("10_000_10_999", 10000, 11000), ("11_000_11_999", 11000, 12000),
         ("12_000_12_999", 12000, 13000), ("13_000_13_999", 13000, 14000), ("14_000_14_999", 14000, 15000),
         ("15_000_17_499", 15000, 17500), ("17_500_19_999", 17500, 20000), ("20_000andOver", 20000, None)]
BAND_COLS = [b[0] for b in BANDS]
# Assumption for the open-ended $20,000+ band: midpoint-equivalent used ONLY for the mean estimate.
# Sensitivity is reported by estimating the mean at 25k / 30k / 40k.
OPEN_BAND_ASSUMPTIONS = (25000, 30000, 40000)
OPEN_BAND_DEFAULT = 30000


def clean_income(raw: pd.DataFrame):
    """Return (planning-area table with wide counts, long ordered-band table)."""
    df = raw.copy()
    df["Number"] = df["Number"].astype(str).str.strip()
    num = [c for c in df.columns if c != "Number"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")  # any suppressed marker -> NaN, not 0
    df = df.rename(columns={"Number": "planning_area", "Total": "resident_households_total",
                            "NoEmployedPerson": "households_no_employed"})
    df = df[df["planning_area"].ne("Total")].reset_index(drop=True)
    df["households_employed"] = df[BAND_COLS].sum(axis=1, min_count=1)
    df["band_sum_check"] = df["households_employed"] + df["households_no_employed"] - df["resident_households_total"]
    long = df.melt(id_vars="planning_area", value_vars=BAND_COLS, var_name="income_band", value_name="households")
    long["income_band"] = pd.Categorical(long["income_band"], categories=BAND_COLS, ordered=True)
    return df, long


def _median_interpolated(counts, lowers, uppers):
    """Median by linear interpolation inside the band holding the 50th percentile.
    Returns (median, in_open_band). If the median is in the open-ended band it is NOT estimated:
    returns (lower bound of the open band, True)."""
    total = counts.sum()
    cum = np.cumsum(counts)
    k = int(np.searchsorted(cum, total / 2))
    if uppers[k] is None:
        return float(lowers[k]), True
    prev = cum[k - 1] if k else 0
    frac = (total / 2 - prev) / counts[k]
    return lowers[k] + frac * (uppers[k] - lowers[k]), False


def income_features(df: pd.DataFrame) -> pd.DataFrame:
    lowers = [b[1] for b in BANDS]
    uppers = [b[2] for b in BANDS]
    mids = np.array([(lo + up) / 2 if up else np.nan for _, lo, up in BANDS])
    rows = []
    for _, r in df.iterrows():
        c = r[BAND_COLS].values.astype(float)
        base = np.nansum(c)
        med, open_med = _median_interpolated(np.nan_to_num(c), lowers, uppers)
        out = dict(planning_area=r["planning_area"], resident_households_total=r["resident_households_total"],
                   households_employed=base, share_no_employed=r["households_no_employed"] / r["resident_households_total"],
                   estimated_median_household_income=np.nan if open_med else med,
                   median_in_open_band=open_med, median_lower_bound=med)
        for a in OPEN_BAND_ASSUMPTIONS:
            m = np.where(np.isnan(mids), a, mids)
            out[f"estimated_mean_household_income_open{a // 1000}k"] = np.nansum(c * m) / base
        out["estimated_mean_household_income"] = out[f"estimated_mean_household_income_open{OPEN_BAND_DEFAULT // 1000}k"]
        below = lambda x: c[[u is not None and u <= x for u in uppers]].sum() / base
        out["pct_households_below_5k"] = below(5000)
        out["pct_households_below_10k"] = below(10000)
        out["pct_households_above_15k"] = c[[lo >= 15000 for lo in lowers]].sum() / base
        out["pct_households_20k_plus"] = c[-1] / base
        rows.append(out)
    return pd.DataFrame(rows)
