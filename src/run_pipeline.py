"""Run cleaning + feature engineering end to end and write data/processed/*.parquet-free CSVs.
Usage: python src/run_pipeline.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_cleaning as dc
import feature_engineering as fe
import geography_mapping as gm
import geographic_features as gf
import household_income as hi
from paths import *


def main():
    raw = pd.read_csv(HDB_RAW)
    hdb = dc.clean_hdb(raw)
    events = pd.read_csv(ROOT / "data" / "policy_events.csv", parse_dates=["effective_date"])
    hdb = fe.add_policy_period(fe.add_property_features(fe.add_time_features(hdb)), events)
    hdb = dc.flag_unusual_hdb(hdb)
    hdb.to_csv(PROCESSED / "hdb_clean.csv", index=False)

    full, pa = dc.clean_population(pd.read_csv(POP_RAW))
    pop = fe.population_features(pa)
    full.to_csv(PROCESSED / "population_all_levels_clean.csv", index=False)
    pop.to_csv(PROCESSED / "population_planning_area_clean.csv", index=False)
    dc.population_reconciliation(full).to_csv(TABLES / "population_reconciliation.csv", index=False)

    # Income: same planning-area names/geography as population, so the same mapping applies.
    inc_wide, inc_long = hi.clean_income(pd.read_csv(INCOME_RAW))
    inc = hi.income_features(inc_wide)
    inc_wide.to_csv(PROCESSED / "household_income_planning_area_clean.csv", index=False)
    inc_long.to_csv(PROCESSED / "household_income_bands_long.csv", index=False)  # raw distribution for audit
    ctx = pop.merge(inc, on="planning_area", how="outer")
    ctx.to_csv(PROCESSED / "planning_area_context.csv", index=False)

    mapping = gm.build_mapping(hdb["town"].unique(), pop["planning_area"])
    mapping.to_csv(TABLES / "town_planning_area_mapping.csv", index=False)

    ok = gm.usable_for_join(mapping)[["hdb_town", "population_planning_area"]]
    h20 = hdb[hdb["transaction_year"] == 2020].merge(ok, left_on="town", right_on="hdb_town", how="inner")
    h20 = h20.merge(ctx, left_on="population_planning_area", right_on="planning_area", how="left")
    h20.to_csv(PROCESSED / "hdb_2020_enriched.csv", index=False)

    # Affordability indicator (price-to-income), STATIC 2020 income vs prices from ALL years -> also 2020-only
    for label, base in [("all_years", hdb), ("2020", hdb[hdb["transaction_year"] == 2020])]:
        for keys, name in [(["town"], "town"), (["town", "flat_type"], "town_flat_type")]:
            g = base.groupby(keys).agg(transaction_count=("resale_price", "size"),
                                       median_price=("resale_price", "median"),
                                       median_price_per_sqm=("price_per_sqm", "median")).reset_index()
            g = g.merge(ok, left_on="town", right_on="hdb_town").merge(
                inc[["planning_area", "estimated_median_household_income", "estimated_mean_household_income"]],
                left_on="population_planning_area", right_on="planning_area")
            g = fe.price_to_income(g)
            g["low_sample"] = g["transaction_count"] < 30
            g.drop(columns=["hdb_town", "planning_area"]).to_csv(TABLES / f"affordability_{name}_{label}.csv", index=False)

    props = hdb[["property_id", "block", "street_name", "town"]].drop_duplicates("property_id")
    props.to_csv(PROCESSED / "hdb_property_table.csv", index=False)  # input for geocoding

    acc_path = PROCESSED / "hdb_accessibility_features.csv"
    cache = CACHE / "onemap_geocode_cache.csv"
    if cache.exists():
        loc = pd.read_csv(cache)
        loc = loc[loc["status"].isin(["ok", "ok_road_differs"])]
        locs = props.merge(loc, on="property_id", how="left")
        locs.to_csv(PROCESSED / "hdb_property_locations.csv", index=False)
        acc = gf.build_accessibility(loc[["property_id", "latitude", "longitude"]])
        if acc is not None:
            acc.to_csv(acc_path, index=False)

    dc.rules_table().to_csv(TABLES / "cleaning_rules.csv", index=False)
    print(len(raw), "raw ->", len(hdb), "clean;", len(props), "unique blocks;", len(h20), "2020 enriched rows")
    print(mapping.mapping_status.value_counts().to_dict())


if __name__ == "__main__":
    main()
