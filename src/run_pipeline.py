"""Cleaning + feature engineering, as two pure stages that return {table_name: DataFrame}.

Local:       python src/run_pipeline.py          (CSV in, CSV out -- same files as before)
Databricks:  jobs/02_clean.py and jobs/03_features.py call the same stage functions with Delta tables.
"""
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


def clean_stage(raw: dict) -> dict:
    """raw: hdb, population, income (as read from the source files / raw tables)."""
    hdb = dc.clean_hdb(raw["hdb"])
    full, pa = dc.clean_population(raw["population"])
    inc_wide, inc_long = hi.clean_income(raw["income"])
    mapping = gm.build_mapping(hdb["town"].unique(), pa["planning_area"])
    return {"hdb_resale": hdb,
            "population_all_levels": full,
            "population_planning_area": pa,
            "population_reconciliation": dc.population_reconciliation(full),
            "household_income_wide": inc_wide,
            "household_income_long": inc_long,
            "town_planning_area_mapping": mapping,
            "cleaning_rules": dc.rules_table()}


def features_stage(clean: dict, events: pd.DataFrame, geocode=None, mrt=None, bus=None) -> dict:
    """clean: output of clean_stage (or the clean tables). events: policy events, effective_date as datetime."""
    n_rules = len(dc.RULES)  # RULES is process-global; keep only the rules logged by this stage
    hdb = fe.add_policy_period(fe.add_property_features(fe.add_time_features(clean["hdb_resale"])), events)
    hdb = dc.flag_unusual_hdb(hdb)

    pop = fe.population_features(clean["population_planning_area"])
    inc = hi.income_features(clean["household_income_wide"])
    # Income: same planning-area names/geography as population, so the same mapping applies.
    ctx = pop.merge(inc, on="planning_area", how="outer")

    ok = gm.usable_for_join(clean["town_planning_area_mapping"])[["hdb_town", "population_planning_area"]]
    h20 = hdb[hdb["transaction_year"] == 2020].merge(ok, left_on="town", right_on="hdb_town", how="inner")
    h20 = h20.merge(ctx, left_on="population_planning_area", right_on="planning_area", how="left")

    out = {"hdb_transactions": hdb, "population_planning_area": pop, "planning_area_context": ctx,
           "hdb_2020_enriched": h20}

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
            out[f"affordability_{name}_{label}"] = g.drop(columns=["hdb_town", "planning_area"])

    props = hdb[["property_id", "block", "street_name", "town"]].drop_duplicates("property_id")
    out["property_table"] = props  # input for geocoding

    if geocode is not None:  # completed OneMap geocoding is an input; this stage never calls the API
        loc = geocode[geocode["status"].isin(["ok", "ok_road_differs"])]
        out["property_locations"] = props.merge(loc, on="property_id", how="left")
        acc = gf.accessibility_from_poi(loc[["property_id", "latitude", "longitude"]], mrt, bus)
        if acc is not None:
            out["property_accessibility"] = acc

    out["flag_rules"] = dc.rules_table().iloc[n_rules:].reset_index(drop=True)
    return out


# ------------------------------------------------------------------ local CSV runner
def main():
    raw = {"hdb": pd.read_csv(HDB_RAW), "population": pd.read_csv(POP_RAW), "income": pd.read_csv(INCOME_RAW)}
    clean = clean_stage(raw)
    events = pd.read_csv(ROOT / "data" / "policy_events.csv", parse_dates=["effective_date"])
    cache = CACHE / "onemap_geocode_cache.csv"
    feats = features_stage(
        clean, events,
        geocode=pd.read_csv(cache) if cache.exists() else None,
        mrt=gf.load_mrt_stations() if MRT_RAW.exists() else None,
        bus=gf.load_bus_stops() if BUS_RAW.exists() else None)

    stages = {"clean": clean, "feats": feats}
    to_processed = {  # (stage, key) -> file in data/processed
        ("clean", "population_all_levels"): "population_all_levels_clean.csv",
        ("clean", "household_income_wide"): "household_income_planning_area_clean.csv",
        ("clean", "household_income_long"): "household_income_bands_long.csv",  # raw distribution for audit
        ("feats", "hdb_transactions"): "hdb_clean.csv",
        ("feats", "population_planning_area"): "population_planning_area_clean.csv",
        ("feats", "planning_area_context"): "planning_area_context.csv",
        ("feats", "hdb_2020_enriched"): "hdb_2020_enriched.csv",
        ("feats", "property_table"): "hdb_property_table.csv",
        ("feats", "property_locations"): "hdb_property_locations.csv",
        ("feats", "property_accessibility"): "hdb_accessibility_features.csv"}
    for (stage, key), fname in to_processed.items():
        if key in stages[stage]:
            stages[stage][key].to_csv(PROCESSED / fname, index=False)
    for key in ("population_reconciliation", "town_planning_area_mapping"):
        clean[key].to_csv(TABLES / f"{key}.csv", index=False)
    for key in [k for k in feats if k.startswith("affordability_")]:
        feats[key].to_csv(TABLES / f"{key}.csv", index=False)
    pd.concat([clean["cleaning_rules"], feats["flag_rules"]]).to_csv(TABLES / "cleaning_rules.csv", index=False)

    print(len(raw["hdb"]), "raw ->", len(clean["hdb_resale"]), "clean;",
          len(feats["property_table"]), "unique blocks;", len(feats["hdb_2020_enriched"]), "2020 enriched rows")
    print(clean["town_planning_area_mapping"].mapping_status.value_counts().to_dict())


if __name__ == "__main__":
    main()
