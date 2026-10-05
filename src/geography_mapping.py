"""Explicit HDB town -> Census planning-area mapping. No fuzzy matching:
every non-exact case is written out by hand with a note."""
import pandas as pd

# Manual / ambiguous decisions. Anything not listed here must match a planning area exactly (case-insensitive).
MANUAL = {
    "KALLANG/WHAMPOA": ("Kallang", "manually_mapped",
        "HDB town combines Kallang and Whampoa estates. Census 2020 file has no Whampoa subzone and the "
        "estates (Bendemeer, Boon Keng, Whampoa, Geylang Bahru) sit mostly in Kallang PA; some blocks may "
        "fall in neighbouring Novena/Toa Payoh. Treated as approximate."),
    "CENTRAL AREA": (None, "ambiguous",
        "HDB 'Central Area' is a town label for city-centre blocks spread across Downtown Core, Outram, "
        "Rochor, Museum, Singapore River etc. No single planning area; left unmapped on purpose."),
}


def build_mapping(hdb_towns, planning_areas) -> pd.DataFrame:
    pa_lookup = {p.upper(): p for p in planning_areas}
    rows = []
    for t in sorted(hdb_towns):
        if t in MANUAL:
            pa, status, note = MANUAL[t]
        elif t in pa_lookup:
            pa, status, note = pa_lookup[t], "exact", "Same name in Census planning-area list"
        else:
            pa, status, note = None, "unmatched", "No planning area with this name; needs manual review"
        rows.append(dict(hdb_town=t, population_planning_area=pa, mapping_status=status, mapping_note=note))
    return pd.DataFrame(rows)


def usable_for_join(mapping: pd.DataFrame) -> pd.DataFrame:
    """Towns safe to attach Census context to (exact + manually_mapped); ambiguous/unmatched excluded."""
    return mapping[mapping["mapping_status"].isin(["exact", "manually_mapped"])]
