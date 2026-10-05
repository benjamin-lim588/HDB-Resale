# FlatFair - HDB resale analytics (EDA and data cleaning stage)

```
data/raw/        original files (never modified): HDB resale CSV, Census 2020 population CSV, HDB AR 2025 PDF
data/processed/  cleaned tables written by src/run_pipeline.py
data/policy_events.csv   policy/cooling-measure timeline (rows with verified_in_repo=no must be checked before citing)
src/             paths, data_cleaning, geography_mapping, feature_engineering, geographic_features, run_pipeline
notebooks/       01_initial_data_inspection, 02_eda
outputs/         figures/ and tables/ (cleaning rules, town-planning-area mapping, summaries)
```

Run: `python src/run_pipeline.py`, then run the notebooks top to bottom.

**Still missing:** Census 2020 household income by planning area, MRT/LRT stations, bus stops
(drop them in `data/raw/` using the filenames in `src/paths.py`; income and accessibility sections activate then).
`price_per_sqm` is EDA-only (target leakage). Census variables are static 2020 context.

## Frontend — Phase 1 Market Explorer

The vanilla HTML/CSS/JavaScript dashboard reads a separately prepared local export.
It does not change or invoke the `src/` pipeline.

```powershell
python frontend/prepare_data.py
python -m http.server 8000 --bind 127.0.0.1 --directory frontend
```

Open http://127.0.0.1:8000. See [frontend/README.md](frontend/README.md) for data
preparation, filter behavior, verification and the future API boundary.
