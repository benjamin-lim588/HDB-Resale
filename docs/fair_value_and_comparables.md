# FlatFair valuation: fair value + comparables

Three separate components, never collapsed into one number:

| | Question | Where |
|---|---|---|
| **Market forecast** (existing) | What will the town x flat_type median be in 1/3/6 months? | `docs/forecasting.md`, `features.price_forecasts` |
| **Fair value** | What is *this* flat likely worth now? | `src/fair_value_*.py`, `run_fair_value.py`, UC model `workspace.models.flatfair_fair_value_xgboost@champion` |
| **Comparables** | Which actual recent sales are most similar? Shown as the **"Comparable market reference"** (not the fair value). | `src/comparables.py`, `comparables_config.json` |

Interface: `src/valuation_api.py` (`ValuationEngine.estimate_fair_value`, `.find_comparables`, `get_market_outlook`).

## Conventions
- **Valuation month.** A flat valued in month V uses only sales from months strictly before V (the source has no sale day).
  Current product: valuation month 2026-10, latest complete market month 2026-09. Same convention in every backtest.
- **Anchor** = pooled median of the town x flat_type sales in V-3..V-1; widened to 6 then 12 months, then to the island-wide
  flat_type median, when fewer than 10 sales (the tier used is recorded). ML models predict log(price / anchor).
- **Never used as predictors / for matching:** `resale_price` or `price_per_sqm` of the target, `flag_*`, policy / time-index columns.
- **Leakage audit** (`audit_market_features`): features recomputed from data truncated before V must be identical (240 sampled rows,
  0 mismatches); verified to fail when a leak is injected. Comparables tests (`tests/`): ranking is invariant to prices, only prior
  months are eligible, fewer than 3 reasonable comparables returns "Insufficient comparable transactions".
- **Validation:** chronological. cv1: train <= 2022-09, val 2022-10..2023-09; cv2: <= 2023-09, val 2023-10..2024-09; cv3: <= 2024-09,
  val 2024-10..2025-09; final holdout: train <= 2025-09, test 2025-10..2026-09 (24,646 sales), scored once. Hyperparameters and
  the comparables methodology were chosen on CV only.
- **Feature sets.** CORE (property + prior-market anchors + month of year) is the official basis for selection.
  `core_access` (+ MRT / bus) is reported separately: the accessibility data is today's network with no station opening dates,
  so it is anachronistic for historical sales. `core_year` (+ valuation year) is also an ablation. Neither influenced selection.

## Fair value results (MAE in $; holdout = 2025-10..2026-09)
Champion rule fixed in advance: CV winner (ML must beat hedonic by >1% CV MAE), confirmed against the baseline on the holdout; the
holdout never chooses between models.

| model (CV-selected config) | CV MAE | holdout MAE | MedAE | RMSE | sMAPE | within 5 / 10 / 15% | holdout vs baseline |
|---|---|---|---|---|---|---|---|
| **XGBoost (champion)** depth 8, mcw 10 | 34,981 | **40,346** | 27,824 | 59,464 | 5.89 | 53.6 / 82.8 / 93.9 | -51.4% |
| Random Forest depth 30, leaf 5 | 36,061 | 41,380 | 28,398 | 60,971 | 6.03 | 53.1 / 81.8 / 93.4 | -50.1% |
| Hedonic Ridge alpha 10000 | 52,572 | 59,762 | 36,644 | 91,575 | 8.75 | 42.5 / 70.4 / 84.5 | -28.0% |
| Baseline: recent town x flat_type median | 68,403 | 83,008 | 52,500 | 125,279 | 12.24 | 31.3 / 55.9 / 72.4 | - |

ML genuinely improves individual-flat valuation here (unlike the market forecast): cross-sectional attributes (area, lease, storey,
flat model) carry a lot of signal, and the anchor-relative target keeps models robust to the 2025-26 flattening.
Hedonic regression is the transparent option but is far less accurate (non-linear lease / size effects).

Ablations (holdout, separate from selection): `core_access` XGBoost 34,050 (-15.6% vs CORE; 87.9% within 10%), Random Forest 35,329,
hedonic 57,993; `core_year` XGBoost 39,651 (-1.7%), Random Forest 41,036, hedonic 60,152.

**Weak segments (XGBoost, holdout).** Dear flats: >= $1m only 65.8% within 10% (sMAPE 9.1); < $400k 77.8%. Towns: Clementi 61.6%,
Serangoon 63.3%, Geylang 65.1%, Bukit Merah 65.1%, Toa Payoh 69.9% (best: Yishun 92.2%, Tampines 91.7%, Pasir Ris 91.5%). Leases
< 60 years 79.0% and flats 40+ years old 77.9%. 5 ROOM 79.8%. Thin cells (anchor widened / island fallback) 76.3% and only -38% vs
baseline. Full tables: `features.fair_value_backtest_metrics`.

## Comparables
Hard filters: same town and flat_type; sales strictly before the valuation month; exact duplicates removed; window 12 -> 18 -> 24 months
(first window with >= 10 candidates); loose sanity limits (area +-30%, storey +-10, lease +-20 years, lease normalised to the valuation
date). `flat_model` is a similarity penalty, not a filter (21 models; only 41 of 127 town x flat_type cells have a dominant model > 80%).
Fewer than 3 reasonable comparables -> "Insufficient comparable transactions".

**Canonical methodology (CV-selected):** hedonic-weighted distance in modelled log-price units (area elasticity 0.634, storey and lease
curves, flat-model penalty 0.048) + recency 0.005 per month; K = 5. Frozen in `src/comparables_config.json` (weights, filters, K, recency,
fallback rules, quality thresholds). **Ben's UI should consume / reproduce this file**; his prototype matcher is a reference only.

| method (pooled CV, common target set) | MAE | within 10% |
|---|---|---|
| **hedonic-weighted, K=5 (canonical)** | 30,363 | 88.6% |
| Ben's prototype weights (no location tiers) | 31,121 | 88.0% |
| equal-weight distance | 31,705 | 87.6% |
| standardised KNN distance | 33,702 | 86.0% |
| town x flat_type x flat_model 12m median | 51,774 | 70.6% |
| town x flat_type recent median (baseline) | 68,270 | 60.8% |

Final holdout (3,000 sampled targets, scored once): canonical K=5 **MAE 33,813, MedAE 22,000, sMAPE 5.04, 62.8% within 5% / 87.1%
within 10%**, -58.7% vs the baseline (81,846); flat_model median 57,058; K=3 35,237; K=10 33,991. Accessibility ablation (separate):
30,575.

**Quality label** (HIGH / MEDIUM / LOW): HIGH = 12-month window, >= 20 candidates, mean top-5 distance <= 0.021; MEDIUM = >= 10 candidates
and distance <= 0.0357; else LOW. The distance cut-offs are CV terciles, so by construction a third of CV queries are LOW; the label is
informative because error rises with it on the holdout: HIGH MAE 26,646 (89.6% within 10%), MEDIUM 31,552 (88.2%), LOW 41,998 (84.0%).
It describes the comparable evidence, not a statistical confidence interval.

## Limitations / next steps
- **Comparables currently beat the ML fair value on accuracy** (33.8k vs 40.3k holdout MAE; samples differ in size but baselines are similar:
  81.8k vs 83.0k). Reason: comparables see block / street-level sales; the fair-value models see only town-level anchors. A v2 candidate is a
  hyperlocal prior-sales feature for the fair-value model. Not done: it would be a post-hoc redesign and blur the separation of components.
- The best fair-value configs sit at the top of their (once-extended) grids (depth 8, depth 30, alpha 10000); more search might help slightly.
- Comparables were evaluated on random samples (2,000 targets per CV fold, 3,000 on the holdout), not every sale.
- Accessibility reflects today's MRT / bus network (no opening dates); do not use `core_access` results for historical claims.
- 646 holdout sales (2.6%) used a widened or island-wide anchor and are less accurate; thin and expensive segments are the weak spots.
- XGBoost is not bit-reproducible across machines; the Databricks run is the system of record.

## Tables (Unity Catalog, `workspace.features`) and model
`fair_value_model_comparison`, `fair_value_backtest_metrics`, `fair_value_backtest_predictions`, `fair_value_hedonic_effects`,
`fair_value_market_anchors` (latest valuation month), `comparables_evaluation`, `comparables_config`, `comparables_quality_calibration`.
Registry: only `workspace.models.flatfair_fair_value_xgboost` (alias `champion`). Output of the model = log(price / anchor);
use `ValuationEngine`, which builds the anchor and exponentiates. The comparables engine is not registered. MLflow experiments:
`/Users/<you>/flatfair-fair-value`, `/Users/<you>/flatfair-comparables`.

## Run
- Databricks: `databricks bundle deploy && databricks bundle run flatfair_valuation` (tasks `fair_value`, `comparables_eval`).
- Local: `.venv/bin/python src/run_fair_value.py [--cv-only]`, `.venv/bin/python src/run_comparables_evaluation.py [--cv-only]`,
  `.venv/bin/python -m pytest tests`.
