# FlatFair Market Explorer — Phase 1

From the repository root:

```powershell
python frontend/prepare_data.py
python -m http.server 8000 --bind 127.0.0.1 --directory frontend
```

Open http://127.0.0.1:8000. Do not open index.html directly. No Python packages,
Node build tool, or internet connection are needed to run the dashboard. Plotly.js
is vendored locally in `vendor/plotly.min.js` (v2.35.2, MIT license).

Optional verification after preparation (Node 22+, no npm install required):
`node --test frontend/tests/*.test.mjs`. The tests verify exact medians,
all filters, cross-filter clicks, chips, comparison limits, legends, metric toggles,
partial-month exclusion, empty/error states and chart update calls
against the current supplied dataset. They do not replace browser visual testing.

Re-run preparation whenever Ryan regenerates `data/processed/hdb_clean.csv`, then
reload the page. Preparation validates required fields, dates, categories and
positive numeric prices; it fails with a readable message on missing/invalid data.
It reads the CSV without changing it and writes `data/transactions.json` atomically.
This generated file is ignored by Git.

The export keeps only the six values needed per transaction: dictionary-encoded
calendar month, town, flat type and storey range, plus resale price and price per
sqm. Year/month are validated against transaction_date. Sorted transaction indexes
for both prices are generated once. Exact medians cannot be reconstructed from
medians of aggregated groups, so the export preserves individual price values.
The browser fetches and validates data once, uses typed arrays, and queries via
linear scans of sorted indexes. Queries are debounced and the latest result is
cached; charts update through Plotly.react without a page reload.

`js/data.js` exposes `getFilterOptions`, `getDashboardData`,
`getMonthlyPriceTrend`, `getMonthlyPsmTrend`, `getTransactionVolume` and
`getTownComparison`. The UI and chart components depend only on their result
shapes. A future authorized API integration can replace this provider without
changing those components. No Databricks integration is included.

Assumptions:
- The latest month is considered partial when it is the current or a future
  calendar month in Singapore. Historical completeness cannot be proven from
  month-level rows; low volume alone is not treated as evidence of partial data.
- Partial-month transactions remain in KPIs and town comparison, but are excluded
  from all three time-series charts. Missing intervening months appear as line
  gaps and zero volume.
- Town comparison shows **all towns**, honoring flat type, storey and year filters.
  The Town selection applies to KPIs and time-series charts. This scope is labeled
  on the comparison card.
- All pipeline rows, including flagged duplicates, remain included. No cleaning,
  modelling, estimation, forecasting, affordability or scraping is performed.

Dashboard interactions:
- **Town cross-filtering:** click a town bar to select only that town. All other
  filters stay in place. The bar chart continues to show all towns under those
  other filters, with the active towns highlighted, so another bar can be clicked.
- **Town comparison:** All towns overview preserves the combined market view.
  Choosing individual towns switches to separate trend lines for up to three
  towns, with consistent colors across both trends and the highlighted town bars.
  After three selections, additional town checkboxes are disabled until a town
  is removed. KPIs and volume cover the combined selected transactions.
- **Filter chips:** non-default categorical selections and the year range appear
  above the charts. Removing a categorical chip removes that selected value;
  removing the last chip for a dimension restores all its values. Removing a year
  chip restores the full year range. Clear selections get a removable empty-state
  chip. Reset all and the sidebar Reset restore every filter.
- **Primary metric:** switch between median resale price and price per sqm without
  changing selections. Reset controls reset filters and retain the metric choice.
- **Hover details:** monthly points expose month, town/selection, both exact
  medians and transaction count. Town bars expose both medians and count.

Event tests use small DOM/Plotly test doubles to exercise the real application
handlers, including listener cleanup and empty-state recovery. Browser visual
layout and console verification require a connected browser.

## Fair Value Check — comparable-sales prototype

On `ben/fair-value-comparables`, prepare both independent frontend exports and serve:

```powershell
python frontend/prepare_data.py
python frontend/prepare_comparables_data.py
python -m http.server 8000 --bind 127.0.0.1 --directory frontend
```

Open `http://127.0.0.1:8000/fair-value.html`, or use the Market Explorer navigation
link. The new page reuses the existing stylesheet and local Plotly library. No
`src/`, `jobs/`, forecasting or affordability files are changed by this feature.

The 12-field export `data/comparables.json` contains dictionary-encoded date,
town, street, block, flat type, model and storey range, plus area, remaining lease
months, storey midpoint, price and price per sqm. It is ignored by Git and must be
regenerated after the cleaned source changes. Identical exported records are
deduplicated locally to avoid counting indistinguishable transactions twice;
no source rows are changed. Dates retain the source's registration-month
granularity; a first-of-month date does not identify the actual sale day.

All thresholds and weights are in `js/comparable-matching.js` (`MATCHING`):

- Same town and flat type are required; use only non-future sales within 36
  calendar months of the analysis date (today in Singapore).
- Candidates must be within 20% of the target area, 6 floors of the target storey
  midpoint, and 120 months of the target remaining lease. For lease matching,
  subtract elapsed calendar months from each sale's lease. The table preserves
  lease at sale and separately labels estimated remaining lease today.
- Prefer same block **and street**, then other blocks on the same street, then
  other streets in the same town. Widen only while fewer than 5 eligible sales
  have been selected. Within each visited tier take up to the remaining capacity
  of 10 sales, ranked by normalized weighted differences: area 25%, storey 15%,
  lease 20%, recency 35%, optional model mismatch 5%. Ties use newer date,
  street, block, then export row ID. Matching never reads asking price.
- Quality is High with at least 5 recent (≤12 months) block/street sales; Medium
  with at least 5 sales including at least 3 block/street sales; otherwise Low.
  No comparables yields Unavailable. This is a rule about evidence, not model
  prediction confidence or a statistical interval.

`Comparable-implied value = median(selected price_per_sqm) × target floor area`.
Median transaction price is computed separately from actual selected prices.
Range uses the selected minimum/maximum price per sqm scaled to target area.
Difference is asking price minus implied value; premium/discount is
`(asking price / implied value − 1) × 100%`. Neutral above/within/below labels
compare asking price with that range. Prices are not time-adjusted, and the page
does not claim an appraisal. The chart shows actual transaction prices and
clearly labels the separate target-area benchmark and asking price.

`getComparableAnalysis(property)` is the presentation-facing provider boundary.
`getModelFairValue(property)` currently returns `null`; ML fair value is displayed
as Unavailable. No ML, external listings, scraping or Databricks are implemented.

Verification from the repository root:

```powershell
node --test frontend/tests/*.test.mjs
python -m unittest discover -s frontend/tests -p "test_*.py"
```

The new tests cover matching tiers, type/location requirements, similarity,
recency/threshold boundaries, asking-price independence, value and range math,
confidence, missing data, form validation, dependent dropdowns, empty-result
recovery, Plotly calls and unchanged source files. Event tests use DOM/Plotly
test doubles; visual layout and browser console checks need a connected browser.
