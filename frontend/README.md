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
`node --test frontend/tests/dashboard.test.mjs`. The tests verify exact medians,
all filters, partial-month exclusion, empty/error states and chart update calls
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
