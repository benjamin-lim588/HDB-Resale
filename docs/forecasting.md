# FlatFair price forecasting (town x flat_type)

Owner: Ryan. Source table: `workspace.features.hdb_transactions`. Code: `src/forecast_data.py`, `forecast_models.py`,
`forecast_mlflow.py`, `run_backtest.py`; Databricks tasks `jobs/04-06` (bundle job `flatfair-forecasting`).

## Conclusion
**Production forecast = pooled 3-month median** (`forecast_method = "pooled_3m_median"`): the median of all
transactions of a town x flat_type in the 3 months ending at the latest complete month (2026-09). It generalised best
out of sample. Random Forest was the best ML model in cross-validation, but its edge over a smoothed, drift-aware
baseline was small, and in the final test window **no ML model consistently beat the pooled 3-month median**.
Random Forest / XGBoost / Ridge remain **challengers** tracked in MLflow; Random Forest is registered in Unity Catalog
as a challenger only (`workspace.models.flatfair_rf_challenger_h{1,3,6}`, alias `challenger`, tag `production=false`).
The pooled median is not an ML model and is deliberately **not** registered. No prediction intervals are produced:
none have been validated.

## Setup
- **Grain:** month x town x flat_type; target = median resale price; direct forecasts at 1, 3 and 6 months.
- **Cohort (locked):** `active_months >= 84` AND `median monthly transaction count >= 5` -> **85 of 131 series**
  (96.43% of transactions). The 46 excluded series are kept in the data (`features.series_coverage`,
  `forecast_status = 'insufficient data for reliable forecast'`): historical analytics only, no ML forecast.
- **Partial month:** 2026-10 (212 transactions) is dropped; the last complete month is 2026-09.
- **Features:** only data up to the forecast origin (price lags 0/1/2/5/11, 3/6/12m rolling prices, transaction-count
  lags/rolling, momentum, calendar of the target month, town, flat_type). `price_per_sqm` and anything dated after the
  origin are excluded. An automated truncation audit rebuilds features with all later data removed and requires identical
  values (0 mismatches over 8,840 origin rows; verified to fail when a leak is injected).
- **ML target:** log(y_t+h / pooled 3m median at t); price-level features are fed as log(feature / anchor).
- **Validation:** chronological expanding windows. CV: cv1 (train <= 2022-09, val 2022-10..2023-09), cv2 (<= 2023-09,
  val 2023-10..2024-09), cv3 (<= 2024-09, val 2024-10..2025-09). **Final test:** train <= 2025-09, val 2025-10..2026-09,
  scored once; hyperparameters were chosen on CV only. 3,059 scored points per horizon in CV, 1,019 in test.

## Results (MAE in $; improvement is MAE reduction vs naive previous-month)
| model | horizon | status | config | CV MAE | test MAE | test RMSE | test sMAPE % | CV impr. vs naive | test impr. vs naive |
|---|---|---|---|---|---|---|---|---|---|
| pooled_3m_median | 1m | production | - | 30,082 | 36,785 | 68,150 | 4.81 | +14.4% | +17.3% |
| random_forest | 1m | challenger | depth=6,leaf=5 | 27,934 | 37,142 | 63,669 | 4.97 | +20.5% | +16.5% |
| xgboost | 1m | challenger | depth=3,mcw=20 | 28,495 | 38,039 | 64,481 | 5.14 | +18.9% | +14.5% |
| ridge | 1m | challenger | alpha=1 | 31,052 | 43,022 | 65,467 | 5.88 | +11.6% | +3.2% |
| drift_pooled_3m | 1m | baseline | - | 29,409 | 37,851 | 69,031 | 4.96 | +16.3% | +14.9% |
| rolling3_median | 1m | baseline | - | 30,578 | 38,217 | 70,267 | 5.01 | +13.0% | +14.1% |
| naive_prev_month | 1m | baseline | - | 35,133 | 44,465 | 78,546 | 5.91 | +0.0% | +0.0% |
| seasonal_naive | 1m | baseline | - | 55,783 | 46,604 | 81,488 | 6.23 | -58.8% | -4.8% |
| pooled_3m_median | 3m | production | - | 32,903 | 37,374 | 71,000 | 4.88 | +9.7% | +14.6% |
| random_forest | 3m | challenger | depth=12,leaf=20 | 29,878 | 42,571 | 67,797 | 5.80 | +18.0% | +2.7% |
| xgboost | 3m | challenger | depth=3,mcw=5 | 30,766 | 43,930 | 70,873 | 5.98 | +15.6% | -0.4% |
| ridge | 3m | challenger | alpha=1 | 34,858 | 50,635 | 71,568 | 6.98 | +4.4% | -15.7% |
| rolling3_median | 3m | baseline | - | 33,707 | 38,009 | 71,177 | 4.98 | +7.5% | +13.2% |
| drift_pooled_3m | 3m | baseline | - | 30,324 | 41,223 | 73,394 | 5.46 | +16.8% | +5.8% |
| naive_prev_month | 3m | baseline | - | 36,445 | 43,765 | 81,051 | 5.75 | +0.0% | +0.0% |
| seasonal_naive | 3m | baseline | - | 55,783 | 46,604 | 81,488 | 6.23 | -53.1% | -6.5% |
| pooled_3m_median | 6m | production | - | 39,613 | 38,369 | 71,174 | 5.04 | +5.8% | +15.9% |
| random_forest | 6m | challenger | depth=12,leaf=20 | 32,304 | 48,572 | 69,974 | 6.77 | +23.2% | -6.5% |
| xgboost | 6m | challenger | depth=3,mcw=5 | 33,008 | 49,297 | 71,979 | 6.86 | +21.5% | -8.1% |
| ridge | 6m | challenger | alpha=1 | 42,364 | 62,752 | 80,811 | 8.75 | -0.8% | -37.6% |
| rolling3_median | 6m | baseline | - | 39,824 | 39,281 | 71,690 | 5.20 | +5.3% | +13.9% |
| naive_prev_month | 6m | baseline | - | 42,037 | 45,600 | 82,576 | 6.02 | +0.0% | +0.0% |
| drift_pooled_3m | 6m | baseline | - | 32,568 | 46,510 | 75,578 | 6.29 | +22.5% | -2.0% |
| seasonal_naive | 6m | baseline | - | 55,783 | 46,604 | 81,488 | 6.23 | -32.7% | -2.2% |

The same table is in `workspace.features.model_comparison`; segment cuts (town, flat_type, per fold) are in
`workspace.features.forecast_backtest_metrics`. Weak segments for Random Forest on test: it is worse than naive in
large young estates (Sembawang, Sengkang, Jurong West, Woodlands, Yishun) and for 3 ROOM / 2 ROOM; it is better in
expensive central/mature towns (Central Area, Queenstown, Clementi, Geylang).

## Key finding: regime change
Median price change over the horizon (`workspace.features.forecast_regime_drift`). Training-window drift is what the
models learned; realised drift is what happened in the validation window.

| window | 1m train | 1m realised | 3m train | 3m realised | 6m train | 6m realised |
|---|---|---|---|---|---|---|
| cv1 | +0.9% | +0.8% | +2.0% | +1.8% | +3.1% | +3.6% |
| cv2 | +0.9% | +1.4% | +2.0% | +2.6% | +3.3% | +3.7% |
| cv3 | +1.0% | +1.1% | +2.0% | +2.1% | +3.4% | +4.3% |
| test | +1.0% | -0.3% | +2.0% | -0.5% | +3.6% | -0.4% |

Positive drift (+3% to +4% over 6 months) held in every CV window, but weakened and **reversed** in the final test
window (about -0.4%; median year-on-year change fell from roughly +6% in 2025 to -1% by 2026-08). Models trained on
earlier growth regimes over-predicted. A shuffled-target control (anchor x average drift, no signal) already
captured most of the CV gain over naive, so much of the apparent ML skill was learned drift. The test fold was inspected
once; no method was redesigned or tuned against it.

## Reproducibility note
The Databricks run is the system of record. A local run on the same data reproduces the production method, all
baselines, Ridge and Random Forest (3m, 6m) exactly. Random Forest 1m differs by < $1 MAE and XGBoost by about $20 at most on test
MAE (multithreaded XGBoost is not bit-reproducible across machines); XGBoost 3m selected a different, near-tied config
(CV MAE within 0.9%). None of this involved the test fold and none changes any conclusion.

## Tables written (Unity Catalog, `workspace.features`)
`hdb_monthly_series`, `series_coverage`, `price_forecasts`, `model_comparison`, `forecast_backtest_metrics`,
`forecast_backtest_predictions`, `forecast_regime_drift`. `price_forecasts` is a full refresh each run (current
forecast only): `forecast_generated_at, last_observed_month, forecast_month, forecast_horizon_months, town, flat_type,
predicted_median_price, forecast_method, transactions_in_window`.

## Run
- Local: `python -m venv .venv && .venv/bin/pip install -r requirements-ml.txt && .venv/bin/python src/run_backtest.py`
  (reads `data/processed/hdb_clean.csv`, writes `data/cache/forecast/`).
- Databricks: `databricks bundle deploy && databricks bundle run flatfair_forecasting`.
