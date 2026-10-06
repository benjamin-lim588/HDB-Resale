"""Chronological backtest of baselines + Ridge / RandomForest / XGBoost for town x flat_type price forecasts.

Phase 1 (model selection): expanding-window folds cv1-cv3, small fixed hyperparameter grids.
Phase 2 (final test): the best config per model/horizon is refit through 2025-09 and scored ONCE on
2025-10..2026-09. Nothing from the test window is used in phase 1.

Usage (local, CSV):  .venv/bin/python src/run_backtest.py
Databricks:          jobs/05_backtest.py calls backtest() with an MLflow `on_result` hook.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecast_data as fd
import forecast_models as fm
from paths import CACHE, PROCESSED


def inputs_from_tables(monthly: pd.DataFrame, coverage: pd.DataFrame):
    """Panel + audited leakage-safe datasets from the persisted monthly / coverage tables."""
    months = sorted(monthly["month"].unique())
    panel = fd.build_panel(monthly, coverage, months)
    audit = fd.audit_features(panel)
    datasets = {h: fd.make_dataset(panel, h) for h in fm.HORIZONS}
    return months, panel, audit, datasets


def build_inputs(hdb: pd.DataFrame):
    months = fd.complete_months(hdb)
    monthly = fd.monthly_series(hdb, months)
    coverage = fd.series_coverage(monthly, months)
    monthly = monthly.merge(coverage[fd.KEYS + ["eligible_for_forecast"]], on=fd.KEYS)
    _, panel, audit, datasets = inputs_from_tables(monthly, coverage)
    return months, monthly, coverage, panel, audit, datasets


def backtest(datasets: dict, on_result=None, grid=None):
    """Returns (cv_preds, test_preds, selected). on_result(phase, model, config, params, fold, h, preds, fitted)
    is called for every evaluated candidate (used for MLflow logging); baselines have params={}."""
    grid = grid or fm.model_grid()
    emit = on_result or (lambda *a, **k: None)
    cv = []
    for h, ds in datasets.items():
        for fold in fm.FOLDS:
            tr, ev = fm.split(ds, fold)
            b = fm.baseline_predictions(ev, fold, tr)
            cv.append(b)
            for name in b["model"].unique():
                emit("cv", name, "-", {}, fold, h, b[b["model"] == name], None)
            for name, cfg, params, factory in grid:
                m, p = fm.model_predictions(name, cfg, factory, tr, ev, fold)
                cv.append(p)
                emit("cv", name, cfg, params, fold, h, p, m)
    cv = pd.concat(cv, ignore_index=True)

    # select the config with the lowest pooled cv MAE per model x horizon
    s = fm.summarise(cv, by=("model", "config", "horizon"))
    ml = s[~s["model"].isin(list(fm.BASELINES) + [fm.DRIFT])]
    selected = ml.loc[ml.groupby(["model", "horizon"])["mae"].idxmin(), ["model", "config", "horizon"]]

    test = []
    fold = fm.TEST_FOLD
    for h, ds in datasets.items():
        tr, ev = fm.split(ds, fold)
        b = fm.baseline_predictions(ev, fold, tr)
        test.append(b)
        for name in b["model"].unique():
            emit("test", name, "-", {}, fold, h, b[b["model"] == name], None)
        for r in selected[selected["horizon"] == h].itertuples():
            name, cfg, params, factory = next(g for g in grid if g[0] == r.model and g[1] == r.config)
            m, p = fm.model_predictions(name, cfg, factory, tr, ev, fold)
            test.append(p)
            emit("test", name, cfg, params, fold, h, p, m)
    return cv, pd.concat(test, ignore_index=True), selected.reset_index(drop=True)


def main():
    hdb = pd.read_csv(PROCESSED / "hdb_clean.csv", parse_dates=["transaction_date"])
    months, monthly, coverage, panel, audit, datasets = build_inputs(hdb)
    print("months", months[0].date(), "->", months[-1].date(), "| eligible series", int(coverage.eligible_for_forecast.sum()),
          "| leakage audit", audit)
    cv, test, selected = backtest(datasets)
    out = CACHE / "forecast"
    out.mkdir(parents=True, exist_ok=True)
    cv.to_csv(out / "cv_predictions.csv", index=False)
    test.to_csv(out / "test_predictions.csv", index=False)
    selected.to_csv(out / "selected_configs.csv", index=False)
    fm.metrics_long(cv, test).to_csv(out / "metrics_long.csv", index=False)
    comp = fm.comparison_table(cv, test, selected)
    comp.to_csv(out / "model_comparison.csv", index=False)
    fd.production_forecast(panel).to_csv(out / "price_forecasts.csv", index=False)
    pd.set_option("display.width", 220)
    print(comp.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
