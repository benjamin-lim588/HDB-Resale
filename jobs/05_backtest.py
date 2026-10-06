"""Task 5: chronological backtest of all candidates, logged to MLflow; comparison tables to Delta.
Plain run of the same code as src/run_backtest.py. Usage: 05_backtest.py --catalog workspace --experiment /Users/<you>/flatfair-price-forecasting"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# __file__ is undefined when Databricks exec()s a script task; argv[0] holds the script path
sys.path.insert(0, str(Path(globals().get("__file__") or sys.argv[0]).resolve().parents[1] / "src"))
import databricks_io as dio
import forecast_data as fd
import forecast_models as fm
import forecast_mlflow as fmf
from run_backtest import backtest, inputs_from_tables

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=dio.DEFAULT_CATALOG)
    ap.add_argument("--experiment", required=True)
    a = ap.parse_args()
    monthly = dio.read_table(a.catalog, "features", "hdb_monthly_series")
    coverage = dio.read_table(a.catalog, "features", "series_coverage")
    months, panel, audit, datasets = inputs_from_tables(monthly, coverage)
    print("leakage audit:", audit)

    test_models = {}
    def hook(phase, model, config, params, fold, h, preds, fitted):
        if phase == "test" and fitted is not None:
            test_models[(model, h)] = fitted

    cv, test, selected = backtest(datasets, on_result=hook)
    mlong, comp, regime = fm.metrics_long(cv, test), fm.comparison_table(cv, test, selected), fmf.regime_drift_table(datasets)

    # predictions kept for dashboards/diagnostics: all baselines + the selected ML configs (cv) and the test fold
    keep = cv.merge(selected.assign(_sel=1), on=["model", "config", "horizon"], how="left")
    cv_keep = cv[keep["_sel"].eq(1).to_numpy() | cv["model"].isin(list(fm.BASELINES) + [fm.DRIFT]).to_numpy()]
    preds = pd.concat([cv_keep.assign(phase="cv"), test.assign(phase="test")], ignore_index=True)
    for name, df in (("forecast_backtest_metrics", mlong), ("model_comparison", comp),
                     ("forecast_regime_drift", regime), ("forecast_backtest_predictions", preds)):
        dio.write_table(df, a.catalog, "features", name)

    run_id = fmf.log_experiment(a.experiment, a.catalog, cv, test, selected, datasets, fm.model_grid(), audit,
                                coverage, test_models, mlong, comp, regime)
    print("MLflow parent run:", run_id)
