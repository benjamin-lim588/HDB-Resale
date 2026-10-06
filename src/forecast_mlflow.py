"""MLflow logging for the forecasting backtest (Databricks). Kept separate so the modelling code has no MLflow dependency.

Layout: one parent run per backtest; child runs = one per CV candidate (model x config, all horizons/folds)
plus one per model for the final test. Only the Random Forest challenger is registered in Unity Catalog,
tagged challenger / not production. The production method (pooled 3m median) is not an ML model and is not registered.
"""
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

import forecast_data as fd
import forecast_models as fm

FINDING = ("Regime change: median 6-month price drift was +3.1% to +4.3% in every CV window (matching what models "
           "learned from 2017-2025) but -0.4% in the final test window (2025-10..2026-09; YoY fell from ~+6% to ~-1%). "
           "Models trained on earlier growth regimes over-predicted, so no ML model consistently beat the pooled "
           "3-month median out of sample. Production = pooled_3m_median; Random Forest / XGBoost are challengers.")
COLORS = {"production": "#1b6ca8", "challenger": "#e08a1e", "baseline": "#9aa5b1"}


def regime_drift_table(datasets: dict) -> pd.DataFrame:
    rows = []
    for h, ds in datasets.items():
        for f in fm.FOLDS + [fm.TEST_FOLD]:
            tr, ev = fm.split(ds, f)
            rows.append(dict(fold=f["fold"], horizon=h, val_start=f["val_start"], val_end=f["val_end"],
                             train_median_drift_pct=(np.exp(tr["y_rel"].median()) - 1) * 100,
                             realised_eval_drift_pct=(np.exp(ev["y_rel"].median()) - 1) * 100))
    return pd.DataFrame(rows)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def plot_regime(regime, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for h, g in regime.groupby("horizon"):
        ax.plot(g["fold"], g["realised_eval_drift_pct"], marker="o", label=f"realised drift, {h}m horizon")
    ax.axvspan(2.5, 3.5, color="#f3d9d9", alpha=0.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("median price change over horizon (%)")
    ax.set_title("Realised drift by backtest window: the final test window turns flat/negative")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_comparison(comp, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (h, g) in zip(axes, comp.groupby("horizon")):
        g = g.sort_values("test_mae")
        ax.barh(g["model"], g["test_mae"], color=[COLORS[s] for s in g["status"]])
        ax.invert_yaxis()
        ax.set_title(f"{h}-month horizon")
        ax.set_xlabel("final-test MAE ($)")
    fig.suptitle("Final test (2025-10..2026-09): blue = production, orange = ML challenger, grey = baseline")
    return _save(fig, path)


def plot_pred_vs_actual(preds, model, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, (h, g) in zip(axes, preds.groupby("horizon")):
        ax.scatter(g["y"] / 1e3, g["y_pred"] / 1e3, s=6, alpha=0.4)
        lim = [g["y"].min() / 1e3, g["y"].max() / 1e3]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_title(f"{model}, h={h}m")
        ax.set_xlabel("actual median ($k)")
        ax.set_ylabel("predicted ($k)")
    return _save(fig, path)


def plot_residuals(preds, model, path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for j, (h, g) in enumerate(preds.groupby("horizon")):
        pct = (g["y_pred"] - g["y"]) / g["y"] * 100
        axes[0, j].hist(pct.clip(-30, 30), bins=40)
        axes[0, j].axvline(0, color="k", lw=0.8)
        axes[0, j].set_title(f"{model} h={h}m: % error (clipped +-30)")
        m = pct.groupby(g["target_month"]).median()
        axes[1, j].plot(m.index, m.values, marker="o")
        axes[1, j].axhline(0, color="k", lw=0.8)
        axes[1, j].set_title("median % error by target month (bias)")
        axes[1, j].tick_params(axis="x", rotation=45)
    return _save(fig, path)


def plot_importance(pipe, model, h, path, top=15):
    est = pipe.named_steps["m"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    vals = np.abs(est.coef_) if model == "ridge" else est.feature_importances_
    s = pd.Series(vals, index=[n.split("__", 1)[1] for n in names]).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(s.index, s.values)
    ax.set_title(f"{model} h={h}m: top {top} " + ("|coef| (standardised)" if model == "ridge" else "importances"))
    return _save(fig, path)


def _flat(prefix, d):
    return {f"{prefix}{k}": v for k, v in d.items()}


def log_experiment(experiment, catalog, cv, test, selected, datasets, grid, audit, coverage, test_models,
                   metrics_long, comparison, regime, register=True):
    """test_models: {(model, horizon): fitted pipeline} collected from the backtest's test phase."""
    mlflow.set_experiment(experiment)
    tmp = Path(tempfile.mkdtemp())
    params = {(n, c): p for n, c, p, _ in grid}
    folds_str = "; ".join(f"{f['fold']}: train<= {f['train_end'][:7]}, val {f['val_start'][:7]}..{f['val_end'][:7]}"
                          for f in fm.FOLDS)
    common = {"feature_set": "A_origin_features_relative_to_anchor", "target": "log(y_t+h / pooled_3m_median_t)",
              "horizons": ",".join(map(str, fm.HORIZONS)), "cv_folds": folds_str,
              "test_window": f"train<= {fm.TEST_FOLD['train_end'][:7]}, val {fm.TEST_FOLD['val_start'][:7]}..{fm.TEST_FOLD['val_end'][:7]}",
              "min_active_months": fd.MIN_ACTIVE_MONTHS, "min_median_monthly_count": fd.MIN_MEDIAN_COUNT,
              "n_series": int(coverage["eligible_for_forecast"].sum())}
    cv_h = fm.summarise(cv, by=("model", "config", "horizon"))
    cv_f = fm.summarise(cv, by=("model", "config", "fold", "horizon"))
    te_h = fm.summarise(test, by=("model", "config", "horizon"))

    with mlflow.start_run(run_name="flatfair_backtest") as parent:
        mlflow.log_params(common)
        mlflow.log_metrics({"leakage_audit_rows_checked": audit["rows_checked"],
                            "leakage_audit_mismatches": audit["mismatches"]})
        mlflow.set_tags({"production_method": fd.PRODUCTION_METHOD, "finding": FINDING[:5000],
                         "mlflow.note.content": FINDING})
        for name, df in (("model_comparison", comparison), ("metrics_long", metrics_long),
                         ("regime_drift", regime), ("series_coverage", coverage)):
            df.to_csv(tmp / f"{name}.csv", index=False)
            mlflow.log_artifact(str(tmp / f"{name}.csv"))
        mlflow.log_artifact(plot_regime(regime, tmp / "regime_drift.png"))
        mlflow.log_artifact(plot_comparison(comparison, tmp / "model_comparison.png"))

        # ---- CV candidates (every baseline + every hyperparameter config)
        for (model, config), g in cv_h.groupby(["model", "config"]):
            with mlflow.start_run(run_name=f"cv|{model}|{config}", nested=True):
                mlflow.log_params({**common, "model": model, "config": config,
                                   **_flat("hp_", params.get((model, config), {}))})
                mlflow.set_tags({"phase": "cv", "status": fm.STATUS[model]})
                for r in g.itertuples():
                    h = r.horizon
                    mlflow.log_metrics({f"cv_h{h}_mae": r.mae, f"cv_h{h}_rmse": r.rmse, f"cv_h{h}_smape_pct": r.smape_pct,
                                        f"cv_h{h}_mape_pct": r.mape_pct,
                                        f"cv_h{h}_improvement_vs_naive_pct": r.mae_improvement_vs_naive_pct})
                for r in cv_f[(cv_f["model"] == model) & (cv_f["config"] == config)].itertuples():
                    mlflow.log_metrics({f"{r.fold}_h{r.horizon}_mae": r.mae,
                                        f"{r.fold}_h{r.horizon}_improvement_vs_naive_pct": r.mae_improvement_vs_naive_pct})

        # ---- final test: one run per model (ML models use the config selected on CV)
        prod = comparison[comparison["model"] == fm.PRODUCTION].set_index("horizon")["test_mae"]
        for model, g in te_h.groupby("model"):
            sel = {int(r.horizon): r.config for r in g.itertuples()}
            with mlflow.start_run(run_name=f"test|{model}", nested=True):
                mlflow.log_params({**common, "model": model, **{f"h{h}_config": c for h, c in sel.items()}})
                mlflow.set_tags({"phase": "test", "status": fm.STATUS[model]})
                for r in g.itertuples():
                    h = r.horizon
                    mlflow.log_metrics({f"test_h{h}_mae": r.mae, f"test_h{h}_rmse": r.rmse,
                                        f"test_h{h}_smape_pct": r.smape_pct, f"test_h{h}_mape_pct": r.mape_pct,
                                        f"test_h{h}_improvement_vs_naive_pct": r.mae_improvement_vs_naive_pct,
                                        f"test_h{h}_improvement_vs_production_pct": (prod[h] - r.mae) / prod[h] * 100})
                seg = metrics_long[(metrics_long["phase"] == "test") & (metrics_long["model"] == model)]
                seg.to_csv(tmp / f"segments_{model}.csv", index=False)
                mlflow.log_artifact(str(tmp / f"segments_{model}.csv"))
                p = test[test["model"] == model]
                mlflow.log_artifact(plot_pred_vs_actual(p, model, tmp / f"pred_vs_actual_{model}.png"))
                mlflow.log_artifact(plot_residuals(p, model, tmp / f"residuals_{model}.png"))
                for h in fm.HORIZONS:
                    pipe = test_models.get((model, h))
                    if pipe is None:
                        continue
                    mlflow.log_artifact(plot_importance(pipe, model, h, tmp / f"importance_{model}_h{h}.png"))
                    if register and model == "random_forest":
                        _register_challenger(pipe, h, catalog, datasets, g, prod)
    return parent.info.run_id


def _register_challenger(pipe, h, catalog, datasets, te_h, prod):
    """Register the Random Forest as a CHALLENGER only. Its output is log(y/anchor): price = anchor * exp(output)."""
    _, ev = fm.split(datasets[h], fm.TEST_FOLD)
    X = fm.design(ev).head(5)
    name = f"{catalog}.models.flatfair_rf_challenger_h{h}"
    info = mlflow.sklearn.log_model(pipe, name=f"model_h{h}", signature=infer_signature(X, pipe.predict(X)),
                                    input_example=X, registered_model_name=name,
                                    serialization_format="cloudpickle")  # own model; avoids skops type allow-listing
    r = te_h[te_h["horizon"] == h].iloc[0]
    client = mlflow.MlflowClient()
    v = info.registered_model_version
    client.set_registered_model_alias(name, "challenger", v)
    client.update_registered_model(name, description=(
        "ML CHALLENGER (not production). Random Forest for direct h-month-ahead HDB median resale price, "
        "town x flat_type. Output = log(y_{t+h}/anchor_t); price = anchor_t * exp(output), anchor = trailing 3-month "
        "pooled median. Production forecast is the pooled 3-month median because no ML model consistently beat it "
        "in the final out-of-sample window."))
    for k, val in {"status": "challenger", "production": "false", "horizon_months": h,
                   "test_mae": round(float(r.mae), 2),
                   "test_improvement_vs_production_pct": round(float((prod[h] - r.mae) / prod[h] * 100), 2)}.items():
        client.set_model_version_tag(name, v, k, str(val))
