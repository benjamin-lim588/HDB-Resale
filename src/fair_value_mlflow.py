"""MLflow tracking for the FAIR-VALUE experiments (Databricks / any MLflow tracking URI).

Parent run + one child per CV candidate + one per family on the final holdout + ablation children. Only the CHAMPION
(if it is not the baseline) is registered in Unity Catalog; challengers are tracked, never registered.
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

import fair_value_data as fv
import fair_value_models as fm

COLORS = {"champion": "#1b6ca8", "challenger": "#e08a1e", "baseline": "#9aa5b1"}
METRIC_KEYS = ["mae", "rmse", "medae", "smape_pct", "mape_pct", "within_5pct", "within_10pct", "within_15pct",
               "mae_improvement_vs_baseline_pct", "medae_improvement_vs_baseline_pct"]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def _metrics(prefix, row):
    return {f"{prefix}_{k}": float(row[k]) for k in METRIC_KEYS if k in row and row[k] == row[k]}


def plot_comparison(comp, path):
    c = comp[comp["feature_set"] == "core"].copy()
    c["status_key"] = np.where(c["status"].str.startswith("champion"), "champion", np.where(c["model"] == fm.BASELINE, "baseline", "challenger"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in ((axes[0], "cv_mae", "pooled CV MAE ($)"), (axes[1], "test_mae", "final holdout MAE ($)")):
        if col not in c or c[col].isna().all():
            continue
        g = c.sort_values(col)
        ax.barh(g["model"], g[col], color=[COLORS[s] for s in g["status_key"]])
        ax.invert_yaxis()
        ax.set_title(title)
    fig.suptitle("Fair value: CORE features (blue = champion, orange = challenger, grey = baseline)")
    return _save(fig, path)


def plot_pred_vs_actual(p, name, path):
    s = p.sample(min(6000, len(p)), random_state=0)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(s["resale_price"] / 1e3, s["y_pred"] / 1e3, s=4, alpha=0.3)
    lim = [s["resale_price"].min() / 1e3, s["resale_price"].max() / 1e3]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("actual resale price ($k)")
    ax.set_ylabel("estimated fair value ($k)")
    ax.set_title(f"{name}: holdout estimate vs actual")
    return _save(fig, path)


def plot_residuals(p, name, path):
    pct = (p["y_pred"] - p["resale_price"]) / p["resale_price"] * 100
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(pct.clip(-40, 40), bins=60)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_title(f"{name}: % error (clipped +-40)")
    m = pct.groupby(p["month"]).median()
    axes[1].plot(m.index, m.values, marker="o")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("median % error by month (bias)")
    axes[1].tick_params(axis="x", rotation=45)
    band = pd.cut(p["resale_price"], [0, 400e3, 500e3, 650e3, 800e3, 1e6, 1e9], labels=["<400k", "400-500k", "500-650k", "650-800k", "800k-1m", ">=1m"])
    pct.abs().groupby(band, observed=True).median().plot.bar(ax=axes[2])
    axes[2].set_title("median |% error| by price band")
    return _save(fig, path)


def plot_importance(imp, name, path, top=15):
    s = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(s["feature"], s["importance"])
    ax.set_title(f"{name}: feature importance (one-hot summed to variable)")
    return _save(fig, path)


def shap_artifacts(pipe, X_eval, name, tmp, n=1000):
    """Optional attribution on a sample of holdout rows. Uses the `shap` package when installed (local); otherwise XGBoost's
    built-in per-feature contributions (identical TreeSHAP values, no extra dependency). Random Forest needs `shap`.
    Failure never breaks the experiment."""
    try:
        Xs = X_eval.sample(min(n, len(X_eval)), random_state=0)
        Z = pd.DataFrame(pipe.named_steps["prep"].transform(Xs), columns=fm.pipeline_feature_names(pipe))
        est = pipe.named_steps["m"]
        try:
            import shap
            sv = shap.TreeExplainer(est).shap_values(Z)
            shap.summary_plot(sv, Z, show=False, max_display=15)
            paths = [_save(plt.gcf(), tmp / f"shap_summary_{name}.png")]
        except ImportError:
            if name != "xgboost":
                print(f"SHAP skipped for {name}: `shap` not installed")
                return []
            import xgboost as xgb
            sv = est.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)[:, :-1]      # last column = bias
            imp = pd.DataFrame({"feature": Z.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
            s = imp.sort_values("mean_abs_shap").tail(15)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(s["feature"], s["mean_abs_shap"])
            ax.set_title(f"{name}: mean |SHAP| (XGBoost built-in contributions, holdout sample)")
            paths = [_save(fig, tmp / f"shap_summary_{name}.png")]
        imp = pd.DataFrame({"feature": Z.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
        imp.to_csv(tmp / f"shap_importance_{name}.csv", index=False)
        return paths + [str(tmp / f"shap_importance_{name}.csv")]
    except Exception as e:  # noqa: BLE001
        print(f"SHAP skipped for {name}: {e}")
        return []


def log_experiment(experiment, catalog, res, d, tables, audit, register=True):
    """res: output of run_fair_value.run_all(with_test=True); tables: build_tables(...)."""
    mlflow.set_experiment(experiment)
    tmp = Path(tempfile.mkdtemp())
    sel, dec = res["selected"], res["decision"]
    comp, seg = tables["fair_value_model_comparison"], tables["fair_value_backtest_metrics"]
    common = {"target": "log(resale_price / anchor)", "anchor": "pooled median of town x flat_type sales in V-3..V-1 (widened if <10)",
              "valuation_convention": "sale in month V uses only months < V",
              "cv_folds": "; ".join(f"{f['fold']}: train<={f['train_end'][:7]}, val {f['val_start'][:7]}..{f['val_end'][:7]}" for f in fm.FOLDS),
              "test_window": f"train<={fm.TEST_FOLD['train_end'][:7]}, val {fm.TEST_FOLD['val_start'][:7]}..{fm.TEST_FOLD['val_end'][:7]}",
              "n_rows_dataset": len(d)}
    grid = {(f, c): p for f, c, p, _ in fm.model_grid("core")}
    cv_core = fm.summarise(res["cv"], ["model", "config", "feature_set"])
    cv_fold = fm.summarise(res["cv"], ["model", "config", "feature_set", "fold"])
    with mlflow.start_run(run_name="flatfair_fair_value") as parent:
        mlflow.log_params(common)
        mlflow.log_metrics({"leakage_audit_rows_checked": audit["rows_checked"], "leakage_audit_mismatches": audit["mismatches"]})
        mlflow.set_tags({"cv_winner": dec["cv_winner"], "champion": dec["champion"],
                         "confirmed_on_holdout": str(dec["confirmed_on_holdout"]),
                         "champion_rule": "CV winner (ML must beat hedonic by >1% CV MAE); confirm vs baseline on holdout; holdout never chooses"})
        for name, df in (("model_comparison", comp), ("backtest_metrics", seg)):
            df.to_csv(tmp / f"{name}.csv", index=False)
            mlflow.log_artifact(str(tmp / f"{name}.csv"))
        mlflow.log_artifact(plot_comparison(comp, tmp / "model_comparison.png"))

        # ---- every CORE CV candidate (all configs)
        for r in cv_core.itertuples():
            with mlflow.start_run(run_name=f"cv|{r.model}|{r.config}", nested=True):
                mlflow.log_params({**common, "model": r.model, "config": r.config, "feature_set": r.feature_set,
                                   **{f"hp_{k}": v for k, v in grid.get((r.model, r.config), {}).items()}})
                mlflow.set_tags({"phase": "cv", "selected_config": str(sel.get(r.model) == r.config)})
                mlflow.log_metrics(_metrics("cv", r._asdict()))
                for f in cv_fold[(cv_fold["model"] == r.model) & (cv_fold["config"] == r.config)].itertuples():
                    mlflow.log_metric(f"{f.fold}_mae", f.mae)

        # ---- ablations (CORE-selected configs)
        for fs in ("core_year", "core_access"):
            cvs = fm.summarise(res[f"cv_{fs}"], ["model", "config", "feature_set"])
            tes = fm.summarise(res[f"test_{fs}"], ["model", "config", "feature_set"]).set_index("model")
            for r in cvs[cvs["model"] != fm.BASELINE].itertuples():
                with mlflow.start_run(run_name=f"ablation|{fs}|{r.model}", nested=True):
                    mlflow.log_params({**common, "model": r.model, "config": r.config, "feature_set": fs})
                    mlflow.set_tags({"phase": "ablation", "not_used_for_selection": "true"})
                    mlflow.log_metrics(_metrics("cv", r._asdict()))
                    if r.model in tes.index:
                        mlflow.log_metrics(_metrics("test", tes.loc[r.model].to_dict()))

        # ---- final holdout: one run per CORE model (CV-selected config), baseline included
        te = fm.summarise(res["test"], ["model", "config", "feature_set"])
        seg_test = seg[(seg["phase"] == "test") & (seg["feature_set"] == "core")]
        for r in te.itertuples():
            fam = r.model
            status = "champion" if fam == dec["champion"] else ("baseline" if fam == fm.BASELINE else "challenger")
            with mlflow.start_run(run_name=f"test|{fam}", nested=True):
                mlflow.log_params({**common, "model": fam, "config": r.config, "feature_set": "core",
                                   **{f"hp_{k}": v for k, v in grid.get((fam, r.config), {}).items()}})
                mlflow.set_tags({"phase": "test", "status": status})
                mlflow.log_metrics(_metrics("test", r._asdict()))
                s = seg_test[seg_test["model"] == fam]
                s.to_csv(tmp / f"segments_{fam}.csv", index=False)
                mlflow.log_artifact(str(tmp / f"segments_{fam}.csv"))
                p = res["test"][res["test"]["model"] == fam]
                mlflow.log_artifact(plot_pred_vs_actual(p, fam, tmp / f"pred_vs_actual_{fam}.png"))
                mlflow.log_artifact(plot_residuals(p, fam, tmp / f"residuals_{fam}.png"))
                pipe = res["test_models"].get(fam)
                if pipe is None:
                    continue
                _, ev = fm.split(d, fm.TEST_FOLD)
                if fam == "hedonic_ridge":
                    eff = fm.hedonic_effects(pipe)
                    eff.to_csv(tmp / "hedonic_effects.csv", index=False)
                    mlflow.log_artifact(str(tmp / "hedonic_effects.csv"))
                else:
                    imp = fm.grouped_importance(pipe)
                    imp.to_csv(tmp / f"importance_{fam}.csv", index=False)
                    mlflow.log_artifact(str(tmp / f"importance_{fam}.csv"))
                    mlflow.log_artifact(plot_importance(imp, fam, tmp / f"importance_{fam}.png"))
                    for path in shap_artifacts(pipe, fm.design(ev, "core"), fam, tmp):
                        mlflow.log_artifact(path)
                if register and fam == dec["champion"] and fam != fm.BASELINE:
                    X = fm.design(ev, "core").head(5)
                    name = f"{catalog}.models.flatfair_fair_value_{fam}"
                    info = mlflow.sklearn.log_model(pipe, name="model", signature=infer_signature(X, pipe.predict(X)), input_example=X,
                                                    registered_model_name=name, serialization_format="cloudpickle")
                    c = mlflow.MlflowClient()
                    v = info.registered_model_version
                    c.set_registered_model_alias(name, "champion", v)
                    c.update_registered_model(name, description=(
                        "FAIR-VALUE CHAMPION: estimates what a specific HDB flat is worth at a valuation month. Output = log(price/anchor); "
                        "fair value = anchor * exp(output), anchor = pooled median of the town x flat_type sales in the 3 prior complete months "
                        "(widened if thin). Features exclude accessibility (CORE). Not the market forecast and not the comparables engine."))
                    for k, val in {"status": "champion", "feature_set": "core", "config": r.config, "test_mae": round(float(r.mae), 2),
                                   "test_within_10pct": round(float(r.within_10pct), 2),
                                   "test_mae_improvement_vs_baseline_pct": round(float(r.mae_improvement_vs_baseline_pct), 2)}.items():
                        c.set_model_version_tag(name, v, k, str(val))
    return parent.info.run_id
