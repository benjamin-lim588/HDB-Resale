"""Fair-value estimator: baseline vs hedonic Ridge vs Random Forest vs XGBoost, chronological CV + untouched holdout.

Flow (fixed in advance): (1) CORE features, tune on cv1-cv3 only; (2) pick each family's best config by pooled CV MAE;
(3) ablations: core_year (CV only), core_access (CV + holdout, reported SEPARATELY, never used for selection);
(4) evaluate the CV-selected CORE models on the final holdout ONCE; (5) apply the pre-registered champion rule.

Local:       .venv/bin/python src/run_fair_value.py [--cv-only]     (CSV in, data/cache/fair_value/ out)
Databricks:  jobs/07_fair_value.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fair_value_data as fv
import fair_value_models as fm
from paths import CACHE, PROCESSED


def run_cv(d, grid, feature_set="core", configs=None, on_result=None):
    """configs: optional {family: config_id} to restrict to the selected config (ablations)."""
    out = []
    for fold in fm.FOLDS:
        tr, ev = fm.split(d, fold)
        out.append(fm.baseline_predictions(ev, fold))
        for fam, cfg, params, factory in grid:
            if configs and configs.get(fam) != cfg:
                continue
            m, p = fm.model_predictions(fam, cfg, factory, tr, ev, fold, feature_set)
            out.append(p)
            if on_result:
                on_result("cv", fam, cfg, params, fold, feature_set, p, m)
    return pd.concat(out, ignore_index=True)


def select_configs(cv: pd.DataFrame) -> dict:
    s = fm.summarise(cv, ["model", "config"])
    s = s[s["model"] != fm.BASELINE]
    return s.loc[s.groupby("model")["mae"].idxmin()].set_index("model")["config"].to_dict()


def run_test(d, grid, configs, feature_set="core", on_result=None):
    fold = fm.TEST_FOLD
    tr, ev = fm.split(d, fold)
    out, models = [fm.baseline_predictions(ev, fold)], {}
    for fam, cfg, params, factory in grid:
        if configs.get(fam) != cfg:
            continue
        m, p = fm.model_predictions(fam, cfg, factory, tr, ev, fold, feature_set)
        out.append(p)
        models[fam] = m
        if on_result:
            on_result("test", fam, cfg, params, fold, feature_set, p, m)
    return pd.concat(out, ignore_index=True), models


def run_all(d, with_test=True, on_result=None, log=print):
    """Returns dict with cv/test predictions, selection, ablations, champion decision."""
    grid = fm.model_grid("core")
    t = time.time()
    cv = run_cv(d, grid, on_result=on_result)
    sel = select_configs(cv)
    log(f"CV done in {time.time() - t:.0f}s; selected: {sel}")
    res = {"cv": cv, "selected": sel}
    # ablations use the CORE-selected configs (no extra tuning)
    for fs in ("core_year", "core_access"):
        g = fm.model_grid(fs)
        res[f"cv_{fs}"] = run_cv(d, g, fs, configs=sel, on_result=on_result)
    res["test"], res["test_models"] = None, {}
    if with_test:
        res["test"], res["test_models"] = run_test(d, grid, sel, on_result=on_result)   # the ONE look at the holdout
        for fs in ("core_year", "core_access"):      # ablations, scored in the same single pass, never used for selection
            res[f"test_{fs}"], res[f"test_models_{fs}"] = run_test(d, fm.model_grid(fs), sel, fs, on_result)
        cvs = fm.summarise(cv, ["model", "config"])
        cvb = cvs[(cvs["model"] == fm.BASELINE) | cvs.apply(lambda r: sel.get(r["model"]) == r["config"], axis=1)]
        tes = fm.summarise(res["test"], ["model", "config"])
        res["decision"] = fm.choose_champion(cvb, tes)
    return res


def comparison_table(res, d) -> pd.DataFrame:
    """One row per model x feature set: pooled CV + holdout metrics and status."""
    sel = res["selected"]
    dec = res.get("decision")
    rows = []
    sets = {"core": (res["cv"], res["test"]), "core_year": (res["cv_core_year"], res.get("test_core_year")),
            "core_access": (res["cv_core_access"], res.get("test_core_access"))}
    for fs, (cv, te) in sets.items():
        cvs = fm.summarise(cv, ["model", "config", "feature_set"])
        cvs = cvs[(cvs["model"] == fm.BASELINE) | cvs.apply(lambda r: sel.get(r["model"]) == r["config"], axis=1)]
        folds = fm.summarise(cv, ["model", "config", "feature_set", "fold"]).pivot_table(
            index=["model", "config", "feature_set"], columns="fold", values="mae").add_prefix("cv_mae_").reset_index()
        cvs = cvs.merge(folds, on=["model", "config", "feature_set"], how="left").add_prefix("cv_").rename(
            columns={"cv_model": "model", "cv_config": "config", "cv_feature_set": "feature_set"})
        cvs = cvs.rename(columns={c: c.replace("cv_cv_mae_", "cv_mae_") for c in cvs.columns})
        if te is not None:
            ts = fm.summarise(te, ["model", "config", "feature_set"]).add_prefix("test_").rename(
                columns={"test_model": "model", "test_config": "config", "test_feature_set": "feature_set"})
            cvs = cvs.merge(ts, on=["model", "config", "feature_set"], how="left")
        rows.append(cvs)
    out = pd.concat(rows, ignore_index=True).drop_duplicates(["model", "config", "feature_set"]).reset_index(drop=True)
    out["status"] = np.where(out["feature_set"] != "core", "ablation: " + out["feature_set"],
                             np.where(out["model"] == fm.BASELINE, "baseline", "challenger"))
    if dec:
        out.loc[(out["feature_set"] == "core") & (out["model"] == dec["champion"]), "status"] = "champion"
        if dec["champion"] == fm.BASELINE:
            out.loc[(out["feature_set"] == "core") & (out["model"] == fm.BASELINE), "status"] = "champion (baseline recommended)"
    return out


def predictions_wide(res, d) -> pd.DataFrame:
    """Compact transaction-level predictions for the CORE selected models: cv (pooled folds) + test."""
    sel = res["selected"]
    frames = []
    for phase, p in (("cv", res["cv"]), ("test", res["test"])):
        if p is None:
            continue
        p = p[(p["model"] == fm.BASELINE) | p.apply(lambda r: sel.get(r["model"]) == r["config"], axis=1)]
        w = p.pivot_table(index=["txn_id", "fold"], columns="model", values="y_pred").add_prefix("pred_").reset_index()
        base = p[p["model"] == fm.BASELINE][["txn_id", "fold", "month", "town", "flat_type", "resale_price", "anchor"]]
        frames.append(base.merge(w, on=["txn_id", "fold"]).assign(phase=phase))
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-only", action="store_true", help="development: never touch the final holdout")
    a = ap.parse_args()
    hdb = pd.read_csv(PROCESSED / "hdb_clean.csv", parse_dates=["transaction_date"])
    acc = pd.read_csv(PROCESSED / "hdb_accessibility_features.csv")
    d, idx, info = fv.build_dataset(hdb, acc)
    audit = fv.audit_market_features(fv.prepare_transactions(hdb), d)
    print(info, "| audit", audit)
    res = run_all(d, with_test=not a.cv_only)
    pd.set_option("display.width", 250)
    cols = ["model", "config", "feature_set", "n", "mae", "rmse", "medae", "smape_pct", "within_5pct", "within_10pct", "within_15pct", "mae_improvement_vs_baseline_pct"]
    for k in ("cv", "cv_core_year", "cv_core_access"):
        s = fm.summarise(res[k], ["model", "config", "feature_set"])
        if k == "cv":
            print("\n== CORE CV, all configs (pooled cv1-cv3) =="); print(s.sort_values("mae")[cols].round(2).to_string(index=False))
        else:
            sel = s[s.apply(lambda r: res["selected"].get(r["model"]) == r["config"], axis=1)]
            print(f"\n== ABLATION {k} CV (CORE-selected configs) =="); print(sel[cols].round(2).to_string(index=False))
    out = CACHE / "fair_value"
    out.mkdir(parents=True, exist_ok=True)
    if not a.cv_only:
        print("\n== FINAL HOLDOUT, CORE (CV-selected configs, scored once) ==")
        print(fm.summarise(res["test"], ["model", "config", "feature_set"]).sort_values("mae")[cols].round(2).to_string(index=False))
        for fs in ("core_year", "core_access"):
            print(f"\n== FINAL HOLDOUT, ablation {fs} (reported separately) ==")
            print(fm.summarise(res[f"test_{fs}"], ["model", "config", "feature_set"]).sort_values("mae")[cols].round(2).to_string(index=False))
        print("\nDECISION:", res["decision"])
        comparison_table(res, d).to_csv(out / "model_comparison.csv", index=False)
        predictions_wide(res, d).to_csv(out / "predictions.csv", index=False)
    res["cv"].to_csv(out / "cv_predictions.csv", index=False)


if __name__ == "__main__":
    main()


def build_tables(res, d, idx, champion_family=None) -> dict:
    """Compact Unity Catalog outputs (name -> DataFrame)."""
    sel = res["selected"]
    keep = lambda p: p[(p["model"] == fm.BASELINE) | p.apply(lambda r: sel.get(r["model"]) == r["config"], axis=1)]
    mets = [fm.segment_metrics(keep(res["cv"]), d, "cv_pooled")]
    for fs in ("core_year", "core_access"):
        m_ = fm.segment_metrics(keep(res[f"cv_{fs}"]), d, "cv_pooled")
        mets.append(m_[m_["model"] != fm.BASELINE])
    if res["test"] is not None:
        mets.append(fm.segment_metrics(res["test"], d, "test"))
        for fs in ("core_year", "core_access"):
            m_ = fm.segment_metrics(res[f"test_{fs}"], d, "test")
            mets.append(m_[m_["model"] != fm.BASELINE])
    metrics = pd.concat(mets, ignore_index=True)
    last = int(d["m"].max()) + 1                        # valuation month = month after the latest complete month
    anchors = idx.table(range(last, last + 1))
    anchors["valuation_month"] = pd.Timestamp(year=last // 12, month=last % 12 + 1, day=1)
    anchors["latest_complete_market_month"] = d["month"].max()
    out = {"fair_value_model_comparison": comparison_table(res, d), "fair_value_backtest_metrics": metrics,
           "fair_value_backtest_predictions": predictions_wide(res, d),
           "fair_value_market_anchors": anchors[["valuation_month", "latest_complete_market_month", "town", "flat_type", "anchor",
                                                 "anchor_n", "anchor_tier", "anchor_change_3m", "prior6_vs_anchor",
                                                 "log_prior3_volume"]]}
    return out
