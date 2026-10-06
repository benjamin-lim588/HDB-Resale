"""Historical evaluation of the COMPARABLES engine and selection of ONE canonical methodology (CV only).

For each sampled target transaction: hide its price, use only sales from months strictly before its month, retrieve the
top-K by similarity, take the median comparable price ("comparable market reference") and compare it with the actual price.
Compared against: recent town x flat_type median (the fair-value baseline) and town x flat_type x flat_model 12-month median.

Selection rule (fixed in advance, CV only): lowest pooled-CV MAE over (method, K) on the COMMON target set where every
method returns >= 3 comparables; a transparent method (equal-weight or hedonic-weighted distance, or Ben's reference
weights) replaces a KNN winner if within 1% of its MAE; K = lowest MAE (ties within 0.5% -> most stable across folds).
The final holdout is scored once for the chosen methodology only.

Local: .venv/bin/python src/run_comparables_evaluation.py [--cv-only]      Databricks: jobs/08_comparables_eval.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparables as cp
import fair_value_data as fv
import fair_value_models as fm
from paths import CACHE, PROCESSED

KS = (3, 5, 10)
N_TARGETS_CV, N_TARGETS_TEST = 2000, 3000
TRANSPARENT = {"equal_l1", "hedonic", "ben"}
SELECTION_TOL_PCT, K_TIE_PCT = 1.0, 0.5
FOLDS, TEST_FOLD = fm.FOLDS, fm.TEST_FOLD


def month_idx(s: str) -> int:
    t = pd.Timestamp(s)
    return t.year * 12 + t.month - 1


def variant_list() -> list:
    v = []
    for limits in ("ben", "loose"):
        v.append(dict(kind="knn", limits=limits, recency=0.0))
        v.append(dict(kind="equal_l1", limits=limits, recency=0.0))
        for r in (0.0, 0.001, 0.003, 0.005, 0.01):
            v.append(dict(kind="hedonic", limits=limits, recency=r))
    v.append(dict(kind="ben", limits="ben", recency=0.0))
    for x in v:
        x["name"] = f"{x['kind']}|{x['limits']}|rec={x['recency']}"
    return v


def to_query(r) -> dict:
    return dict(town=r.town, flat_type=r.flat_type, area=float(r.floor_area_sqm), storey=float(r.storey_mid),
                lease_months=float(r.remaining_lease_months), model=r.flat_model,
                log_mrt=None if r.log_mrt_dist != r.log_mrt_dist else float(r.log_mrt_dist))


def run_targets(pool, targets, cfg, prices):
    """Per target: reference median for each K, candidates, window, mean distance of the top-K."""
    rows = []
    for r in targets.itertuples(index=False):
        rk = cp.rank(pool, cfg, to_query(r), int(r.m), top=max(KS))
        n = len(rk["ix"])
        row = dict(txn_id=r.txn_id, n_candidates=rk["n_candidates"], window=rk["window_used"] or np.nan, k_found=n)
        for k in KS:
            if n >= cp.MIN_COMPARABLES:
                kk = min(k, n)
                row[f"ref_k{k}"] = float(np.median(prices[rk["ix"][:kk]]))
                row[f"dist_k{k}"] = float(rk["dist"][:kk].mean())
            else:
                row[f"ref_k{k}"], row[f"dist_k{k}"] = np.nan, np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def flat_model_median(pool, targets, months=12, min_n=10):
    """Baseline: town x flat_type x flat_model median of sales in the prior `months` months (NaN if < min_n)."""
    g = {k: (v["m"].to_numpy(), v["resale_price"].to_numpy()) for k, v in pool.df.groupby(["town", "flat_type", "flat_model"])}
    out = []
    for r in targets.itertuples(index=False):
        mm, pp = g.get((r.town, r.flat_type, r.flat_model), (np.empty(0), np.empty(0)))
        sel = (mm >= r.m - months) & (mm < r.m)
        out.append(float(np.median(pp[sel])) if sel.sum() >= min_n else np.nan)
    return np.array(out)


def evaluate_fold(pool, d, fold, variants, n_targets, seed=0, access_for=None):
    tr, ev = fm.split(d, fold)
    targets = ev.sample(min(n_targets, len(ev)), random_state=seed).sort_values("txn_id")
    scales = pool.scales(month_idx(fold["train_end"]))
    hed = cp.hedonic_weights(tr, "core")
    hed_access = cp.hedonic_weights(tr, "core_access") if access_for else None
    prices = pool.df["resale_price"].to_numpy()
    res = {}
    for v in variants:
        cfg = cp.make_config(v["kind"], v["limits"], v["recency"], False, scales, hed)
        res[v["name"]] = run_targets(pool, targets, cfg, prices)
    for v in (access_for or []):      # accessibility ablation: same method + accessibility term in the distance
        cfg = cp.make_config(v["kind"], v["limits"], v["recency"], True, scales, hed_access)
        res[v["name"] + "+access"] = run_targets(pool, targets, cfg, prices)
    return targets, res, dict(scales=scales, hedonic=hed, hedonic_access=hed_access)


def predictions_frame(targets, res, fold, pool):
    base = targets[fm.ID_COLS].copy()
    base["feature_set"], base["fold"] = "core", fold["fold"]
    frames = [base.assign(y_pred=targets["anchor"].to_numpy(), model=fm.BASELINE, config="-")]
    fmm = flat_model_median(pool, targets)
    frames.append(base.assign(y_pred=fmm, model="flat_model_median_12m", config="-"))
    for name, r in res.items():
        r = r.set_index("txn_id").reindex(targets["txn_id"])
        for k in KS:
            frames.append(base.assign(y_pred=r[f"ref_k{k}"].to_numpy(), model=f"cmp|{name}|K={k}", config=name))
    out = pd.concat(frames, ignore_index=True)
    meta = pd.concat([r.assign(variant=n) for n, r in res.items()], ignore_index=True)
    return out, meta


def metrics_on_common(preds: pd.DataFrame, by) -> pd.DataFrame:
    """Common set: targets for which EVERY comparables method returned a value. flat_model_median is NaN where fewer than
    10 prior sales exist, so it is evaluated on its own available rows against the baseline on those same rows."""
    cmp_ = preds[preds["model"].str.startswith("cmp|")]
    bad = set(cmp_.loc[cmp_["y_pred"].isna(), "txn_id"])
    keep = preds[~preds["txn_id"].isin(bad) & (preds["model"] != "flat_model_median_12m")]
    s = fm.summarise(keep, by)
    ids = set(preds.loc[(preds["model"] == "flat_model_median_12m") & preds["y_pred"].notna(), "txn_id"])
    sub = preds[preds["txn_id"].isin(ids) & preds["model"].isin([fm.BASELINE, "flat_model_median_12m"])]
    s2 = fm.summarise(sub, by)
    return pd.concat([s, s2[s2["model"] == "flat_model_median_12m"]], ignore_index=True)


def choose(cv_summary: pd.DataFrame, fold_mae: pd.DataFrame) -> dict:
    """cv_summary: pooled metrics rows for cmp models (model=cmp|variant|K=k, config=variant)."""
    s = cv_summary[cv_summary["model"].str.startswith("cmp|") & ~cv_summary["model"].str.contains(r"\+access")].copy()
    s["variant"] = s["config"]
    s["K"] = s["model"].str.extract(r"K=(\d+)").astype(int)
    s["kind"] = s["variant"].str.split("|").str[0]
    best = s.loc[s["mae"].idxmin()]
    tr = s[s["kind"].isin(TRANSPARENT)]
    pick = tr.loc[tr["mae"].idxmin()] if (len(tr) and tr["mae"].min() <= best["mae"] * (1 + SELECTION_TOL_PCT / 100)) else best
    v = s[s["variant"] == pick["variant"]]
    near = v[v["mae"] <= v["mae"].min() * (1 + K_TIE_PCT / 100)]
    sd = fold_mae[fold_mae["config"] == pick["variant"]].groupby("K")["mae"].std()
    kbest = int(min(near["K"], key=lambda k: (sd.get(k, np.inf), k)))
    return {"variant": pick["variant"], "K": kbest, "overall_best": f"{best['variant']} K={int(best['K'])}", "kind": pick["kind"]}


def quality_thresholds(meta_choice: pd.DataFrame, k: int) -> dict:
    """High/medium distance cut-offs = CV terciles of the mean top-K distance; HIGH also needs a 12-month window and >=20 candidates."""
    dist = meta_choice[f"dist_k{k}"].dropna()
    return {"k": k, "high_min_candidates": 20, "high_max_dist": float(dist.quantile(1 / 3)), "medium_max_dist": float(dist.quantile(2 / 3))}


def quality_table(meta, preds, variant, k, qc, phase):
    m = meta[meta["variant"] == variant].set_index("txn_id")
    p = preds[preds["model"] == f"cmp|{variant}|K={k}"].dropna(subset=["y_pred"]).set_index("txn_id")
    j = p.join(m[["n_candidates", "window", "k_found", f"dist_k{k}"]], how="inner")
    j["quality"] = [cp.quality_label(r.n_candidates, r.window, getattr(r, f"dist_k{k}"), r.k_found, qc) for r in j.itertuples()]
    j["ae"] = (j["y_pred"] - j["resale_price"]).abs()
    j["w10"] = (j["ae"] / j["resale_price"] <= 0.10)
    t = j.groupby("quality").agg(n=("ae", "size"), mae=("ae", "mean"), medae=("ae", "median"), within_10pct=("w10", "mean")).reset_index()
    t["within_10pct"] *= 100
    t["share_pct"] = t["n"] / t["n"].sum() * 100
    return t.assign(phase=phase, variant=variant, K=k)


def build_engine_config(variant: dict, k: int, qc: dict, scales: dict, hedonic: dict) -> dict:
    return {"name": "FlatFair comparables v1", "reference_label": "Comparable market reference",
            "k": int(k), "engine": cp.make_config(variant["kind"], variant["limits"], variant["recency"], False, scales, hedonic),
            "windows_months": list(cp.WINDOWS), "min_candidates": cp.MIN_CANDIDATES, "min_comparables": cp.MIN_COMPARABLES,
            "limits": cp.LIMITS[variant["limits"]], "quality": qc,
            "notes": ("same town + flat_type; sales strictly before the valuation month; exact duplicates removed; window 12->18->24 "
                      "months (first with >= min_candidates); flat_model is a similarity penalty, not a filter; remaining lease "
                      "is normalised to the valuation date; price is never used for matching")}


def run(d, pool, with_test=True, n_cv=N_TARGETS_CV, n_test=N_TARGETS_TEST, log=print):
    variants = variant_list()
    t0 = time.time()
    cv_preds, cv_meta, fold_info = [], [], {}
    for fold in FOLDS:
        targets, res, info = evaluate_fold(pool, d, fold, variants, n_cv)
        p, m = predictions_frame(targets, res, fold, pool)
        cv_preds.append(p)
        cv_meta.append(m.assign(fold=fold["fold"]))
        fold_info[fold["fold"]] = info
        log(f"  {fold['fold']} done ({time.time() - t0:.0f}s)")
    cvp, cvm = pd.concat(cv_preds, ignore_index=True), pd.concat(cv_meta, ignore_index=True)
    pooled = metrics_on_common(cvp, ["model", "config", "feature_set"])
    per_fold = metrics_on_common(cvp, ["model", "config", "feature_set", "fold"])
    pf = per_fold[per_fold["model"].str.startswith("cmp|")].assign(K=lambda x: x["model"].str.extract(r"K=(\d+)").astype(int))
    choice = choose(pooled, pf)
    variant = next(v for v in variants if v["name"] == choice["variant"])
    qc = quality_thresholds(cvm[cvm["variant"] == choice["variant"]], choice["K"])
    out = dict(variants=variants, cv_preds=cvp, cv_meta=cvm, pooled=pooled, per_fold=per_fold, choice=choice,
               variant=variant, quality=qc, fold_info=fold_info,
               quality_cv=quality_table(cvm, cvp, choice["variant"], choice["K"], qc, "cv_pooled"))
    # access ablation on the chosen method (CV only, reported separately, never used for selection)
    abl = []
    for fold in FOLDS:
        targets, res, _ = evaluate_fold(pool, d, fold, [variant], n_cv, access_for=[variant])
        p, m = predictions_frame(targets, res, fold, pool)
        abl.append(p)
    acc_cv = metrics_on_common(pd.concat(abl, ignore_index=True), ["model", "config", "feature_set"])
    out["access_cv"] = acc_cv[acc_cv["model"].str.contains(f"K={choice['K']}") | ~acc_cv["model"].str.startswith("cmp|")]
    cov = cvm.assign(ok=cvm["k_found"] >= cp.MIN_COMPARABLES, w12=cvm["window"] == 12).groupby("variant").agg(
        coverage=("ok", "mean"), mean_candidates=("n_candidates", "mean"), share_window_12m=("w12", "mean")).reset_index()
    out["coverage"] = cov
    if with_test:   # the ONE look at the holdout, chosen methodology + baselines only
        targets, res, info = evaluate_fold(pool, d, TEST_FOLD, [variant], n_test, seed=1, access_for=[variant])
        p, m = predictions_frame(targets, res, TEST_FOLD, pool)
        out.update(test_preds=p, test_meta=m, test_info=info,
                   test=metrics_on_common(p, ["model", "config", "feature_set"]),
                   quality_test=quality_table(m, p, choice["variant"], choice["K"], qc, "test"))
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def production_config(r, d, pool) -> dict:
    """Frozen canonical methodology for serving: weights/scales fitted on ALL sales (up to the latest complete month)."""
    scales = pool.scales(int(d["m"].max()))
    hed = cp.hedonic_weights(d, "core")
    return build_engine_config(r["variant"], r["choice"]["K"], r["quality"], scales, hed)


def build_tables(r, d, pool) -> dict:
    """Compact Unity Catalog outputs for the comparables evaluation."""
    variants = {v["name"]: v for v in r["variants"]}
    chosen = r["choice"]
    frames = []
    for phase, df in (("cv_pooled", r["pooled"]), ("cv_pooled_access_ablation", r["access_cv"]), ("test", r.get("test"))):
        if df is None:
            continue
        x = df.copy()
        x["phase"] = phase
        frames.append(x)
    t = pd.concat(frames, ignore_index=True)
    is_cmp = t["model"].str.startswith("cmp|")
    t["method"] = np.where(is_cmp, t["config"].str.replace("+access", "", regex=False), t["model"])
    t["K"] = t["model"].str.extract(r"K=(\d+)")[0].astype(float)
    t["use_accessibility"] = t["model"].str.contains(r"\+access")
    t["kind"] = np.where(is_cmp, t["method"].str.split("|").str[0], "baseline")
    t["limits"] = np.where(is_cmp, t["method"].str.split("|").str[1], None)
    t["status"] = np.where(~is_cmp, "baseline", np.where(t["use_accessibility"], "ablation: accessibility",
                  np.where((t["method"] == chosen["variant"]) & (t["K"] == chosen["K"]), "canonical",
                  np.where(t["kind"] == "ben", "reference: prototype weights", "candidate"))))
    cov = r["coverage"].rename(columns={"variant": "method"})
    t = t.merge(cov, on="method", how="left")
    cfg = production_config(r, d, pool)
    config = pd.DataFrame([{"created_at": pd.Timestamp.now(tz="UTC").tz_localize(None), "name": cfg["name"], "reference_label": cfg["reference_label"],
                            "k": cfg["k"], "method": chosen["variant"], "config_json": json.dumps(_jsonable(cfg))}])
    q = pd.concat([r["quality_cv"]] + ([r["quality_test"]] if "quality_test" in r else []), ignore_index=True)
    return {"comparables_evaluation": t.drop(columns=["feature_set", "config"], errors="ignore"),
            "comparables_config": config, "comparables_quality_calibration": q}, cfg


def log_mlflow(experiment, r, tables, cfg):
    import tempfile
    import mlflow
    mlflow.set_experiment(experiment)
    tmp = Path(tempfile.mkdtemp())
    pooled, per_fold = r["pooled"], r["per_fold"]
    choice = r["choice"]
    with mlflow.start_run(run_name="flatfair_comparables") as parent:
        mlflow.log_params({"hard_filters": "same town + flat_type; sales strictly before valuation month; exact duplicates removed",
                           "windows_months": "12->18->24 (first with >=10 candidates)", "min_comparables": cp.MIN_COMPARABLES,
                           "K_grid": str(KS), "flat_model": "similarity penalty (not a hard filter)",
                           "lease": "normalised to valuation date", "price_used_for_matching": "never",
                           "selection_rule": "CV only: lowest MAE on common set; transparent method within 1% of best preferred; K by MAE then stability",
                           "chosen_method": choice["variant"], "chosen_K": choice["K"], "n_targets_per_cv_fold": N_TARGETS_CV})
        mlflow.set_tags({"canonical_method": choice["variant"], "reference_label": "Comparable market reference",
                         "registry": "not registered (retrieval engine, not a model)"})
        for name, df in tables.items():
            df.to_csv(tmp / f"{name}.csv", index=False)
            mlflow.log_artifact(str(tmp / f"{name}.csv"))
        (tmp / "comparables_config.json").write_text(json.dumps(_jsonable(cfg), indent=2))
        mlflow.log_artifact(str(tmp / "comparables_config.json"))
        for v in r["variants"]:
            with mlflow.start_run(run_name=f"cv|{v['name']}", nested=True):
                mlflow.log_params({"kind": v["kind"], "limits": v["limits"], "recency_per_month": v["recency"]})
                mlflow.set_tag("phase", "cv")
                for k in KS:
                    row = pooled[pooled["model"] == f"cmp|{v['name']}|K={k}"]
                    if len(row):
                        x = row.iloc[0]
                        mlflow.log_metrics({f"cv_k{k}_mae": x["mae"], f"cv_k{k}_medae": x["medae"], f"cv_k{k}_smape_pct": x["smape_pct"],
                                            f"cv_k{k}_within_5pct": x["within_5pct"], f"cv_k{k}_within_10pct": x["within_10pct"],
                                            f"cv_k{k}_improvement_vs_baseline_pct": x["mae_improvement_vs_baseline_pct"]})
                    for f in per_fold[per_fold["model"] == f"cmp|{v['name']}|K={k}"].itertuples():
                        mlflow.log_metric(f"{f.fold}_k{k}_mae", f.mae)
        if "test" in r:
            with mlflow.start_run(run_name=f"test|{choice['variant']}|K={choice['K']}", nested=True):
                mlflow.set_tag("phase", "test")
                for x in r["test"].itertuples():
                    tag = x.model.replace("|", "_").replace("=", "")
                    mlflow.log_metrics({f"test_{tag}_mae": x.mae, f"test_{tag}_within_10pct": x.within_10pct,
                                        f"test_{tag}_improvement_vs_baseline_pct": x.mae_improvement_vs_baseline_pct})
    return parent.info.run_id


def load_inputs():
    hdb = pd.read_csv(PROCESSED / "hdb_clean.csv", parse_dates=["transaction_date"])
    acc = pd.read_csv(PROCESSED / "hdb_accessibility_features.csv")
    d, idx, info = fv.build_dataset(hdb, acc)
    pool = cp.Pool(fv.prepare_transactions(hdb), acc)
    return d, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-only", action="store_true")
    a = ap.parse_args()
    d, pool = load_inputs()
    r = run(d, pool, with_test=not a.cv_only)
    pd.set_option("display.width", 250)
    cols = ["model", "n", "mae", "medae", "smape_pct", "within_5pct", "within_10pct", "mae_improvement_vs_baseline_pct"]
    s = r["pooled"].sort_values("mae")
    print("\n== CV pooled (common target set), best 12 + baselines =="); print(s[cols].head(12).round(2).to_string(index=False))
    print(s[~s["model"].str.startswith("cmp|")][cols].round(2).to_string(index=False))
    print("\nCHOICE:", r["choice"], "| quality thresholds:", {k: round(v, 3) if isinstance(v, float) else v for k, v in r["quality"].items()})
    print(r["quality_cv"].round(2).to_string(index=False))
    print("\naccess ablation CV:"); print(r["access_cv"][cols].round(2).to_string(index=False))
    if not a.cv_only:
        print("\n== FINAL HOLDOUT (chosen method + baselines, scored once) =="); print(r["test"][cols].round(2).to_string(index=False))
        print(r["quality_test"].round(2).to_string(index=False))
    out = CACHE / "comparables"
    out.mkdir(parents=True, exist_ok=True)
    tables, cfg = build_tables(r, d, pool)
    for k, v in tables.items():
        v.to_csv(out / f"{k}.csv", index=False)
    if not a.cv_only:
        (Path(__file__).resolve().parent / "comparables_config.json").write_text(json.dumps(_jsonable(cfg), indent=2))
        print("wrote src/comparables_config.json")


if __name__ == "__main__":
    main()
