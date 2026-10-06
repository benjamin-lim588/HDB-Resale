"""Fair-value models, chronological folds, metrics and the (pre-registered) champion rule.

All ML models predict log(resale_price / anchor); dollars = anchor * exp(prediction) and every metric is computed on
dollars. The baseline is the anchor itself (recent town x flat_type median, strictly prior months).
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fair_value_data import FEATURE_SETS

FOLDS = [
    dict(fold="cv1", train_end="2022-09-01", val_start="2022-10-01", val_end="2023-09-01"),
    dict(fold="cv2", train_end="2023-09-01", val_start="2023-10-01", val_end="2024-09-01"),
    dict(fold="cv3", train_end="2024-09-01", val_start="2024-10-01", val_end="2025-09-01"),
]
TEST_FOLD = dict(fold="test", train_end="2025-09-01", val_start="2025-10-01", val_end="2026-09-01")
BASELINE = "market_median_baseline"
FAMILIES = ["hedonic_ridge", "random_forest", "xgboost"]
# Simplicity preference used by the champion rule: an ML model must beat hedonic by more than this to be preferred.
ML_OVER_HEDONIC_MARGIN_PCT = 1.0


def split(d: pd.DataFrame, fold: dict):
    """Chronological: train on sales up to train_end; evaluate on the validation window (no shuffling)."""
    tr = d[d["month"] <= pd.Timestamp(fold["train_end"])]
    ev = d[(d["month"] >= pd.Timestamp(fold["val_start"])) & (d["month"] <= pd.Timestamp(fold["val_end"]))]
    return tr, ev


def design(d: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    cat, num = FEATURE_SETS[feature_set]
    return d[cat + num]


def _prep(feature_set: str, linear: bool):
    cat, num = FEATURE_SETS[feature_set]
    ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False)
    numeric = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]) if linear else "passthrough"
    return ColumnTransformer([("cat", ohe, cat), ("num", numeric, num)])


def model_grid(feature_set: str = "core") -> list:
    """(family, config_id, params, factory). Small fixed grids, tuned on cv1-cv3 only."""
    from xgboost import XGBRegressor
    out = []
    for a in (1, 100, 1000, 10000):
        out.append(("hedonic_ridge", f"alpha={a}", dict(alpha=a),
                    lambda a=a: Pipeline([("prep", _prep(feature_set, True)), ("m", Ridge(alpha=a))])))
    for d, leaf in [(12, 10), (12, 30), (20, 10), (20, 30), (30, 10), (30, 5)]:
        p = dict(n_estimators=200, max_depth=d, min_samples_leaf=leaf, max_features=0.5)
        out.append(("random_forest", f"depth={d},leaf={leaf}", p,
                    lambda p=p: Pipeline([("prep", _prep(feature_set, False)),
                                          ("m", RandomForestRegressor(**p, n_jobs=-1, random_state=0))])))
    for d, mcw in itertools.product((4, 6, 8), (10, 50)):
        p = dict(n_estimators=400, learning_rate=0.05, max_depth=d, min_child_weight=mcw, subsample=0.8,
                 colsample_bytree=0.8)
        out.append(("xgboost", f"depth={d},mcw={mcw}", p,
                    lambda p=p: Pipeline([("prep", _prep(feature_set, False)),
                                          ("m", XGBRegressor(**p, tree_method="hist", n_jobs=4, random_state=0))])))
    return out


ID_COLS = ["txn_id", "month", "town", "flat_type", "resale_price", "anchor"]


def baseline_predictions(ev: pd.DataFrame, fold: dict) -> pd.DataFrame:
    p = ev[ID_COLS].copy()
    p["y_pred"], p["model"], p["config"], p["feature_set"], p["fold"] = ev["anchor"].to_numpy(), BASELINE, "-", "core", fold["fold"]
    return p


def fit_predict(factory, tr, ev, feature_set):
    m = factory()
    m.fit(design(tr, feature_set), tr["y_rel"])
    return m, ev["anchor"].to_numpy() * np.exp(m.predict(design(ev, feature_set)))


def model_predictions(family, config, factory, tr, ev, fold, feature_set="core"):
    m, pred = fit_predict(factory, tr, ev, feature_set)
    p = ev[ID_COLS].copy()
    p["y_pred"], p["model"], p["config"], p["feature_set"], p["fold"] = pred, family, config, feature_set, fold["fold"]
    return m, p


# ---------------------------------------------------------------- metrics
def summarise(preds: pd.DataFrame, by, extra_cols=None) -> pd.DataFrame:
    """MAE/RMSE/MedAE/sMAPE/MAPE and share within +-5/10/15%, plus improvement vs the baseline on the SAME rows."""
    by = list(by)
    d = preds.assign(ae=(preds["y_pred"] - preds["resale_price"]).abs())
    d["se"] = d["ae"] ** 2
    d["ape"] = d["ae"] / d["resale_price"]
    d["sm"] = 2 * d["ae"] / (d["resale_price"] + d["y_pred"].abs())
    for t in (5, 10, 15):
        d[f"w{t}"] = (d["ape"] <= t / 100).astype(float)

    def agg(frame, keys):
        g = frame.groupby(keys, observed=True).agg(n=("ae", "size"), mae=("ae", "mean"), mse=("se", "mean"),
                                                   medae=("ae", "median"), smape_pct=("sm", "mean"),
                                                   mape_pct=("ape", "mean"), within_5pct=("w5", "mean"),
                                                   within_10pct=("w10", "mean"), within_15pct=("w15", "mean")).reset_index()
        g["rmse"] = np.sqrt(g.pop("mse"))
        for c in ("smape_pct", "mape_pct", "within_5pct", "within_10pct", "within_15pct"):
            g[c] *= 100
        return g

    out = agg(d, by)
    rest = [c for c in by if c not in ("model", "config", "feature_set")]
    bl = d[d["model"] == BASELINE]
    names = {"mae": "baseline_mae", "medae": "baseline_medae", "within_10pct": "baseline_within_10pct"}
    if len(bl) == 0:  # no baseline rows supplied: improvement columns are NaN
        out = out.assign(**{v: np.nan for v in names.values()})
    else:
        base = (agg(bl, rest) if rest else agg(bl.assign(_k=0), ["_k"]))
        base = base[rest + list(names)].rename(columns=names)
        out = out.merge(base, on=rest, how="left") if rest else out.assign(**base.iloc[0].to_dict())
    out["mae_improvement_vs_baseline_pct"] = (out["baseline_mae"] - out["mae"]) / out["baseline_mae"] * 100
    out["medae_improvement_vs_baseline_pct"] = (out["baseline_medae"] - out["medae"]) / out["baseline_medae"] * 100
    return out


def add_segments(p: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """Attach reporting segments (by ACTUAL attributes/price: for reporting only, never features)."""
    cols = ["txn_id", "remaining_lease_months", "flat_age_at_transaction", "anchor_tier", "acc_missing", "any_flag"]
    s = p.merge(d[cols].drop_duplicates("txn_id"), on="txn_id", how="left")
    s["price_band"] = pd.cut(s["resale_price"], [0, 400e3, 500e3, 650e3, 800e3, 1e6, 1e9],
                             labels=["<400k", "400-500k", "500-650k", "650-800k", "800k-1m", ">=1m"]).astype(str)
    s["lease_band"] = pd.cut(s["remaining_lease_months"] / 12, [0, 60, 70, 80, 90, 200],
                             labels=["<60y", "60-70y", "70-80y", "80-90y", ">=90y"]).astype(str)
    s["age_bucket"] = pd.cut(s["flat_age_at_transaction"], [-1, 10, 20, 30, 40, 100],
                             labels=["<10y", "10-20y", "20-30y", "30-40y", "40y+"]).astype(str)
    s["anchor_basis"] = np.where(s["anchor_tier"] == 0, "3m cell median", "widened window / island fallback")
    s["accessibility_data"] = np.where(s["acc_missing"] == 1, "missing", "available")
    s["flagged_record"] = np.where(s["any_flag"], "flagged", "not flagged")
    return s


SEGMENTS = ["town", "flat_type", "price_band", "lease_band", "age_bucket", "anchor_basis", "accessibility_data",
            "flagged_record"]


def segment_metrics(p: pd.DataFrame, d: pd.DataFrame, phase: str) -> pd.DataFrame:
    s = add_segments(p, d)
    frames = [summarise(s, ["model", "config", "feature_set"]).assign(segment_type="all", segment="all")]
    for seg in SEGMENTS:
        frames.append(summarise(s, ["model", "config", "feature_set", seg]).rename(columns={seg: "segment"}).assign(segment_type=seg))
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "phase", phase)
    return out


# ---------------------------------------------------------------- pre-registered champion rule
def choose_champion(cv_best: pd.DataFrame, test_best: pd.DataFrame) -> dict:
    """cv_best / test_best: one row per model with `mae` (pooled CV / final holdout) for each family's CV-selected
    config plus the baseline. Rule fixed BEFORE results were seen:
      1. Pick the CV winner: lowest pooled-CV MAE among hedonic / RF / XGBoost. A more complex model replaces the
         hedonic model only if it improves CV MAE by more than ML_OVER_HEDONIC_MARGIN_PCT.
      2. Confirm on the untouched holdout: the winner must beat the baseline on pooled-CV MAE AND holdout MAE.
         Otherwise the baseline stays the recommended estimate and the models are challengers.
    The holdout never chooses between models."""
    cv = cv_best.set_index("model")["mae"]
    fams = [f for f in FAMILIES if f in cv.index]
    winner = min(fams, key=lambda f: cv[f])
    if winner != "hedonic_ridge" and "hedonic_ridge" in cv.index:
        if (cv["hedonic_ridge"] - cv[winner]) / cv["hedonic_ridge"] * 100 <= ML_OVER_HEDONIC_MARGIN_PCT:
            winner = "hedonic_ridge"
    te = test_best.set_index("model")["mae"]
    ok = cv[winner] < cv[BASELINE] and te[winner] < te[BASELINE]
    return {"cv_winner": winner, "champion": winner if ok else BASELINE, "confirmed_on_holdout": bool(ok)}


# ---------------------------------------------------------------- interpretation helpers
def pipeline_feature_names(pipe) -> list:
    return [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]


def hedonic_effects(pipe, feature_set: str = "core") -> pd.DataFrame:
    """Hedonic Ridge effects on log(price/anchor). Numeric: effect per ORIGINAL unit (coef / scaler scale);
    categorical: coefficient of the one-hot level (relative to the regularised average). pct_effect = exp(coef)-1."""
    from fair_value_data import FEATURE_SETS
    cat, num = FEATURE_SETS[feature_set]
    ct, ridge = pipe.named_steps["prep"], pipe.named_steps["m"]
    names = list(ct.get_feature_names_out())
    coef = dict(zip(names, ridge.coef_))
    scale = dict(zip(num, ct.named_transformers_["num"].named_steps["sc"].scale_))
    rows = [dict(feature=n, kind="numeric", coef_log_points=coef[f"num__{n}"] / scale[n]) for n in num]
    rows += [dict(feature=n.split("__", 1)[1], kind="categorical", coef_log_points=c) for n, c in coef.items() if n.startswith("cat__")]
    out = pd.DataFrame(rows)
    out["pct_effect"] = (np.exp(out["coef_log_points"]) - 1) * 100
    return out


def grouped_importance(pipe) -> pd.DataFrame:
    """Impurity/gain importance, with one-hot columns summed back to their source variable."""
    imp = pipe.named_steps["m"].feature_importances_
    names = pipeline_feature_names(pipe)
    cats = ("town", "flat_type", "flat_model")
    base = [next((c for c in cats if n == c or n.startswith(c + "_")), n) for n in names]
    return (pd.DataFrame({"feature": base, "importance": imp}).groupby("feature", as_index=False).sum()
            .sort_values("importance", ascending=False).reset_index(drop=True))
