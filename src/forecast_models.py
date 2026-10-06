"""Backtest folds, baselines, ML models and metrics for the direct-horizon price forecasts.

Target for the ML models: log(y_{t+h} / anchor_t), anchor = trailing 3-month pooled median at origin t.
Predictions are converted back to dollars (anchor * exp(pred)) before ANY metric is computed.
Price-level features are fed to the models as log(feature / anchor) (same information, scale-free) so tree
models are not asked to extrapolate absolute price levels through a trending market.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from forecast_data import CATEGORICAL, NUMERIC_FEATURES

# Expanding windows. train_end = last TARGET month whose label is known to the fold's training set.
FOLDS = [
    dict(fold="cv1", train_end="2022-09-01", val_start="2022-10-01", val_end="2023-09-01"),
    dict(fold="cv2", train_end="2023-09-01", val_start="2023-10-01", val_end="2024-09-01"),
    dict(fold="cv3", train_end="2024-09-01", val_start="2024-10-01", val_end="2025-09-01"),
]
TEST_FOLD = dict(fold="test", train_end="2025-09-01", val_start="2025-10-01", val_end="2026-09-01")
HORIZONS = (1, 3, 6)

BASELINES = {"naive_prev_month": "base_prev_month", "rolling3_median": "base_roll3_median",
             "seasonal_naive": "base_seasonal_naive", "pooled_3m_median": "base_pooled_3m"}
NAIVE = "naive_prev_month"
DRIFT = "drift_pooled_3m"   # ML-free benchmark: pooled 3m median x exp(median historical log-change), fit on train only

PRICE_COLS = ["price_lag_0", "price_lag_1", "price_lag_2", "price_lag_5", "price_lag_11",
              "rolling_3m_price", "rolling_6m_price", "rolling_12m_price"]
REL_NUMERIC = ([f"rel_{c}" for c in PRICE_COLS] + ["log_anchor"] +
               [c for c in NUMERIC_FEATURES if c not in PRICE_COLS])


def design(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs. Only information available at the forecast origin (+ the target month's calendar)."""
    X = df[CATEGORICAL].copy()
    for c in PRICE_COLS:
        X[f"rel_{c}"] = np.log(df[c] / df["anchor"])
    X["log_anchor"] = np.log(df["anchor"])
    for c in NUMERIC_FEATURES:
        if c not in PRICE_COLS:
            X[c] = df[c]
    return X


def _prep(scale: bool):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                              ("num", num, REL_NUMERIC)])


def model_grid() -> list:
    """(model_name, config_id, params, factory). Small fixed grids -- tuned on cv1-cv3 only."""
    from xgboost import XGBRegressor
    out = []
    for a in (1, 10, 100):
        out.append(("ridge", f"alpha={a}", dict(alpha=a),
                    lambda a=a: Pipeline([("prep", _prep(True)), ("m", Ridge(alpha=a))])))
    for d, leaf in itertools.product((6, 12), (5, 20)):
        out.append(("random_forest", f"depth={d},leaf={leaf}",
                    dict(n_estimators=300, max_depth=d, min_samples_leaf=leaf, max_features=0.5),
                    lambda d=d, leaf=leaf: Pipeline([("prep", _prep(False)), ("m", RandomForestRegressor(
                        n_estimators=300, max_depth=d, min_samples_leaf=leaf, max_features=0.5,
                        n_jobs=-1, random_state=0))])))
    for d, mcw in itertools.product((3, 5), (5, 20)):
        out.append(("xgboost", f"depth={d},mcw={mcw}",
                    dict(n_estimators=300, learning_rate=0.05, max_depth=d, min_child_weight=mcw,
                         subsample=0.8, colsample_bytree=0.8),
                    lambda d=d, mcw=mcw: Pipeline([("prep", _prep(False)), ("m", XGBRegressor(
                        n_estimators=300, learning_rate=0.05, max_depth=d, min_child_weight=mcw,
                        subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=4))])))
    return out


def split(ds: pd.DataFrame, fold: dict):
    """Chronological split. Train: label month <= train_end. Eval: label month inside the validation window.
    Eval origins may be after train_end (their features only use data that exists at their own origin)."""
    te = pd.Timestamp(fold["train_end"])
    tr = ds[ds["target_month"] <= te]
    ev = ds[(ds["target_month"] >= pd.Timestamp(fold["val_start"])) &
            (ds["target_month"] <= pd.Timestamp(fold["val_end"]))]
    return tr, ev


ID_COLS = ["town", "flat_type", "origin_month", "target_month", "horizon", "y", "y_count"]


def baseline_predictions(ev: pd.DataFrame, fold: dict, tr: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for name, col in BASELINES.items():
        d = ev[ID_COLS].copy()
        d["y_pred"], d["model"], d["config"], d["fold"] = ev[col].to_numpy(), name, "-", fold["fold"]
        parts.append(d)
    d = ev[ID_COLS].copy()  # constant market drift learned from the training window only
    d["y_pred"] = ev["anchor"].to_numpy() * np.exp(tr["y_rel"].median())
    d["model"], d["config"], d["fold"] = DRIFT, "-", fold["fold"]
    parts.append(d)
    return pd.concat(parts, ignore_index=True)


def fit_predict(factory, tr: pd.DataFrame, ev: pd.DataFrame):
    m = factory()
    m.fit(design(tr), tr["y_rel"])
    return m, ev["anchor"].to_numpy() * np.exp(m.predict(design(ev)))


def model_predictions(name, config, factory, tr, ev, fold):
    m, pred = fit_predict(factory, tr, ev)
    d = ev[ID_COLS].copy()
    d["y_pred"], d["model"], d["config"], d["fold"] = pred, name, config, fold["fold"]
    return m, d


# ---------------------------------------------------------------- metrics
STATUS = {"pooled_3m_median": "production", "random_forest": "challenger", "xgboost": "challenger",
          "ridge": "challenger", "naive_prev_month": "baseline", "rolling3_median": "baseline",
          "seasonal_naive": "baseline", "drift_pooled_3m": "baseline"}
PRODUCTION = "pooled_3m_median"


def summarise(preds: pd.DataFrame, by=("model", "config", "fold", "horizon")) -> pd.DataFrame:
    """MAE / RMSE / sMAPE / MAPE by `by`, plus % MAE improvement vs the naive previous-month baseline
    computed on the SAME rows (all models predict identical evaluation rows)."""
    by = list(by)
    d = preds.assign(ae=(preds["y_pred"] - preds["y"]).abs())
    d["se"] = d["ae"] ** 2
    d["sm"] = 2 * d["ae"] / (d["y"].abs() + d["y_pred"].abs())
    d["ape"] = d["ae"] / d["y"]

    def agg(frame, keys):
        g = frame.groupby(keys).agg(n=("ae", "size"), mae=("ae", "mean"), mse=("se", "mean"),
                                    smape_pct=("sm", "mean"), mape_pct=("ape", "mean")).reset_index()
        g["rmse"] = np.sqrt(g.pop("mse"))
        g["smape_pct"] *= 100
        g["mape_pct"] *= 100
        return g

    out = agg(d, by)
    rest = [c for c in by if c not in ("model", "config")]
    nv = d[d["model"] == NAIVE]
    if rest:
        out = out.merge(agg(nv, rest)[rest + ["mae"]].rename(columns={"mae": "naive_mae"}), on=rest, how="left")
    else:
        out["naive_mae"] = agg(nv.assign(_k=0), ["_k"])["mae"].iloc[0]
    out["mae_improvement_vs_naive_pct"] = (out["naive_mae"] - out["mae"]) / out["naive_mae"] * 100
    return out[by + ["n", "mae", "rmse", "smape_pct", "mape_pct", "naive_mae", "mae_improvement_vs_naive_pct"]]


def metrics_long(cv: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Long metric table for dashboards: phase x model x config x horizon x segment."""
    frames = []

    def add(preds, phase, seg_type=None, extra=()):
        s = summarise(preds, by=("model", "config", "horizon") + tuple(extra))
        s["segment_type"] = seg_type or "all"
        s["segment"] = s[seg_type] if seg_type else "all"
        frames.append(s.drop(columns=list(extra)).assign(phase=phase))

    for phase, preds in (("cv_pooled", cv), ("test", test)):
        add(preds, phase)
        add(preds, phase, "town", ("town",))
        add(preds, phase, "flat_type", ("flat_type",))
    for f, g in cv.groupby("fold"):
        add(g, f)
    out = pd.concat(frames, ignore_index=True)
    out["status"] = out["model"].map(STATUS)
    cols = ["phase", "model", "config", "status", "horizon", "segment_type", "segment", "n", "mae", "rmse",
            "smape_pct", "mape_pct", "naive_mae", "mae_improvement_vs_naive_pct"]
    return out[cols]


def comparison_table(cv: pd.DataFrame, test: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """One compact row per model x horizon: CV MAE, test MAE, improvement vs naive, production/challenger status.
    For ML models the CV figure is the pre-selected config's CV MAE."""
    key = ["model", "config", "horizon"]
    c = summarise(cv, by=tuple(key)).rename(columns={"mae": "cv_mae", "mae_improvement_vs_naive_pct": "cv_improvement_vs_naive_pct"})
    t = summarise(test, by=tuple(key)).rename(columns={"mae": "test_mae", "rmse": "test_rmse", "smape_pct": "test_smape_pct",
                                                       "mae_improvement_vs_naive_pct": "test_improvement_vs_naive_pct"})
    out = t[key + ["test_mae", "test_rmse", "test_smape_pct", "test_improvement_vs_naive_pct"]].merge(
        c[key + ["cv_mae", "cv_improvement_vs_naive_pct"]], on=key, how="left")
    prod = out[out["model"] == PRODUCTION].set_index("horizon")["test_mae"]
    out["test_improvement_vs_production_pct"] = (out["horizon"].map(prod) - out["test_mae"]) / out["horizon"].map(prod) * 100
    out["status"] = out["model"].map(STATUS)
    out = out.rename(columns={"config": "selected_config"})
    order = {"production": 0, "challenger": 1, "baseline": 2}
    out = out.sort_values(["horizon", "status", "test_mae"], key=lambda s: s.map(order) if s.name == "status" else s)
    return out[["model", "horizon", "status", "selected_config", "cv_mae", "test_mae", "test_rmse", "test_smape_pct",
                "cv_improvement_vs_naive_pct", "test_improvement_vs_naive_pct", "test_improvement_vs_production_pct"]
               ].reset_index(drop=True)
