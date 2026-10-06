"""COMPARABLES engine: which actual recent sales are most similar to this flat? (explainability layer, NOT the valuation model)

Candidate filter (hard): same town, same flat_type, sale month STRICTLY BEFORE the valuation month, exact duplicates
removed, inside a recency window 12 -> 18 -> 24 months (first window with >= MIN_CANDIDATES), plus loose sanity limits on
area / storey / lease. flat_model is a similarity PENALTY, not a hard filter.
Ranking: a distance over floor area, remaining lease (normalised to the valuation date), storey, flat_model mismatch,
(optionally accessibility) and recency. resale_price and price_per_sqm are NEVER used to choose or rank neighbours;
prices are only read AFTER the top-K are selected, to report the "comparable market reference".
"""
import numpy as np
import pandas as pd

WINDOWS = (12, 18, 24)
MIN_CANDIDATES = 10          # candidates needed inside a window before it is accepted
MIN_COMPARABLES = 3          # fewer reasonable comparables -> "Insufficient comparable transactions"
INSUFFICIENT = "Insufficient comparable transactions"
LIMITS = {"ben": dict(area=0.20, storey=6.0, lease_months=120.0),     # Ben's prototype limits (reference)
          "loose": dict(area=0.30, storey=10.0, lease_months=240.0)}
BEN_WEIGHTS = dict(area=0.25, storey=0.15, lease=0.20, recency=0.35, model=0.05)   # Ben's prototype weights (reference)
POOL_COLS = ["txn_id", "m", "town", "flat_type", "flat_model", "floor_area_sqm", "storey_mid", "remaining_lease_months",
             "block", "street_name", "storey_range", "resale_price", "is_exact_duplicate"]


class Pool:
    """Per-(town, flat_type) arrays sorted by month, built once from transactions (+ optional accessibility)."""

    def __init__(self, df: pd.DataFrame, acc: pd.DataFrame | None = None):
        d = df[~df["is_exact_duplicate"].astype(bool)].copy()
        if acc is not None and "distance_to_nearest_mrt_m" not in d.columns:
            d = d.merge(acc[["property_id", "distance_to_nearest_mrt_m"]].drop_duplicates("property_id"), on="property_id", how="left")
        if "distance_to_nearest_mrt_m" not in d.columns:
            d["distance_to_nearest_mrt_m"] = np.nan
        d["log_mrt"] = np.log1p(d["distance_to_nearest_mrt_m"])
        d = d.sort_values(["town", "flat_type", "m"], kind="stable").reset_index(drop=True)
        self.df = d
        self.cells = {}
        for key, g in d.groupby(["town", "flat_type"], sort=False):
            ix = g.index.to_numpy()
            self.cells[key] = dict(ix=ix, m=g["m"].to_numpy(int), area=g["floor_area_sqm"].to_numpy(float),
                                   storey=g["storey_mid"].to_numpy(float), lease=g["remaining_lease_months"].to_numpy(float),
                                   model=g["flat_model"].to_numpy(object), logmrt=g["log_mrt"].to_numpy(float))

    def scales(self, max_m: int) -> dict:
        """Pooled within-cell standard deviations from sales up to month index `max_m` (structure only, no prices)."""
        d = self.df[self.df["m"] <= max_m]
        g = d.assign(la=np.log(d["floor_area_sqm"])).groupby(["town", "flat_type"])
        w = g.size()
        f = lambda c: float(np.sqrt((g[c].var().fillna(0) * w).sum() / w.sum()))
        return {"log_area": f("la"), "storey": f("storey_mid"), "lease_months": f("remaining_lease_months"),
                "log_mrt": float(d["log_mrt"].std()) if d["log_mrt"].notna().any() else 1.0}


def hedonic_weights(train: pd.DataFrame, feature_set: str = "core") -> dict:
    """Price-impact weights from a Ridge hedonic fitted on the TRAINING window (log price per unit of each attribute).
    Used only to put attribute differences on a common 'modelled value difference' scale."""
    import fair_value_models as fm
    from fair_value_data import FEATURE_SETS
    cfg = next(g for g in fm.model_grid(feature_set) if g[0] == "hedonic_ridge" and g[1] == "alpha=100")
    pipe = cfg[3]()
    pipe.fit(fm.design(train, feature_set), train["y_rel"])
    ct, ridge = pipe.named_steps["prep"], pipe.named_steps["m"]
    names = list(ct.get_feature_names_out())
    coef = dict(zip(names, ridge.coef_))
    scaler = ct.named_transformers_["num"].named_steps["sc"]
    cat, num = FEATURE_SETS[feature_set]
    per_unit = {n: coef[f"num__{n}"] / s for n, s in zip(num, scaler.scale_)}
    models = {k.split("flat_model_", 1)[1]: v for k, v in coef.items() if k.startswith("cat__flat_model_")}
    freq = train["flat_model"].value_counts()
    common = [m for m in models if freq.get(m, 0) >= 500]
    vals = np.array([models[m] for m in common])
    pen = float(np.mean(np.abs(vals[:, None] - vals[None, :]))) if len(vals) > 1 else 0.0
    return {"log_area": per_unit["log_area"], "storey": per_unit["storey_mid"], "storey_sq": per_unit["storey_sq"],
            "lease_years": per_unit["lease_years"], "lease_years_sq": per_unit["lease_years_sq"], "model_penalty": pen,
            "log_mrt": per_unit.get("log_mrt_dist", 0.0)}


def make_config(kind: str, limits: str = "loose", recency_per_month: float = 0.0, use_access: bool = False,
                scales: dict | None = None, hedonic: dict | None = None) -> dict:
    return dict(kind=kind, limits=limits, recency_per_month=recency_per_month, use_access=use_access,
                scales=scales, hedonic=hedonic)


def _distance(cfg: dict, c: dict, sel: np.ndarray, q: dict, v: int):
    """Distance of candidates `sel` (indices into cell arrays) to query q at valuation month v. Never touches prices."""
    age = (v - c["m"][sel]).astype(float)                                  # months before valuation (>= 1)
    lease_now = c["lease"][sel] - age                                      # candidate lease normalised to the valuation date
    d_area = np.abs(np.log(c["area"][sel] / q["area"]))
    d_lease = np.abs(lease_now - q["lease_months"])
    d_storey = np.abs(c["storey"][sel] - q["storey"])
    mism = (c["model"][sel] != q["model"]).astype(float) if q.get("model") else np.zeros(len(sel))
    use_mrt = cfg["use_access"] and q.get("log_mrt") is not None and q["log_mrt"] == q["log_mrt"]
    d_mrt = np.where(np.isnan(c["logmrt"][sel]), 0.0, np.abs(c["logmrt"][sel] - q["log_mrt"])) if use_mrt else np.zeros(len(sel))
    k, lim = cfg["kind"], LIMITS[cfg["limits"]]
    if k == "ben":      # Ben's prototype formula (no location tiers here): weighted L1 / his limits
        w = BEN_WEIGHTS
        return (w["area"] * (np.abs(c["area"][sel] - q["area"]) / q["area"]) / LIMITS["ben"]["area"]
                + w["storey"] * d_storey / LIMITS["ben"]["storey"] + w["lease"] * d_lease / LIMITS["ben"]["lease_months"]
                + w["recency"] * age / 36 + w["model"] * mism)
    if k == "equal_l1":  # equal weights, each attribute normalised by its allowed limit
        terms = [np.abs(c["area"][sel] - q["area"]) / q["area"] / lim["area"], d_storey / lim["storey"],
                 d_lease / lim["lease_months"], mism, age / WINDOWS[-1]]
        if use_mrt:
            terms.append(d_mrt / max(cfg["scales"]["log_mrt"], 1e-6))
        return np.sum(terms, axis=0)
    if k == "knn":       # standardised Euclidean distance
        s = cfg["scales"]
        terms = [d_area / s["log_area"], d_storey / s["storey"], d_lease / s["lease_months"], mism,
                 age / (WINDOWS[0] / np.sqrt(12))]
        if use_mrt:
            terms.append(d_mrt / max(s["log_mrt"], 1e-6))
        return np.sqrt(np.sum(np.square(terms), axis=0))
    if k == "hedonic":   # distance in modelled log-price units
        h = cfg["hedonic"]
        ly_c, ly_q = lease_now / 12, q["lease_months"] / 12
        t_lease = np.abs(h["lease_years"] * (ly_c - ly_q) + h["lease_years_sq"] * (ly_c ** 2 - ly_q ** 2))
        t_storey = np.abs(h["storey"] * (c["storey"][sel] - q["storey"]) + h["storey_sq"] * (c["storey"][sel] ** 2 - q["storey"] ** 2))
        return (np.abs(h["log_area"]) * d_area + t_lease + t_storey + h["model_penalty"] * mism
                + np.abs(h["log_mrt"]) * d_mrt * use_mrt + cfg["recency_per_month"] * age)
    raise ValueError(k)


def rank(pool: Pool, cfg: dict, q: dict, v: int, top: int = 10) -> dict:
    """q: town, flat_type, area, storey, lease_months, model (optional), log_mrt (optional). Returns ranking + provenance."""
    c = pool.cells.get((q["town"], q["flat_type"]))
    lim = LIMITS[cfg["limits"]]
    out = dict(n_candidates=0, window_used=None, order=np.empty(0, int), dist=np.empty(0), ix=np.empty(0, int))
    if c is None:
        return out
    lo_all = np.searchsorted(c["m"], v - WINDOWS[-1], side="left")
    hi = np.searchsorted(c["m"], v, side="left")                           # strictly before the valuation month
    for w in WINDOWS:
        lo = np.searchsorted(c["m"], v - w, side="left")
        sel = np.arange(lo, hi)
        if len(sel):
            age = (v - c["m"][sel]).astype(float)
            ok = ((np.abs(c["area"][sel] - q["area"]) / q["area"] <= lim["area"])
                  & (np.abs(c["storey"][sel] - q["storey"]) <= lim["storey"])
                  & (np.abs((c["lease"][sel] - age) - q["lease_months"]) <= lim["lease_months"]))
            sel = sel[ok]
        out.update(n_candidates=len(sel), window_used=w)
        if len(sel) >= MIN_CANDIDATES or w == WINDOWS[-1]:
            break
    sel = np.arange(np.searchsorted(c["m"], v - out["window_used"], side="left"), hi)
    age = (v - c["m"][sel]).astype(float)
    ok = ((np.abs(c["area"][sel] - q["area"]) / q["area"] <= lim["area"])
          & (np.abs(c["storey"][sel] - q["storey"]) <= lim["storey"])
          & (np.abs((c["lease"][sel] - age) - q["lease_months"]) <= lim["lease_months"]))
    sel = sel[ok]
    if len(sel) == 0:
        return out
    dist = _distance(cfg, c, sel, q, v)
    # deterministic tie-break: smaller distance, then more recent
    o = np.lexsort((-c["m"][sel], dist))[:top]
    out.update(order=sel[o], dist=dist[o], ix=c["ix"][sel[o]])
    return out


# ---------------------------------------------------------------- product-facing
def quality_label(n_candidates, window, mean_dist, k_found, cfg_q: dict, model_match_share=None, acc_available=None):
    """HIGH / MEDIUM / LOW from validated thresholds (distance terciles on CV). < MIN_COMPARABLES -> insufficient."""
    if k_found < MIN_COMPARABLES:
        return INSUFFICIENT
    if window == WINDOWS[0] and n_candidates >= cfg_q["high_min_candidates"] and mean_dist <= cfg_q["high_max_dist"] and k_found >= cfg_q["k"]:
        return "HIGH"
    if n_candidates >= MIN_CANDIDATES and mean_dist <= cfg_q["medium_max_dist"]:
        return "MEDIUM"
    return "LOW"


def find_comparables(pool: Pool, config: dict, town, flat_type, floor_area_sqm, storey, remaining_lease_months,
                     valuation_month: int, flat_model=None, k=5, mrt_distance_m=None, display_k=None) -> dict:
    """Top-k comparable transactions + the 'comparable market reference' (median of the K reference comparables).
    config = frozen validated methodology (see comparables_config.json). `k` rows are returned for display; the reference
    median uses config['k'] (the validated K) unless k is larger."""
    cfg = config["engine"]
    q = dict(town=town, flat_type=flat_type, area=float(floor_area_sqm), storey=float(storey),
             lease_months=float(remaining_lease_months), model=flat_model,
             log_mrt=None if mrt_distance_m is None else float(np.log1p(mrt_distance_m)))
    r = rank(pool, cfg, q, valuation_month, top=max(k, config["k"]))
    n = len(r["ix"])
    base = dict(n_candidates=int(r["n_candidates"]), search_window_months=r["window_used"])
    if n < MIN_COMPARABLES:
        return dict(base, status=INSUFFICIENT, quality=INSUFFICIENT, comparables=pd.DataFrame(), reference=None)
    top = pool.df.loc[r["ix"]].copy()
    top["comparable_rank"] = np.arange(1, n + 1)
    top["distance"] = r["dist"]
    top["similarity_score"] = 100 / (1 + r["dist"])               # 100 = identical, decreasing with distance
    top["transaction_month"] = pd.to_datetime((top["m"] // 12).astype(str) + "-" + (top["m"] % 12 + 1).astype(str).str.zfill(2) + "-01")
    top["flat_model_match"] = (top["flat_model"] == flat_model) if flat_model else np.nan
    kk = config["k"]
    ref = top["resale_price"].iloc[:kk]
    qc = config["quality"]
    mean_dist = float(r["dist"][:kk].mean())
    label = quality_label(r["n_candidates"], r["window_used"], mean_dist, n, qc)
    shown = top.head(k)[["comparable_rank", "similarity_score", "transaction_month", "town", "block", "street_name",
                         "flat_type", "floor_area_sqm", "storey_range", "remaining_lease_months", "flat_model",
                         "distance_to_nearest_mrt_m", "resale_price"]]
    return dict(base, status="ok", quality=label, comparables=shown.reset_index(drop=True),
                reference=dict(name="Comparable market reference", median_price=float(ref.median()), min_price=float(ref.min()),
                               max_price=float(ref.max()), q25=float(ref.quantile(.25)), q75=float(ref.quantile(.75)),
                               n_used=int(len(ref)), k=kk, mean_distance=mean_dist))
