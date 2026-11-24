"""Lead-time analytics and delay nowcasting for open purchase orders.

Question answered: "this PO is open today — when will it really arrive?" (not "what did the
supplier promise"). Trained on monthly snapshots of the open order book: for every PO open at a
cutoff we know its age, how overdue it is and the supplier's recent behaviour, and later its real
delivery date. Three quantile gradient-boosting models give P10 / P50 / P90 of the delay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..config import AS_OF, TEST_START, TIER1_COMMODITIES, TRAIN_END
from .features import FeatureBuilder

SUPPLIER_FEATS = ["d90_n", "d90_otd", "d90_delay", "d90_delay_std", "d90_reject", "d365_otd", "d365_delay",
                  "d365_lt_cv", "delay_trend", "commodity_delay_90", "rel_delay_90", "peer_delay_90",
                  "neg_signals_90", "neg_severity_90", "fin_score", "fin_delta_2q", "country_risk",
                  "upstream_neg_signals_180"]
PO_FEATS = ["quoted_lead_time", "days_to_promise", "age_days", "overdue_days", "commodity_code",
            "sup_open_overdue_share", "sup_open_overdue_mean", "d30_delay"]
FEATURES = PO_FEATS + SUPPLIER_FEATS
QUANTILES = (0.1, 0.5, 0.9)
COMMODITY_CODE = {c: i for i, c in enumerate(TIER1_COMMODITIES)}


def open_book(po: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """POs ordered before the cutoff and not delivered by then."""
    m = (po.order_date < cutoff) & (po.delivered_date.isna() | (po.delivered_date >= cutoff))
    return po[m]


def po_features(fb: FeatureBuilder, book: pd.DataFrame, cutoff, sf: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features of the open POs in ``book`` at ``cutoff`` (``sf``: precomputed supplier features)."""
    cutoff = pd.Timestamp(cutoff)
    X = pd.DataFrame(index=book.index)
    X["quoted_lead_time"] = book.quoted_lead_time
    X["days_to_promise"] = (book.promised_date - cutoff).dt.days
    X["age_days"] = (cutoff - book.order_date).dt.days
    X["overdue_days"] = (-X.days_to_promise).clip(lower=0)
    X["commodity_code"] = book.supplier_id.map(fb.sup.commodity).map(COMMODITY_CODE)
    # real-time signal: how much of the supplier's open order book is already overdue
    g = X.groupby(book.supplier_id)
    X["sup_open_overdue_share"] = g.overdue_days.transform(lambda v: (v > 0).mean())
    X["sup_open_overdue_mean"] = g.overdue_days.transform("mean")
    if sf is None:
        sids = book.supplier_id.unique()
        sf = fb.supplier_features(sids, [cutoff] * len(sids))
        sf.index = sids
        d30 = fb.delivery_stats(sids, [cutoff] * len(sids), 30, "d30")
        sf["d30_delay"] = d30.d30_delay.to_numpy()
    X = X.join(sf[SUPPLIER_FEATS + ["d30_delay"]], on=book.supplier_id)
    return X[FEATURES]


def supplier_panel(fb: FeatureBuilder, cutoffs) -> pd.DataFrame:
    """Supplier features for every (tier-1 supplier, cutoff) pair in one vectorised pass."""
    sids = fb.sup.index[fb.sup.tier == 1].to_numpy()
    cut = pd.DatetimeIndex(cutoffs)
    S = np.repeat(sids, len(cut))
    T = np.tile(cut.to_numpy(), len(sids))
    f = fb.supplier_features(S, T)
    f["d30_delay"] = fb.delivery_stats(S, T, 30, "d30").d30_delay.to_numpy()
    f["supplier_id"], f["cutoff"] = S, T
    return f


def snapshots(fb: FeatureBuilder, cutoffs, require_delivered_by=None) -> pd.DataFrame:
    po = fb.po
    panel = supplier_panel(fb, cutoffs).set_index(["cutoff", "supplier_id"])
    rows = []
    for c in cutoffs:
        book = open_book(po, c)
        if require_delivered_by is not None:
            book = book[book.delivered_date.notna() & (book.delivered_date <= pd.Timestamp(require_delivered_by))]
        if book.empty:
            continue
        X = po_features(fb, book, c, panel.loc[pd.Timestamp(c)])
        X["y"] = book.delay_days
        X["cutoff"] = pd.Timestamp(c)
        X["po_id"] = book.po_id
        X["supplier_id"] = book.supplier_id
        rows.append(X)
    return pd.concat(rows, ignore_index=True)


def baseline_delay(X: pd.DataFrame) -> np.ndarray:
    """Buyer's rule of thumb: the supplier's recent mean delay (a PO cannot arrive before today)."""
    return np.maximum(-X["days_to_promise"].to_numpy(dtype=float), np.nan_to_num(X["d90_delay"].to_numpy(dtype=float)))


def fit_quantiles(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> dict:
    """Quantile GBMs on the *residual* over the baseline: the model only learns corrections to the
    supplier's recent behaviour, which keeps it robust when regimes change (shallow trees, big leaves)."""
    models = {}
    r = np.asarray(y, dtype=float) - baseline_delay(X)
    for q in QUANTILES:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=60, learning_rate=0.05,
                                          max_depth=3, min_samples_leaf=800, l2_regularization=1.0,
                                          categorical_features=[FEATURES.index("commodity_code")],
                                          random_state=seed)
        models[q] = m.fit(X[FEATURES], r)
    return models


def predict_quantiles(models: dict, X: pd.DataFrame, offsets: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """(n, 3) array of P10, P50, P90 delay (days), monotone, with conformal tail offsets applied."""
    P = np.column_stack([models[q].predict(X[FEATURES]) for q in QUANTILES]) + baseline_delay(X)[:, None]
    P.sort(axis=1)
    lo, med, hi = P[:, 0] - offsets[0], P[:, 1], P[:, 2] + offsets[1]
    lo, hi = np.minimum(lo, med), np.maximum(hi, med)
    # a PO still open at the cutoff cannot arrive before it
    floor = -X["days_to_promise"].to_numpy(dtype=float)
    return np.column_stack([np.maximum(lo, floor), np.maximum(med, floor), np.maximum(hi, floor)])


def sample_delays(P: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Monte-Carlo delays from the 3 quantiles (piecewise-linear inverse CDF, heavier right tail)."""
    u = rng.random((n, len(P)))
    lo, med, hi = P[:, 0], P[:, 1], P[:, 2]
    q0 = lo - 0.5 * (med - lo)
    q1 = hi + 0.5 * (hi - med)
    xs = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    out = np.empty_like(u)
    for j in range(len(P)):
        out[:, j] = np.interp(u[:, j], xs, [q0[j], lo[j], med[j], hi[j], q1[j]])
    return out


def pinball(y, q, tau):
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def conformal_offsets(tr: pd.DataFrame) -> tuple[tuple[float, float], dict]:
    """Conformalized quantile regression (Romano et al., 2019) with expanding-window folds.

    Each fold trains on the past and scores the next year after a 3-month gap; tail residuals are
    pooled over folds so the offsets reflect several regimes (COVID, shortages, recovery).
    """
    folds = [("2021-06-30", "2021-10-01", "2022-06-30"), ("2022-06-30", "2022-10-01", "2023-06-30"),
             ("2023-06-30", "2023-10-01", "2024-06-30")]
    e_lo, e_hi = [], []
    for fit_end, c0, c1 in folds:
        f = tr[tr.cutoff <= fit_end]
        f = f[f.cutoff + pd.to_timedelta(f.days_to_promise.clip(lower=0) + f.y.clip(lower=0), "D") <= fit_end]
        c = tr[(tr.cutoff >= c0) & (tr.cutoff <= c1)]
        P = predict_quantiles(fit_quantiles(f, f.y), c)
        e_lo.append(P[:, 0] - c.y.to_numpy())
        e_hi.append(c.y.to_numpy() - P[:, 2])
    e_lo, e_hi = np.concatenate(e_lo), np.concatenate(e_hi)
    off = (float(np.quantile(e_lo, 0.9)), float(np.quantile(e_hi, 0.9)))
    return off, {"raw_coverage_cv": float(np.mean((e_lo <= 0) & (e_hi <= 0)))}


def train(fb: FeatureBuilder) -> tuple[dict, dict]:
    train_cut = pd.date_range("2019-03-01", TRAIN_END, freq="MS")
    test_cut = pd.date_range(TEST_START, "2026-01-01", freq="MS")
    tr = snapshots(fb, train_cut, require_delivered_by=TRAIN_END)
    te = snapshots(fb, test_cut, require_delivered_by=AS_OF)
    offsets, cv = conformal_offsets(tr)
    models = fit_quantiles(tr, tr.y)
    raw = predict_quantiles(models, te)
    P = predict_quantiles(models, te, offsets)
    y = te.y.to_numpy()
    naive = te.overdue_days.to_numpy().astype(float)                       # MRP: arrives as promised
    trailing = baseline_delay(te)                                          # supplier's recent mean delay
    h = pd.cut(te.days_to_promise, [-1e9, 0, 30, 90, 1e9], labels=["overdue", "due ≤30 d", "due 31-90 d", "due >90 d"])
    by_h = pd.DataFrame({"h": h, "model": np.abs(y - P[:, 1]), "promised": np.abs(y - naive),
                         "trailing": np.abs(y - trailing)}).groupby("h", observed=True).mean()
    metrics = {
        "n_train_snapshots": int(len(tr)), "n_test_snapshots": int(len(te)),
        "n_test_pos": int(te.po_id.nunique()),
        "train_cutoffs": [str(train_cut[0].date()), str(train_cut[-1].date())],
        "test_cutoffs": [str(test_cut[0].date()), str(test_cut[-1].date())],
        "mae_model_p50": float(np.mean(np.abs(y - P[:, 1]))),
        "mae_promised_date": float(np.mean(np.abs(y - naive))),
        "mae_supplier_trailing_mean": float(np.mean(np.abs(y - trailing))),
        "mae_by_horizon": {k: {c: round(float(v), 2) for c, v in r.items()} for k, r in by_h.iterrows()},
        "pinball_p10": pinball(y, P[:, 0], 0.1), "pinball_p90": pinball(y, P[:, 2], 0.9),
        "interval_80_coverage_raw": float(np.mean((y >= raw[:, 0]) & (y <= raw[:, 2]))),
        "interval_80_coverage": float(np.mean((y >= P[:, 0]) & (y <= P[:, 2]))),
        "conformal_offsets_days": list(offsets), **cv,
        "late_po_share_test": float(np.mean(y > 3)),
    }
    imp = feature_importance(models[0.5], te.sample(min(4000, len(te)), random_state=0))
    metrics["feature_importance"] = imp
    return {"models": models, "offsets": offsets}, metrics


def feature_importance(model, X: pd.DataFrame) -> list[dict]:
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X[FEATURES], X.y - baseline_delay(X), n_repeats=3, random_state=0,
                               scoring="neg_mean_absolute_error")
    order = np.argsort(-r.importances_mean)
    return [{"feature": FEATURES[i], "importance": float(r.importances_mean[i])} for i in order]


def lead_time_table(fb: FeatureBuilder) -> pd.DataFrame:
    """Descriptive lead-time analysis per supplier: quoted vs actual, drift, variability."""
    po = fb.po.dropna(subset=["delivered_date"])
    asof = pd.Timestamp(AS_OF)
    recent = po[po.delivered_date >= asof - pd.Timedelta(days=365)]
    before = po[(po.delivered_date < asof - pd.Timedelta(days=365)) & (po.delivered_date >= asof - pd.Timedelta(days=730))]
    g = recent.groupby("supplier_id")
    t = pd.DataFrame({
        "quoted_lt_median": g.quoted_lead_time.median(),
        "actual_lt_median": g.actual_lead_time.median(),
        "actual_lt_p90": g.actual_lead_time.quantile(0.9),
        "delay_mean": g.delay_days.mean(),
        "otd_12m": g.on_time.mean(),
        "deliveries_12m": g.size(),
    })
    t["quoted_lt_change"] = t.quoted_lt_median - before.groupby("supplier_id").quoted_lead_time.median()
    return t
