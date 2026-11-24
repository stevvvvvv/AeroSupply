"""Supplier risk scoring: probability of a severe delivery failure in the next 90 days.

Target (observable by any OEM, no hidden truth needed): over the next 90 days the supplier's
on-time delivery falls below 60 % or its mean delay exceeds 30 days (>= 3 deliveries).
Monthly snapshots of every tier-1 supplier; strictly time-ordered train / validation / test.

Sub-tier suppliers have no delivery history at the OEM, so they get a transparent rule-based
screening score instead (country, financial health, negative signals, common-cause suspicion).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import AS_OF, COUNTRIES, RISK_HORIZON_DAYS, SEVERE_DELAY_DAYS, SEVERE_OTD, TEST_START, TRAIN_END
from .features import FeatureBuilder
from .leadtime import supplier_panel

FEATURES = ["d90_otd", "d90_delay", "d90_delay_std", "d90_reject", "d365_otd", "d365_delay", "d365_lt_cv",
            "delay_trend", "otd_trend", "d30_delay", "rel_delay_90", "commodity_delay_90", "peer_delay_90",
            "open_overdue_share", "anomaly_weeks_13", "neg_signals_90", "neg_severity_90", "neg_signals_365",
            "fin_score", "fin_delta_2q", "country_risk", "upstream_neg_signals_180"]
LABELS = {
    "d90_otd": "On-time delivery (90 d)", "d90_delay": "Mean delay (90 d)", "d90_delay_std": "Delay volatility (90 d)",
    "d90_reject": "Reject rate (90 d)", "d365_otd": "On-time delivery (12 m)", "d365_delay": "Mean delay (12 m)",
    "d365_lt_cv": "Lead-time variability (12 m)", "delay_trend": "Delay trend (90 d vs 12 m)",
    "otd_trend": "OTD trend (90 d vs 12 m)", "d30_delay": "Mean delay (30 d)", "rel_delay_90": "Delay vs commodity peers",
    "commodity_delay_90": "Commodity-wide delay", "peer_delay_90": "Delay of suppliers sharing its sub-tier",
    "open_overdue_share": "Open orders already overdue", "anomaly_weeks_13": "Anomalous weeks (last quarter)",
    "neg_signals_90": "Negative signals (90 d)", "neg_severity_90": "Negative signal severity (90 d)",
    "neg_signals_365": "Negative signals (12 m)", "fin_score": "Financial score",
    "fin_delta_2q": "Financial score change (2 quarters)", "country_risk": "Country risk",
    "upstream_neg_signals_180": "Negative signals on its sub-tier (6 m)",
}


WINDOW_MONTHS = 36


def labels(fb: FeatureBuilder, sids, cutoffs) -> np.ndarray:
    end = pd.to_datetime(pd.Series(cutoffs)) + pd.Timedelta(days=RISK_HORIZON_DAYS)
    st = fb._stats(fb.by_sup.sums(sids, end, RISK_HORIZON_DAYS), "f")
    y = ((st["f_otd"] < SEVERE_OTD) | (st["f_delay"] > SEVERE_DELAY_DAYS)).astype(float)
    y[st["f_n"] < 3] = np.nan
    y[(end > pd.Timestamp(AS_OF)).to_numpy()] = np.nan
    return y


def open_overdue_share(fb: FeatureBuilder, sids, cutoffs) -> np.ndarray:
    po = fb.po
    out = np.full(len(sids), np.nan)
    for c in pd.DatetimeIndex(pd.unique(pd.to_datetime(pd.Series(cutoffs)))):
        book = po[(po.order_date < c) & (po.delivered_date.isna() | (po.delivered_date >= c))]
        share = (book.promised_date < c).groupby(book.supplier_id).mean()
        m = pd.to_datetime(pd.Series(cutoffs)).to_numpy() == c.to_datetime64()
        out[m] = share.reindex(np.asarray(sids)[m]).to_numpy()
    return out


def panel(fb: FeatureBuilder, weekly: pd.DataFrame, cutoffs) -> pd.DataFrame:
    p = supplier_panel(fb, cutoffs)
    p["open_overdue_share"] = open_overdue_share(fb, p.supplier_id.to_numpy(), p.cutoff)
    a = weekly[["supplier_id", "week", "anomaly"]].copy()
    a["anomaly"] = a.anomaly.astype(float)
    cnt = []
    for sid, c in zip(p.supplier_id, p.cutoff):
        w = a[(a.supplier_id == sid)]
        cnt.append(w[(w.week < c) & (w.week >= c - pd.Timedelta(weeks=13))].anomaly.sum())
    p["anomaly_weeks_13"] = cnt
    p["y"] = labels(fb, p.supplier_id.to_numpy(), p.cutoff)
    return p


def _models():
    return {
        "logistic_regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(C=0.3, max_iter=2000)),
        "gradient_boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                            min_samples_leaf=40, l2_regularization=1.0,
                                                            random_state=0),
    }


def _metrics(y, p) -> dict:
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "n": int(len(y)), "positive_rate": float(np.mean(y))}


def baseline_score(df: pd.DataFrame) -> np.ndarray:
    """What a buyer looks at today: last-quarter OTD (lower = riskier)."""
    return 1 - df.d90_otd.fillna(df.d90_otd.median()).to_numpy()


def train(fb: FeatureBuilder, weekly: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    cut = pd.date_range("2019-07-01", pd.Timestamp(AS_OF) - pd.Timedelta(days=RISK_HORIZON_DAYS), freq="MS")
    P = panel(fb, weekly, cut).dropna(subset=["y"])
    h = pd.Timedelta(days=RISK_HORIZON_DAYS)
    tr = P[P.cutoff + h <= pd.Timestamp(TRAIN_END)]
    te = P[P.cutoff >= pd.Timestamp(TEST_START)]
    # model selection on a time-ordered validation block at the end of the training period
    val_start = pd.Timestamp(TRAIN_END) - pd.DateOffset(months=12)
    fit, val = tr[tr.cutoff + h <= val_start - pd.Timedelta(days=1)], tr[tr.cutoff >= val_start]
    selection = {}
    for name, m in _models().items():
        m.fit(fit[FEATURES], fit.y)
        selection[name] = _metrics(val.y, m.predict_proba(val[FEATURES])[:, 1])
    best = max(selection, key=lambda k: selection[k]["roc_auc"])
    # walk-forward test: refit every quarter on the last WINDOW_MONTHS of labelled snapshots,
    # exactly how the model is operated (labels mature 90 days after each snapshot)
    p = np.full(len(te), np.nan)
    static = _models()[best].fit(tr[FEATURES], tr.y).predict_proba(te[FEATURES])[:, 1]
    others = {}
    for q0 in pd.date_range(te.cutoff.min(), te.cutoff.max(), freq="QS"):
        hist = P[(P.cutoff + h <= q0) & (P.cutoff >= q0 - pd.DateOffset(months=WINDOW_MONTHS))]
        m = (te.cutoff >= q0) & (te.cutoff < q0 + pd.DateOffset(months=3))
        if m.any():
            p[m.to_numpy()] = _models()[best].fit(hist[FEATURES], hist.y).predict_proba(te.loc[m, FEATURES])[:, 1]
    others["static_model_trained_to_" + str(pd.Timestamp(TRAIN_END).date())] = _metrics(te.y, static)
    healthy = (te.d90_otd >= SEVERE_OTD) & ~(te.d90_delay > SEVERE_DELAY_DAYS)
    base = baseline_score(te)
    top = te.assign(p=p).sort_values("p", ascending=False).groupby("cutoff").head(10)
    top_base = te.assign(b=base).sort_values("b", ascending=False).groupby("cutoff").head(10)
    bins = pd.cut(p, [0, .1, .2, .3, .5, .7, 1.0], include_lowest=True)
    calib = pd.DataFrame({"bin": bins.astype(str), "p": p, "y": te.y.to_numpy()}).groupby("bin", observed=True).agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()
    metrics = {
        "served_model": best, "selection_validation": selection,
        "train_snapshots": int(len(tr)), "test_snapshots": int(len(te)),
        "test_period": [str(te.cutoff.min().date()), str(te.cutoff.max().date())],
        "test_walk_forward": _metrics(te.y, p), "test_other": others,
        "baseline_last_quarter_otd": _metrics(te.y, base),
        "early_warning": {  # suppliers that look fine today
            "n": int(healthy.sum()), "positive_rate": float(te.y[healthy].mean()),
            "model_roc_auc": float(roc_auc_score(te.y[healthy], p[healthy.to_numpy()])),
            "baseline_roc_auc": float(roc_auc_score(te.y[healthy], base[healthy.to_numpy()])),
        },
        "top10_precision": {"model": float(top.y.mean()), "baseline": float(top_base.y.mean())},
        "calibration": calib.to_dict(orient="records"),
    }
    # production model: same recipe on the most recent window available at AS_OF
    recent = P[P.cutoff >= P.cutoff.max() - pd.DateOffset(months=WINDOW_MONTHS)]
    model = _models()[best].fit(recent[FEATURES], recent.y)
    ref = recent[FEATURES].median()
    metrics["production_fit"] = {"snapshots": int(len(recent)), "from": str(recent.cutoff.min().date()),
                                 "to": str(recent.cutoff.max().date())}
    metrics["global_drivers"] = global_drivers(model, te.sample(min(1500, len(te)), random_state=0), ref)
    return {"model": model, "reference": ref, "name": best}, metrics, P


def predict(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    return bundle["model"].predict_proba(X[FEATURES])[:, 1]


def drivers(bundle: dict, X: pd.DataFrame, top: int = 5) -> list[list[dict]]:
    """Per-row explanation: change in probability when a feature is reset to the fleet median."""
    base = predict(bundle, X)
    deltas = np.zeros((len(X), len(FEATURES)))
    for j, f in enumerate(FEATURES):
        Xr = X.copy()
        Xr[f] = bundle["reference"][f]
        deltas[:, j] = base - predict(bundle, Xr)
    out = []
    for i in range(len(X)):
        order = np.argsort(-np.abs(deltas[i]))[:top]
        out.append([{"feature": FEATURES[j], "label": LABELS[FEATURES[j]],
                     "value": None if pd.isna(X.iloc[i][FEATURES[j]]) else float(X.iloc[i][FEATURES[j]]),
                     "reference": float(bundle["reference"][FEATURES[j]]), "contribution": float(deltas[i, j])}
                    for j in order if abs(deltas[i, j]) > 0.005])
    return out


def global_drivers(model, X: pd.DataFrame, ref: pd.Series) -> list[dict]:
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X[FEATURES], X.y, n_repeats=5, random_state=0, scoring="roc_auc")
    order = np.argsort(-r.importances_mean)
    return [{"feature": FEATURES[i], "label": LABELS[FEATURES[i]], "importance": float(r.importances_mean[i])}
            for i in order[:12]]


def screening_score(fb: FeatureBuilder, sids, cc_recent: pd.Series, at=AS_OF) -> pd.DataFrame:
    """Rule-based 0-100 screening for sub-tier suppliers (no delivery data at the OEM)."""
    sids = np.asarray(sids)
    dates = [pd.Timestamp(at)] * len(sids)
    sig = fb.sig.sums(sids, dates, 180)
    fin = fb.financial(sids, pd.Series(dates))
    country = np.array([COUNTRIES[c][1] for c in fb.sup.country.reindex(sids)])
    fin_pen = np.clip((70 - fin.fin_score.fillna(60).to_numpy()) / 40, 0, 1)
    fin_drop = np.clip(-fin.fin_delta_2q.fillna(0).to_numpy() / 20, 0, 1)
    neg = np.clip(sig[:, 2] / 6, 0, 1)
    cc = cc_recent.reindex(sids).fillna(0).to_numpy()
    s = 100 * (0.25 * country + 0.25 * fin_pen + 0.15 * fin_drop + 0.2 * neg + 0.15 * np.clip(cc, 0, 1))
    return pd.DataFrame({"supplier_id": sids, "score": s.round(1), "country_risk": country,
                         "fin_score": fin.fin_score.to_numpy(), "fin_delta_2q": fin.fin_delta_2q.to_numpy(),
                         "neg_signals_180": sig[:, 1], "common_cause_weeks_13": cc})
