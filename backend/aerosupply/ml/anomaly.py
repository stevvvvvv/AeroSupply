"""Anomaly detection on delivery histories.

Three layers, all causal (week w only uses deliveries up to w):

1. Supplier anomalies — each tier-1 supplier is compared with *its own* recent baseline after
   removing what its commodity peers experience the same week (so a market-wide shortage does not
   flag every supplier). Robust z-scores (median / MAD) on delay, on-time rate and reject rate are
   combined by an Isolation Forest; an anomaly must persist 2 weeks to open an episode.
2. Systemic tension — the same idea at commodity level against a 1-year baseline.
3. Common cause — through the supplier network: when several tier-1 customers of the same sub-tier
   supplier become anomalous together, the sub-tier supplier is suspected (OEMs usually have no
   delivery data on tier-2/3, so this is how a hidden problem upstream becomes visible).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..config import AS_OF, HISTORY_START, TRAIN_END
from .features import FeatureBuilder

WINDOW = 35            # deliveries aggregated over 5 weeks
BASE_WEEKS = (30, 5)   # baseline = weeks w-30 .. w-5 (excludes the most recent month)
FLOORS = {"delay": 2.0, "otd": 0.08, "reject": 0.01}
PERSIST = 2
Z_RULE = 3.0
CONTAMINATION_PCT = 97.5


def weeks() -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(HISTORY_START) + pd.Timedelta(days=6), AS_OF, freq="W-SUN")


def _robust_z(x: pd.Series, floor: float) -> pd.Series:
    lo, hi = BASE_WEEKS
    base = x.shift(hi).rolling(lo - hi, min_periods=8)
    med = base.median()
    mad = (x.shift(hi) - med).abs().rolling(lo - hi, min_periods=8).median() * 1.4826
    return (x - med) / np.maximum(mad, floor)


def weekly_table(fb: FeatureBuilder) -> pd.DataFrame:
    W = weeks()
    q = W + pd.Timedelta(days=1)                     # [end-27, end] inclusive
    sids = fb.sup.index[fb.sup.tier == 1].to_numpy()
    S = np.repeat(sids, len(W))
    T = np.tile(q.to_numpy(), len(sids))
    ss = fb.by_sup.sums(S, T, WINDOW)
    st = pd.DataFrame(fb._stats(ss, "w"))
    com = fb.sup.commodity.reindex(S).to_numpy()
    # leave-one-out peer baseline: the commodity without the supplier itself
    cs = pd.DataFrame(fb._stats(fb.by_com.sums(com, T, WINDOW) - ss, "c"))
    df = pd.DataFrame({"supplier_id": S, "week": np.tile(W.to_numpy(), len(sids)), "commodity": com,
                       "n": st.w_n, "otd": st.w_otd, "delay": st.w_delay, "reject": st.w_reject,
                       "c_otd": cs.c_otd, "c_delay": cs.c_delay})
    df.loc[df.n < 2, ["otd", "delay", "reject"]] = np.nan
    df["rel_delay"] = df.delay - df.c_delay
    df["rel_otd"] = df.otd - df.c_otd
    g = df.groupby("supplier_id", group_keys=False)
    df["z_delay"] = g.rel_delay.apply(lambda s: _robust_z(s, FLOORS["delay"]))
    df["z_otd"] = -g.rel_otd.apply(lambda s: _robust_z(s, FLOORS["otd"]))
    df["z_reject"] = g.reject.apply(lambda s: _robust_z(s, FLOORS["reject"]))
    # own-baseline only (not peer-adjusted): used by the common-cause layer, where peers may share the culprit
    df["z_delay_abs"] = g.delay.apply(lambda s: _robust_z(s, FLOORS["delay"]))
    return df


Z_COLS = ["z_delay", "z_otd", "z_reject"]


def _bad_side(df: pd.DataFrame) -> np.ndarray:
    return np.clip(df[Z_COLS].to_numpy(), 0, 8)   # only deteriorations matter


def fit_detector(df: pd.DataFrame, seed: int = 0) -> dict:
    tr = df[(df.week <= pd.Timestamp(TRAIN_END))].dropna(subset=Z_COLS)
    iso = IsolationForest(n_estimators=200, max_samples=1024, random_state=seed).fit(_bad_side(tr))
    s = -iso.score_samples(_bad_side(tr))
    return {"iso": iso, "threshold": float(np.percentile(s, CONTAMINATION_PCT))}


def score(df: pd.DataFrame, det: dict) -> pd.DataFrame:
    df = df.copy()
    ok = df[Z_COLS].notna().all(axis=1)
    df["score"] = np.nan
    df.loc[ok, "score"] = -det["iso"].score_samples(_bad_side(df[ok]))
    df["flag_raw"] = (df.score > det["threshold"]).fillna(False)
    df["flag_rule"] = (df[Z_COLS].max(axis=1) > Z_RULE).fillna(False)
    df["anomaly"] = _persist(df, "flag_raw")
    df["anomaly_rule"] = _persist(df, "flag_rule")
    df["flag_abs"] = (df.z_delay_abs > Z_RULE).fillna(False)
    df["degraded"] = _persist(df, "flag_abs")
    return df


def _persist(df: pd.DataFrame, col: str) -> pd.Series:
    """Episode opens after PERSIST consecutive flags, closes after PERSIST consecutive clear weeks."""
    out = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("supplier_id").indices.items():
        f = df[col].to_numpy()[idx]
        state, on, off = False, 0, 0
        for k, v in enumerate(f):
            on, off = (on + 1, 0) if v else (0, off + 1)
            if not state and on >= PERSIST:
                state = True
                out[idx[k - PERSIST + 1:k + 1]] = True
            elif state and off >= PERSIST:
                state = False
            if state:
                out[idx[k]] = True
    return pd.Series(out, index=df.index)


def episodes(df: pd.DataFrame, col: str = "anomaly") -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby("supplier_id"):
        a = g[col].to_numpy()
        wk = g.week.to_numpy()
        start = None
        for k in range(len(a)):
            if a[k] and start is None:
                start = k
            if start is not None and (not a[k] or k == len(a) - 1):
                end = k if a[k] else k - 1
                seg = g.iloc[start:end + 1]
                rows.append({"supplier_id": sid, "start": pd.Timestamp(wk[start]), "end": pd.Timestamp(wk[end]),
                             "weeks": end - start + 1, "peak_score": float(seg.score.max()),
                             "peak_delay": float(seg.delay.max()), "min_otd": float(seg.otd.min()),
                             "ongoing": bool(k == len(a) - 1 and a[k])})
                start = None
    return pd.DataFrame(rows)


def systemic(df: pd.DataFrame) -> pd.DataFrame:
    c = df.groupby(["commodity", "week"]).agg(c_delay=("c_delay", "first"), c_otd=("c_otd", "first")).reset_index()
    out = []
    for com, g in c.groupby("commodity"):
        g = g.sort_values("week").copy()
        base = g.c_delay.shift(4).rolling(52, min_periods=20)
        med = base.median()
        mad = (g.c_delay.shift(4) - med).abs().rolling(52, min_periods=20).median() * 1.4826
        g["z"] = (g.c_delay - med) / np.maximum(mad, 2.0)
        g["tension"] = (g.z > Z_RULE) & (g.c_delay > 7)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def common_cause(df: pd.DataFrame, links: pd.DataFrame, min_customers: int = 2, min_share: float = 0.5) -> pd.DataFrame:
    """Sub-tier suppliers whose tier-1 customers are anomalous at the same time."""
    t1 = set(df.supplier_id.unique())
    l1 = links[links.customer_id.isin(t1)]
    act = df[df.anomaly | df.degraded][["supplier_id", "week"]]
    rows = []
    for u, g in l1.groupby("upstream_id"):
        cust = set(g.customer_id)
        if len(cust) < min_customers:
            continue
        a = act[act.supplier_id.isin(cust)].groupby("week").supplier_id.nunique()
        share = a / len(cust)
        hit = a[(a >= min_customers) & (share >= min_share)]
        for wk, n in hit.items():
            rows.append({"upstream_id": u, "week": wk, "customers_anomalous": int(n), "customers": len(cust),
                         "share": float(n / len(cust))})
    return pd.DataFrame(rows, columns=["upstream_id", "week", "customers_anomalous", "customers", "share"])


# ---------------------------------------------------------------------------------- evaluation
def evaluate(df: pd.DataFrame, eps: pd.DataFrame, truth_dis: pd.DataFrame, truth_eff: pd.DataFrame,
             links: pd.DataFrame, cc: pd.DataFrame, col: str = "anomaly") -> dict:
    """Compare detections with the generator's hidden ground truth."""
    t1 = set(df.supplier_id.unique())
    asof = pd.Timestamp(AS_OF)
    d = truth_dis[(truth_dis.scope == "supplier") & truth_dis.supplier_id.isin(t1) & (truth_dis.delay_k >= 20)].copy()
    d["onset"] = pd.to_datetime(d.onset_date)
    d["full"] = pd.to_datetime(d.full_date)
    d = d[(d.onset >= pd.Timestamp("2019-09-01")) & (d.onset <= asof - pd.Timedelta(days=120))]
    lags, det = [], 0
    for _, r in d.iterrows():
        e = eps[(eps.supplier_id == r.supplier_id) & (eps.start >= r.onset - pd.Timedelta(days=14))
                & (eps.start <= r.full + pd.Timedelta(days=120))]
        if len(e):
            det += 1
            lags.append((e.start.min() - r.onset).days)
    # precision: an episode is "true" if the supplier really was worse than its peers around it
    eff = truth_eff.copy()
    eff.index = pd.to_datetime(eff.index)
    t1_cols = [c for c in eff.columns if c in t1]
    com = df.drop_duplicates("supplier_id").set_index("supplier_id").commodity
    excess = pd.DataFrame(index=eff.index)
    for c, ids in com.groupby(com):
        cols = [i for i in ids.index if i in t1_cols]
        med = eff[cols].median(axis=1)
        for i in cols:
            excess[i] = eff[i] - med
    true_ep = 0
    for _, e in eps.iterrows():
        win = excess.loc[e.start - pd.Timedelta(days=42): e.end + pd.Timedelta(days=28), e.supplier_id]
        true_ep += bool((win >= 5).any())
    # upstream: tier-2/3 disruptions, did the common-cause layer point at the culprit?
    up = truth_dis[(truth_dis.scope == "supplier") & ~truth_dis.supplier_id.isin(t1) & (truth_dis.delay_k >= 30)].copy()
    up["onset"] = pd.to_datetime(up.onset_date)
    up = up[(up.onset >= pd.Timestamp("2019-09-01")) & (up.onset <= asof - pd.Timedelta(days=150))]
    has_cust = set(links[links.customer_id.isin(t1)].groupby("upstream_id").customer_id.nunique().loc[lambda s: s >= 2].index)
    up = up[up.supplier_id.isin(has_cust)]
    up_hit = 0
    for _, r in up.iterrows():
        c = cc[(cc.upstream_id == r.supplier_id) & (cc.week >= r.onset) & (cc.week <= r.onset + pd.Timedelta(days=240))]
        up_hit += bool(len(c))
    # common-cause alerts (contiguous weeks per upstream supplier): was there a real cause upstream?
    cc_alerts, cc_true = 0, 0
    sup_com = pd.concat([links.drop_duplicates("upstream_id").set_index("upstream_id").input_commodity])
    real = truth_dis.copy()
    real["onset"], real["end"] = pd.to_datetime(real.onset_date), pd.to_datetime(real.end_date)
    for u, g in cc.sort_values("week").groupby("upstream_id"):
        starts = g.week[g.week.diff() != pd.Timedelta(days=7)]
        for st in starts:
            cc_alerts += 1
            r = real[((real.supplier_id == u) | ((real.scope == "systemic") & (real.commodity == sup_com.get(u))))
                     & (real.onset <= st + pd.Timedelta(days=30)) & (real.end >= st - pd.Timedelta(days=90))]
            cc_true += bool(len(r))
    weeks_scored = int(df.score.notna().sum())
    return {
        "disruptions_evaluated": int(len(d)), "detected": int(det),
        "recall": float(det / max(len(d), 1)),
        "median_detection_lag_days": float(np.median(lags)) if lags else None,
        "episodes": int(len(eps)), "episode_precision": float(true_ep / max(len(eps), 1)),
        "flagged_week_rate": float(df[col].sum() / max(weeks_scored, 1)),
        "upstream_disruptions_evaluated": int(len(up)), "upstream_pointed_by_common_cause": int(up_hit),
        "common_cause_alerts": int(cc_alerts), "common_cause_precision": float(cc_true / max(cc_alerts, 1)),
    }


def run(fb: FeatureBuilder) -> tuple[dict, pd.DataFrame]:
    df = weekly_table(fb)
    det = fit_detector(df)
    df = score(df, det)
    return det, df
