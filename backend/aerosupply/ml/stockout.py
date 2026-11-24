"""Stock-out forecasting by Monte-Carlo simulation of the open order book.

For every part: on-hand stock today + open purchase orders arriving at *uncertain* dates (sampled
from the lead-time model's calibrated quantiles) - consumption implied by the production plan.
Orders not yet placed are assumed to arrive at the steady-state rate after the shortest quoted
lead time. The share of scenarios where stock goes negative gives P(stock-out within 30/60/90 d).

The MRP baseline is what an ERP shows: the same projection with every PO arriving on its
promised date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..config import AS_OF, MC_RUNS, STOCKOUT_HORIZONS, TEST_START
from ..data.inventory import day_idx, on_hand_series, part_daily_demand, receipts_daily
from .features import FeatureBuilder
from .leadtime import open_book, po_features, predict_quantiles, sample_delays

HORIZON = 120


class StockoutForecaster:
    def __init__(self, D: dict, fb: FeatureBuilder, lt_bundle: dict):
        self.D, self.fb, self.lt = D, fb, lt_bundle
        self.parts = D["parts"].set_index("part_id")
        self.demand = part_daily_demand(D["bom"], D["production_plan"])
        self.rec_all = receipts_daily(D["purchase_orders"], self.parts.index, AS_OF)

    def on_hand(self, part_id: str, at) -> float:
        a = day_idx(at)
        return float(self.parts.loc[part_id, "initial_stock"] + self.rec_all[part_id][:a].sum()
                     - self.demand[part_id][:a].sum())

    def actual_path(self, part_id: str) -> np.ndarray:
        return on_hand_series(self.parts.loc[part_id, "initial_stock"], self.rec_all[part_id], self.demand[part_id])

    def forecast(self, as_of=AS_OF, runs: int = MC_RUNS, seed: int = 0, keep_samples: bool = False):
        rng = np.random.default_rng(seed)
        as_of = pd.Timestamp(as_of)
        a = day_idx(as_of)
        po = self.fb.po
        book = open_book(po, as_of)
        X = po_features(self.fb, book, as_of)
        Q = predict_quantiles(self.lt["models"], X, self.lt["offsets"])
        rej = self.fb.delivery_stats(book.supplier_id.to_numpy(), [as_of] * len(book), 180, "r").r_reject.fillna(0.01).to_numpy()
        book = book.assign(q10=Q[:, 0], q50=Q[:, 1], q90=Q[:, 2], exp_qty=book.qty_ordered.to_numpy() * (1 - rej),
                           promised_off=(book.promised_date - as_of).dt.days.to_numpy())
        src = self.D["part_sources"]
        rows, paths, samples = [], {}, {}
        for pid, p in self.parts.iterrows():
            dem = self.demand[pid][a:a + HORIZON]
            oh = self.on_hand(pid, as_of)
            b = book[book.part_id == pid]
            rec = np.zeros((runs, HORIZON))
            rec_mrp = np.zeros(HORIZON)
            if len(b):
                dly = sample_delays(b[["q10", "q50", "q90"]].to_numpy(), runs, rng)
                arr = np.clip(np.rint(b.promised_off.to_numpy() + dly).astype(int), 0, None)
                ok = arr < HORIZON
                rr, cc = np.nonzero(ok)
                np.add.at(rec, (rr, arr[rr, cc]), b.exp_qty.to_numpy()[cc])
                mrp_arr = np.clip(b.promised_off.to_numpy(), 0, None)
                for d_, q in zip(mrp_arr, b.qty_ordered):
                    if d_ < HORIZON:
                        rec_mrp[d_] += q
            # orders not placed yet: steady-state supply after the shortest current quoted lead time
            sids = src[src.part_id == pid].supplier_id
            recent_q = po[(po.part_id == pid) & (po.order_date < as_of)].groupby("supplier_id").quoted_lead_time.last()
            lmin = int(recent_q.reindex(sids).min()) if recent_q.reindex(sids).notna().any() else HORIZON
            future = np.where(np.arange(HORIZON) >= lmin, dem, 0.0)
            path = oh + np.cumsum(rec + future - dem, axis=1)
            mrp = oh + np.cumsum(rec_mrp + future - dem)
            short = path < 0
            first = np.where(short.any(axis=1), short.argmax(axis=1), np.nan)
            shortage = np.clip(-path.min(axis=1), 0, None)
            dd = max(dem[:30].mean(), 1e-9)
            row = {"part_id": pid, "on_hand": oh, "daily_demand": float(dd), "days_of_cover": oh / dd,
                   "open_pos": int(len(b)), "open_qty": float(b.qty_ordered.sum()),
                   "overdue_pos": int((b.promised_off < 0).sum()), "in_shortage_now": bool(oh < 0),
                   "expected_shortage_units": float(shortage.mean()),
                   "p90_shortage_units": float(np.quantile(shortage, 0.9)),
                   # median day of first stock-out (defined only if it happens in at least half the scenarios)
                   "median_first_stockout_day": float(np.median(np.where(np.isnan(first), 1e9, first)))
                   if np.mean(~np.isnan(first)) >= 0.5 else None}
            for h in STOCKOUT_HORIZONS:
                row[f"p_stockout_{h}"] = float(short[:, :h].any(axis=1).mean())
                row[f"mrp_stockout_{h}"] = bool((mrp[:h] < 0).any())
            mrp_short = mrp < 0
            row["mrp_first_stockout_day"] = float(mrp_short.argmax()) if mrp_short.any() else None
            rows.append(row)
            paths[pid] = {"p10": np.quantile(path, 0.1, axis=0), "p50": np.quantile(path, 0.5, axis=0),
                          "p90": np.quantile(path, 0.9, axis=0), "mrp": mrp}
            if keep_samples:
                samples[pid] = shortage
        out = pd.DataFrame(rows)
        return (out, paths, samples) if keep_samples else (out, paths)

    def backtest(self, cutoffs=None, runs: int = 200) -> dict:
        if cutoffs is None:
            cutoffs = pd.date_range(TEST_START, pd.Timestamp(AS_OF) - pd.Timedelta(days=max(STOCKOUT_HORIZONS)), freq="MS")
        res = []
        for c in cutoffs:
            f, _ = self.forecast(c, runs=runs, seed=1)
            a = day_idx(c)
            for pid in f.part_id:
                act = self.actual_path(pid)
                for h in STOCKOUT_HORIZONS:
                    f.loc[f.part_id == pid, f"y_{h}"] = bool((act[a:a + h] < 0).any())
            f["cutoff"] = c
            res.append(f)
        R = pd.concat(res, ignore_index=True)
        ev = R[~R.in_shortage_now].copy()
        # time-split recalibration: isotonic fitted on the first half of the backtest, scored on the second
        mid = ev.cutoff.sort_values().unique()[len(ev.cutoff.unique()) // 2]
        cal, hold = ev[ev.cutoff < mid], ev[ev.cutoff >= mid]
        out = {"cutoffs": [str(cutoffs[0].date()), str(cutoffs[-1].date())], "part_months": int(len(ev)),
               "excluded_already_short": int(R.in_shortage_now.sum()),
               "calibration_fit": [str(cal.cutoff.min().date()), str(cal.cutoff.max().date())],
               "holdout": [str(hold.cutoff.min().date()), str(hold.cutoff.max().date())], "calibrators": {}}
        for h in STOCKOUT_HORIZONS:
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(cal[f"p_stockout_{h}"], cal[f"y_{h}"].astype(float))
            y = hold[f"y_{h}"].astype(bool).to_numpy()
            raw = hold[f"p_stockout_{h}"].to_numpy()
            p = iso.predict(raw)
            mrp = hold[f"mrp_stockout_{h}"].astype(bool).to_numpy()
            mc = p >= 0.5
            out[f"h{h}"] = {
                "n": int(len(y)), "positive_rate": float(y.mean()), "roc_auc": float(roc_auc_score(y, raw)),
                "brier_raw": float(brier_score_loss(y, raw)), "brier": float(brier_score_loss(y, p)),
                "mc_precision": _prec(y, mc), "mc_recall": _rec(y, mc),
                "mc_precision_at_0.2": _prec(y, p >= 0.2), "mc_recall_at_0.2": _rec(y, p >= 0.2),
                "mrp_precision": _prec(y, mrp), "mrp_recall": _rec(y, mrp),
            }
            full = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(ev[f"p_stockout_{h}"], ev[f"y_{h}"].astype(float))
            out["calibrators"][h] = {"x": full.X_thresholds_.tolist(), "y": full.y_thresholds_.tolist()}
        p60 = np.interp(hold.p_stockout_60, out["calibrators"][60]["x"], out["calibrators"][60]["y"])
        bins = pd.cut(p60, [0, .05, .2, .4, .6, .8, 1.0], include_lowest=True)
        out["calibration_60_holdout"] = pd.DataFrame({"bin": bins.astype(str), "p": p60, "y": hold.y_60.astype(float)}) \
            .groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")) \
            .reset_index().to_dict(orient="records")
        return out


def calibrate(f: pd.DataFrame, calibrators: dict | None) -> pd.DataFrame:
    """Apply the backtest isotonic maps to raw Monte-Carlo probabilities (kept as *_raw)."""
    f = f.copy()
    for h in STOCKOUT_HORIZONS:
        f[f"p_stockout_{h}_raw"] = f[f"p_stockout_{h}"]
        if calibrators and (h in calibrators or str(h) in calibrators):
            c = calibrators.get(h) or calibrators.get(str(h))
            f[f"p_stockout_{h}"] = np.interp(f[f"p_stockout_{h}"], c["x"], c["y"])
        f.loc[f.in_shortage_now, f"p_stockout_{h}"] = 1.0
    # a stock-out within 30 days is also a stock-out within 60 and 90 days: keep P monotone
    cols = [f"p_stockout_{h}" for h in sorted(STOCKOUT_HORIZONS)]
    f[cols] = np.maximum.accumulate(f[cols].to_numpy(), axis=1)
    return f


def _prec(y, yhat):
    return float(y[yhat].mean()) if yhat.any() else None


def _rec(y, yhat):
    return float(yhat[y].mean()) if y.any() else None
