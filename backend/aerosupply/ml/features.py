"""Causal supplier features: everything is computed from information available *before* date t.

The same builder feeds the lead-time model (open purchase orders at a cutoff), the risk model
(monthly supplier snapshots) and the API (current state at AS_OF), so training and serving cannot
drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import COUNTRIES
from ..data.load import enrich_pos

DAY = np.timedelta64(1, "D")


def _days(x) -> np.ndarray:
    return np.asarray(pd.to_datetime(x).values.astype("datetime64[D]").astype(np.int64))


class Trailing:
    """Windowed sums over time-stamped rows of each key, queried at arbitrary dates on [t-w, t)."""

    def __init__(self, df: pd.DataFrame, key: str, time_col: str, value_cols: list[str]):
        self.cols = value_cols
        self.data = {}
        for k, g in df.sort_values(time_col).groupby(key):
            t = _days(g[time_col])
            v = np.c_[np.ones(len(g)), g[value_cols].to_numpy(dtype=float)]
            cs = np.vstack([np.zeros(v.shape[1]), np.cumsum(v, axis=0)])
            self.data[k] = (t, cs)

    def sums(self, keys, dates, window: int) -> np.ndarray:
        keys = np.asarray(keys)
        t_q = _days(dates)
        out = np.zeros((len(keys), 1 + len(self.cols)))
        for k in np.unique(keys):
            m = keys == k
            if k not in self.data:
                continue
            t, cs = self.data[k]
            hi = np.searchsorted(t, t_q[m], side="left")
            lo = np.searchsorted(t, t_q[m] - window, side="left")
            out[m] = cs[hi] - cs[lo]
        return out


class FeatureBuilder:
    def __init__(self, D: dict[str, pd.DataFrame]):
        self.D = D
        self.sup = D["suppliers"].set_index("supplier_id")
        po = enrich_pos(D["purchase_orders"])
        self.po = po
        dl = po.dropna(subset=["delivered_date"]).copy()
        dl["commodity"] = dl.supplier_id.map(self.sup.commodity)
        dl["delay_pos"] = dl.delay_days.clip(lower=0)
        dl["on_time_f"] = dl.on_time.astype(float)
        cols = ["on_time_f", "delay_days", "delay_sq", "qty_rejected", "qty_ordered", "actual_lead_time", "lt_sq"]
        dl["delay_sq"] = dl.delay_days ** 2
        dl["lt_sq"] = dl.actual_lead_time ** 2
        self.by_sup = Trailing(dl, "supplier_id", "delivered_date", cols)
        self.by_com = Trailing(dl, "commodity", "delivered_date", cols)
        sig = D["signals"].copy()
        sig["neg"] = (sig.polarity < 0).astype(float)
        sig["neg_sev"] = sig.neg * sig.severity
        self.sig = Trailing(sig, "supplier_id", "date", ["neg", "neg_sev"])
        self.fin = D["financials"].sort_values("published_date")
        links = D["supplier_links"]
        self.links = links
        # peer weights: two tier-1 suppliers are "peers" if they share upstream suppliers
        up = links[links.customer_id.map(self.sup.tier) == 1]
        pairs = up.merge(up, on="upstream_id", suffixes=("", "_peer"))
        pairs = pairs[pairs.customer_id != pairs.customer_id_peer]
        pairs["w"] = pairs.share * pairs.share_peer
        self.peers = pairs.groupby(["customer_id", "customer_id_peer"]).w.sum().reset_index()

    # -- elementary blocks -------------------------------------------------------------------
    @staticmethod
    def _stats(s: np.ndarray, prefix: str) -> dict:
        n = s[:, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_delay = s[:, 2] / n
            var = s[:, 3] / n - mean_delay ** 2
            lt_mean = s[:, 6] / n
            lt_var = s[:, 7] / n - lt_mean ** 2
            out = {
                f"{prefix}_n": n,
                f"{prefix}_otd": s[:, 1] / n,
                f"{prefix}_delay": mean_delay,
                f"{prefix}_delay_std": np.sqrt(np.clip(var, 0, None)),
                f"{prefix}_reject": s[:, 4] / np.where(s[:, 5] > 0, s[:, 5], np.nan),
                f"{prefix}_lt_cv": np.sqrt(np.clip(lt_var, 0, None)) / lt_mean,
            }
        return out

    def delivery_stats(self, sids, dates, window: int, prefix: str) -> pd.DataFrame:
        return pd.DataFrame(self._stats(self.by_sup.sums(sids, dates, window), prefix))

    def supplier_features(self, sids, dates) -> pd.DataFrame:
        """One row per (supplier, date) — every value uses data strictly before the date."""
        sids = np.asarray(sids)
        dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
        f = pd.concat([self.delivery_stats(sids, dates, 90, "d90"),
                       self.delivery_stats(sids, dates, 365, "d365")], axis=1)
        f["delay_trend"] = f.d90_delay - f.d365_delay
        f["otd_trend"] = f.d90_otd - f.d365_otd
        com = self.sup.commodity.reindex(sids).to_numpy()
        cs = pd.DataFrame(self._stats(self.by_com.sums(com, dates, 90), "c90"))
        f["commodity_delay_90"] = cs.c90_delay
        f["commodity_otd_90"] = cs.c90_otd
        f["rel_delay_90"] = f.d90_delay - f.commodity_delay_90
        s = self.sig.sums(sids, dates, 90)
        f["neg_signals_90"] = s[:, 1]
        f["neg_severity_90"] = s[:, 2]
        f["neg_signals_365"] = self.sig.sums(sids, dates, 365)[:, 1]
        f = pd.concat([f, self.financial(sids, dates)], axis=1)
        f["country_risk"] = [COUNTRIES[c][1] for c in self.sup.country.reindex(sids)]
        f["upstream_neg_signals_180"] = self.upstream_signals(sids, dates, 180)
        f["peer_delay_90"] = self.peer_delay(sids, dates, f)
        return f

    def financial(self, sids, dates) -> pd.DataFrame:
        q = pd.DataFrame({"supplier_id": np.asarray(sids), "date": pd.to_datetime(pd.Series(dates)).astype("datetime64[ns]").to_numpy(),
                          "_i": np.arange(len(sids))}).sort_values("date")
        fin = self.fin.rename(columns={"published_date": "date"})[["supplier_id", "date", "financial_score"]]
        fin = fin.assign(date=fin.date.astype("datetime64[ns]"))
        m = pd.merge_asof(q, fin, on="date", by="supplier_id", allow_exact_matches=False)
        q2 = q.assign(date=q.date - pd.Timedelta(days=183))
        m2 = pd.merge_asof(q2, fin, on="date", by="supplier_id", allow_exact_matches=False)
        out = pd.DataFrame({"_i": m._i.to_numpy(), "fin_score": m.financial_score.to_numpy(),
                            "fin_delta_2q": m.financial_score.to_numpy() - m2.financial_score.to_numpy()})
        return out.sort_values("_i").drop(columns="_i").reset_index(drop=True)

    def upstream_signals(self, sids, dates, window) -> np.ndarray:
        """Share-weighted negative signals on direct and second-level upstream suppliers."""
        ups = self.links.groupby("customer_id")
        out = np.zeros(len(sids))
        for s in np.unique(sids):
            m = np.asarray(sids) == s
            if s not in ups.groups:
                continue
            for _, r in ups.get_group(s).iterrows():
                out[m] += r.share * self.sig.sums(np.repeat(r.upstream_id, m.sum()), dates[m], window)[:, 1]
                if r.upstream_id in ups.groups:
                    for _, r2 in ups.get_group(r.upstream_id).iterrows():
                        out[m] += 0.5 * r.share * r2.share * self.sig.sums(
                            np.repeat(r2.upstream_id, m.sum()), dates[m], window)[:, 1]
        return out

    def peer_delay(self, sids, dates, f) -> np.ndarray:
        out = f.commodity_delay_90.to_numpy().copy()
        pg = self.peers.groupby("customer_id")
        for s in np.unique(sids):
            if s not in pg.groups:
                continue
            m = np.asarray(sids) == s
            g = pg.get_group(s)
            acc, wsum = np.zeros(m.sum()), np.zeros(m.sum())
            for peer, w in zip(g.customer_id_peer, g.w):
                st = self._stats(self.by_sup.sums(np.repeat(peer, m.sum()), dates[m], 90), "p")
                d = st["p_delay"]
                ok = ~np.isnan(d)
                acc[ok] += w * d[ok]
                wsum[ok] += w
            with np.errstate(invalid="ignore"):
                v = np.where(wsum > 0, acc / wsum, np.nan)
            out[m] = np.where(np.isnan(v), out[m], v)
        return out
