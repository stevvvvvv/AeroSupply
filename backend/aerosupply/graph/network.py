"""Multi-tier dependency network (NetworkX) and disruption-impact simulation.

Graph: sub-tier supplier -> tier-1 supplier -> part -> aircraft program.

Structural dependence of a part on a supplier u = share of the part's supply lost if u stops
completely. Inputs of a supplier are complements (no forgings => no landing gear, whatever the
aluminium supply), suppliers of the same input are substitutes (weighted by share):

    loss(x, u) = 1                                                  if x == u
               = max over inputs i of  sum_{v supplies i to x} share(v, x) * loss(v, u)
    loss(part, u) = sum over tier-1 sources s of share(s, part) * loss(s, u)

This reveals *hidden* single points of failure: parts that look dual-sourced at tier-1 but whose
two sources depend on the same forging house.
"""
from __future__ import annotations

from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd

from ..config import (AS_OF, PROGRAM_PRICE_MEUR, SUBTIER_BUFFER_DAYS, SURGE_CAPACITY, SURGE_START_DAYS)


def build_graph(D: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, s in D["suppliers"].iterrows():
        G.add_node(f"S:{s.supplier_id}", kind="supplier", tier=int(s.tier), label=s["name"],
                   commodity=s.commodity, country=s.country)
    for _, p in D["parts"].iterrows():
        G.add_node(f"P:{p.part_id}", kind="part", label=p["name"], commodity=p.commodity)
    for _, r in D["programs"].iterrows():
        G.add_node(f"A:{r.program}", kind="program", label=r["name"])
    for _, l in D["supplier_links"].iterrows():
        G.add_edge(f"S:{l.upstream_id}", f"S:{l.customer_id}", share=float(l.share), input=l.input_commodity)
    for _, s in D["part_sources"].iterrows():
        G.add_edge(f"S:{s.supplier_id}", f"P:{s.part_id}", share=float(s.share))
    for _, b in D["bom"].iterrows():
        G.add_edge(f"P:{b.part_id}", f"A:{b.program}", qty=int(b.qty_per_aircraft))
    return G


class Network:
    def __init__(self, D: dict):
        self.D = D
        self.G = build_graph(D)
        self.sup = D["suppliers"].set_index("supplier_id")
        self.links = D["supplier_links"]
        self.sources = D["part_sources"]
        self.bom = D["bom"]
        self._ups = {c: g for c, g in self.links.groupby("customer_id")}
        self.loss = self._loss_matrix()          # parts x suppliers

    @lru_cache(maxsize=None)
    def supplier_loss(self, x: str, u: str) -> float:
        if x == u:
            return 1.0
        g = self._ups.get(x)
        if g is None:
            return 0.0
        return max(sum(sh * self.supplier_loss(v, u) for v, sh in zip(gi.upstream_id, gi.share))
                   for _, gi in g.groupby("input_commodity"))

    def _loss_matrix(self) -> pd.DataFrame:
        sids = list(self.sup.index)
        rows = {}
        for pid, g in self.sources.groupby("part_id"):
            rows[pid] = [sum(sh * self.supplier_loss(s, u) for s, sh in zip(g.supplier_id, g.share)) for u in sids]
        return pd.DataFrame.from_dict(rows, orient="index", columns=sids)

    # ------------------------------------------------------------------------------ metrics
    def supplier_metrics(self, spend: pd.Series | None = None) -> pd.DataFrame:
        L = self.loss
        n_sources = self.sources.groupby("part_id").supplier_id.nunique()
        prog_of_part = self.bom.groupby("part_id").program.apply(set)
        H = self.G.to_undirected()
        btw = nx.betweenness_centrality(H, normalized=True, seed=0, k=min(200, H.number_of_nodes()))
        rows = []
        for u in self.sup.index:
            col = L[u]
            dep = col[col > 1e-9]
            full = col[col >= 0.999]
            hidden = [p for p in full.index if n_sources.get(p, 1) > 1 and self.sup.loc[u, "tier"] > 1]
            progs = set().union(*[prog_of_part.get(p, set()) for p in dep.index]) if len(dep) else set()
            rows.append({
                "supplier_id": u, "parts_dependent": int(len(dep)), "parts_fully_dependent": int(len(full)),
                "hidden_spof_parts": int(len(hidden)), "hidden_spof_part_ids": hidden,
                "programs": sorted(progs), "exposure_weighted_parts": float(dep.sum()),
                "centrality": float(btw.get(f"S:{u}", 0.0)),
                "tier1_customers": int(self.links[self.links.upstream_id == u].customer_id.nunique()),
            })
        m = pd.DataFrame(rows).set_index("supplier_id")
        if spend is not None:
            m["spend_12m_meur"] = (spend.reindex(m.index).fillna(0) / 1e6).round(2)
        return m

    def hidden_spofs(self) -> pd.DataFrame:
        """Parts with >= 2 tier-1 sources that all depend on one sub-tier supplier."""
        n_sources = self.sources.groupby("part_id").supplier_id.nunique()
        multi = n_sources[n_sources > 1].index
        sub = [u for u in self.sup.index if self.sup.loc[u, "tier"] > 1]
        L = self.loss.loc[multi, sub]
        rows = []
        for pid, r in L.iterrows():
            for u, v in r[r >= 0.999].items():
                rows.append({"part_id": pid, "upstream_id": u, "tier1_sources": int(n_sources[pid])})
        return pd.DataFrame(rows, columns=["part_id", "upstream_id", "tier1_sources"])

    # --------------------------------------------------------------------------- simulation
    def simulate(self, supplier_id: str, base_paths: dict, demand: dict, duration_days: int = 90,
                 capacity_loss: float = 1.0, horizon: int = 120, start=AS_OF, as_of=AS_OF) -> dict:
        """Impact of losing ``capacity_loss`` of ``supplier_id`` for ``duration_days``.

        base_paths: median projected inventory per part (from the stock-out forecaster), so the
        impact is *incremental* over what is already expected without the disruption.
        """
        from ..data.inventory import day_idx, program_daily_rates
        a = day_idx(as_of)
        offset = max(0, (pd.Timestamp(start) - pd.Timestamp(as_of)).days)
        tier = int(self.sup.loc[supplier_id, "tier"])
        lag = SUBTIER_BUFFER_DAYS[tier]
        col = self.loss[supplier_id] * capacity_loss
        affected = col[col > 1e-9]
        t = np.arange(horizon)
        active = (t >= offset + lag) & (t < offset + lag + duration_days)
        part_rows, short_t = [], {}
        for pid, l in affected.items():
            src = self.sources[self.sources.part_id == pid]
            unaffected = sum(sh * (1 - self.supplier_loss(s, supplier_id)) for s, sh in zip(src.supplier_id, src.share))
            surge = np.where(t >= offset + lag + SURGE_START_DAYS, min(l, SURGE_CAPACITY * unaffected), 0.0)
            dem = demand[pid][a:a + horizon]
            lost = np.where(active, dem * np.clip(l - surge, 0, None), 0.0)
            # supply missed during the outage is recovered only after it ends (catch-up not modelled)
            base = base_paths[pid]["p50"][:horizon]
            disrupted = base - np.cumsum(lost)
            inc = np.clip(-disrupted, 0, None) - np.clip(-base, 0, None)
            short_t[pid] = np.clip(inc, 0, None)
            part_rows.append({"part_id": pid, "dependence": float(l / capacity_loss if capacity_loss else 0),
                              "supply_lost_share": float(l), "surge_share": float(min(l, SURGE_CAPACITY * unaffected)),
                              "units_short_max": float(short_t[pid].max()),
                              "first_short_day": int(np.argmax(short_t[pid] > 0)) if (short_t[pid] > 0).any() else None})
        prog_rows, timeline = [], {}
        # a part's shortage is shared between programs pro rata of their consumption of that part
        rates = {p: r[a] for p, r in program_daily_rates(self.D["production_plan"]).items()}
        b = self.bom.assign(use=lambda x: x.qty_per_aircraft * x.program.map(rates))
        b["prog_share"] = b.use / b.groupby("part_id").use.transform("sum")
        for prog, g in b.groupby("program"):
            per_part = [short_t[p] * sh / q for p, q, sh in zip(g.part_id, g.qty_per_aircraft, g.prog_share)
                        if p in short_t]
            ac = np.max(per_part, axis=0) if per_part else np.zeros(horizon)
            ac = np.floor(ac + 1e-9)
            timeline[prog] = ac.tolist()
            peak = float(ac.max())
            bottleneck = None
            if per_part and peak > 0:
                pids = [p for p in g.part_id if p in short_t]
                bottleneck = pids[int(np.argmax([x.max() for x in per_part]))]
            prog_rows.append({"program": prog, "aircraft_at_risk": peak,
                              "value_at_risk_meur": peak * PROGRAM_PRICE_MEUR[prog], "bottleneck_part": bottleneck,
                              "first_impact_day": int(np.argmax(ac > 0)) if peak > 0 else None})
        parts = pd.DataFrame(part_rows)
        return {
            "supplier_id": supplier_id, "tier": tier, "duration_days": duration_days, "capacity_loss": capacity_loss,
            "impact_lag_days": lag, "parts_affected": int(len(parts)),
            "parts_short": int((parts.units_short_max > 0).sum()) if len(parts) else 0,
            "aircraft_at_risk": float(sum(r["aircraft_at_risk"] for r in prog_rows)),
            "value_at_risk_meur": float(sum(r["value_at_risk_meur"] for r in prog_rows)),
            "programs": prog_rows, "parts": parts.sort_values("units_short_max", ascending=False).to_dict(orient="records") if len(parts) else [],
            "timeline": timeline,
        }

    def layout(self) -> dict:
        """Layered layout for the dashboard: x = tier (3 -> program), y spread within layer."""
        layers = {}
        for n, d in self.G.nodes(data=True):
            key = 3 - d["tier"] if d["kind"] == "supplier" else (3 if d["kind"] == "part" else 4)
            layers.setdefault(key, []).append(n)
        pos = {}
        for x, nodes in layers.items():
            # order by commodity so related suppliers sit together
            nodes.sort(key=lambda n: (self.G.nodes[n].get("commodity", ""), n))
            for i, n in enumerate(nodes):
                pos[n] = (x, (i + 0.5) / len(nodes))
        return pos
