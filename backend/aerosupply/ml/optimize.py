"""Mitigation optimisation: where should a limited expediting budget go?

Decision: extra units x_i of part i bought as an expedited / spot order (premium price, arrives
within ~2 weeks). Objective: minimise the expected number of aircraft that cannot be completed,
using the Monte-Carlo shortage scenarios of the stock-out forecaster. For each part the benefit
E[min(S_i, x_i)] / qty_per_aircraft is concave in x_i (S_i = shortage in a scenario), so the greedy
allocation by marginal benefit per euro is optimal for this separable knapsack (Gross, 1956).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXPEDITE_PREMIUM = 1.35   # spot / expedite price vs contract price


def allocate(samples: dict[str, np.ndarray], parts: pd.DataFrame, qty_per_ac: pd.Series,
             budget_eur: float, step_aircraft: float = 1.0) -> dict:
    """Greedy marginal allocation in steps of one aircraft set of the part."""
    cand = {p: s for p, s in samples.items() if s.mean() > 0}
    price = parts.set_index("part_id").unit_cost * EXPEDITE_PREMIUM
    alloc = {p: 0.0 for p in cand}
    spent, rows = 0.0, []

    def gain(p, x):
        q = qty_per_ac[p]
        s = cand[p]
        return (np.minimum(s, x + step_aircraft * q).mean() - np.minimum(s, x).mean()) / q

    heap = {p: gain(p, 0.0) / (price[p] * step_aircraft * qty_per_ac[p]) for p in cand}
    while heap:
        p = max(heap, key=heap.get)
        cost = price[p] * step_aircraft * qty_per_ac[p]
        g = gain(p, alloc[p])
        if g <= 1e-4 or spent + cost > budget_eur:
            heap.pop(p)
            continue
        alloc[p] += step_aircraft * qty_per_ac[p]
        spent += cost
        heap[p] = gain(p, alloc[p]) / cost
    for p, x in alloc.items():
        if x <= 0:
            continue
        q = qty_per_ac[p]
        s = cand[p]
        rows.append({"part_id": p, "units": float(x), "aircraft_sets": float(x / q),
                     "cost_eur": float(x * price[p]),
                     "expected_aircraft_protected": float(np.minimum(s, x).mean() / q),
                     "expected_shortage_before": float(s.mean() / q),
                     "expected_shortage_after": float(np.clip(s - x, 0, None).mean() / q)})
    plan = pd.DataFrame(rows).sort_values("expected_aircraft_protected", ascending=False) if rows else pd.DataFrame()
    total_before = float(sum(s.mean() / qty_per_ac[p] for p, s in cand.items()))
    protected = float(plan.expected_aircraft_protected.sum()) if len(plan) else 0.0
    return {"budget_eur": float(budget_eur), "spent_eur": float(spent), "parts": plan.to_dict(orient="records"),
            "expected_shortage_aircraft_sets_before": total_before,
            "expected_aircraft_sets_protected": protected}
