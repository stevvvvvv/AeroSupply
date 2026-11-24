"""Inventory arithmetic shared by the generator, the analytics and the API.

On-hand stock is never stored as a free-standing truth: it is *derived* from the initial stock,
the receipts (accepted quantities of delivered purchase orders) and the consumption implied by
the production plan. Negative on-hand = parts missing on the assembly line (back-order).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..config import RATE_PLAN, SIM_END, SIM_START

DAYS = pd.date_range(SIM_START, SIM_END, freq="D")
N_DAYS = len(DAYS)


def day_idx(d) -> int:
    return int((pd.Timestamp(d) - pd.Timestamp(SIM_START)).days)


def day_idx_array(series: pd.Series) -> np.ndarray:
    return ((pd.to_datetime(series) - pd.Timestamp(SIM_START)).dt.days).to_numpy()


def monthly_plan() -> pd.DataFrame:
    """Production plan: one row per program and month (aircraft per month)."""
    rows = []
    for prog, by_year in RATE_PLAN.items():
        for m in pd.period_range(SIM_START, SIM_END, freq="M"):
            rate = by_year[m.year]
            if m.year == 2020 and m.month <= 3:      # pre-COVID quarter at the 2019 rate
                rate = by_year[2019]
            rows.append({"program": prog, "month": m.to_timestamp().date(), "rate": float(rate)})
    return pd.DataFrame(rows)


def program_daily_rates(plan: pd.DataFrame) -> dict[str, np.ndarray]:
    """Aircraft per day for each program over the simulation grid."""
    month_of_day = DAYS.to_period("M")
    out = {}
    for prog, g in plan.groupby("program"):
        rate = pd.Series(g["rate"].to_numpy(), index=pd.PeriodIndex(pd.to_datetime(g["month"]), freq="M"))
        per_day = rate / rate.index.days_in_month
        out[prog] = per_day.reindex(month_of_day).to_numpy(dtype=float)
    return out


def part_daily_demand(bom: pd.DataFrame, plan: pd.DataFrame) -> dict[str, np.ndarray]:
    """Units consumed per day for every part (sum over programs of qty per aircraft x rate)."""
    rates = program_daily_rates(plan)
    out: dict[str, np.ndarray] = {}
    for part_id, g in bom.groupby("part_id"):
        d = np.zeros(N_DAYS)
        for prog, qty in zip(g["program"], g["qty_per_aircraft"]):
            d += qty * rates[prog]
        out[part_id] = d
    return out


def receipts_daily(pos: pd.DataFrame, part_ids, as_of: date | None = None) -> dict[str, np.ndarray]:
    """Accepted quantity received per day per part (only deliveries known at ``as_of``)."""
    delivered = pos.dropna(subset=["delivered_date"])
    if as_of is not None:
        delivered = delivered[pd.to_datetime(delivered["delivered_date"]) <= pd.Timestamp(as_of)]
    out = {p: np.zeros(N_DAYS) for p in part_ids}
    idx = day_idx_array(delivered["delivered_date"])
    for p, i, q in zip(delivered["part_id"], idx, delivered["qty_accepted"]):
        if 0 <= i < N_DAYS:
            out[p][i] += q
    return out


def on_hand_series(initial: float, receipts: np.ndarray, demand: np.ndarray) -> np.ndarray:
    """End-of-day on-hand stock (can be negative = shortage)."""
    return initial + np.cumsum(receipts) - np.cumsum(demand)
