"""Load the raw exports into a dict of typed DataFrames (the in-memory "dataset" used everywhere)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RAW_DIR

DATE_COLS = {
    "purchase_orders": ["order_date", "promised_date", "delivered_date"],
    "financials": ["period_end", "published_date"],
    "production_plan": ["month"],
}
TABLES = ["suppliers", "supplier_links", "parts", "part_sources", "bom", "programs", "production_plan",
          "purchase_orders", "financials"]


def typed(name: str, df: pd.DataFrame) -> pd.DataFrame:
    for c in DATE_COLS.get(name, []):
        df[c] = pd.to_datetime(df[c])
    return df


def load_raw(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    if not (raw_dir / "purchase_orders.csv").exists():
        raise FileNotFoundError(f"No raw data in {raw_dir}. Run: python -m aerosupply.data.generate")
    D = {t: typed(t, pd.read_csv(raw_dir / f"{t}.csv")) for t in TABLES}
    with open(raw_dir / "signals.jsonl", encoding="utf-8") as f:
        sig = pd.DataFrame([json.loads(line) for line in f])
    sig["date"] = pd.to_datetime(sig["date"])
    D["signals"] = sig
    return D


def enrich_pos(po: pd.DataFrame) -> pd.DataFrame:
    """Derived delivery metrics (only meaningful for delivered POs)."""
    from ..config import OTD_TOLERANCE_DAYS
    po = po.copy()
    po["delay_days"] = (po["delivered_date"] - po["promised_date"]).dt.days
    po["actual_lead_time"] = (po["delivered_date"] - po["order_date"]).dt.days
    po["on_time"] = (po["delay_days"] <= OTD_TOLERANCE_DAYS).astype(float)
    po.loc[po["delivered_date"].isna(), "on_time"] = np.nan
    po["reject_rate"] = po["qty_rejected"] / po["qty_ordered"]
    return po
