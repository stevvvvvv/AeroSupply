"""Relational store (PostgreSQL in Docker, SQLite locally).

Master and transactional data get an explicit schema (keys, foreign keys, indexes); analytical
outputs are written as flat result tables, replaced at each pipeline run (``*_current``) with the
run id, so the API always reads one consistent snapshot.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from sqlalchemy import (Boolean, Column, Date, Float, ForeignKey, Index, Integer, MetaData, String, Table,
                        create_engine, inspect, text)

from ..config import DATABASE_URL

metadata = MetaData()

suppliers = Table(
    "suppliers", metadata,
    Column("supplier_id", String(16), primary_key=True),
    Column("name", String(80), nullable=False),
    Column("tier", Integer, nullable=False, index=True),
    Column("commodity", String(40), nullable=False, index=True),
    Column("country", String(2), nullable=False),
    Column("lat", Float), Column("lon", Float),
    Column("size", String(10)), Column("employees", Integer), Column("revenue_meur", Float),
    Column("en9100", Boolean),
)
supplier_links = Table(
    "supplier_links", metadata,
    Column("upstream_id", String(16), ForeignKey("suppliers.supplier_id"), primary_key=True),
    Column("customer_id", String(16), ForeignKey("suppliers.supplier_id"), primary_key=True),
    Column("input_commodity", String(40), nullable=False),
    Column("share", Float, nullable=False),
)
programs = Table(
    "programs", metadata,
    Column("program", String(10), primary_key=True),
    Column("name", String(60)), Column("segment", String(20)),
)
parts = Table(
    "parts", metadata,
    Column("part_id", String(10), primary_key=True),
    Column("part_number", String(16)), Column("name", String(80)),
    Column("commodity", String(40), index=True),
    Column("unit_cost", Float), Column("safety_stock_days", Integer), Column("order_interval_days", Integer),
    Column("initial_stock", Float),
)
part_sources = Table(
    "part_sources", metadata,
    Column("part_id", String(10), ForeignKey("parts.part_id"), primary_key=True),
    Column("supplier_id", String(16), ForeignKey("suppliers.supplier_id"), primary_key=True),
    Column("share", Float, nullable=False),
)
bom = Table(
    "bom", metadata,
    Column("program", String(10), ForeignKey("programs.program"), primary_key=True),
    Column("part_id", String(10), ForeignKey("parts.part_id"), primary_key=True),
    Column("qty_per_aircraft", Integer, nullable=False),
)
production_plan = Table(
    "production_plan", metadata,
    Column("program", String(10), ForeignKey("programs.program"), primary_key=True),
    Column("month", Date, primary_key=True),
    Column("rate", Float, nullable=False),
)
purchase_orders = Table(
    "purchase_orders", metadata,
    Column("po_id", String(12), primary_key=True),
    Column("part_id", String(10), ForeignKey("parts.part_id"), nullable=False),
    Column("supplier_id", String(16), ForeignKey("suppliers.supplier_id"), nullable=False),
    Column("order_date", Date, nullable=False),
    Column("qty_ordered", Float, nullable=False), Column("unit_price", Float),
    Column("quoted_lead_time", Integer), Column("promised_date", Date, nullable=False),
    Column("delivered_date", Date), Column("qty_accepted", Float), Column("qty_rejected", Float),
    Column("status", String(10), index=True),
    Index("ix_po_supplier_delivered", "supplier_id", "delivered_date"),
    Index("ix_po_part_order", "part_id", "order_date"),
)
financials = Table(
    "financials", metadata,
    Column("supplier_id", String(16), ForeignKey("suppliers.supplier_id"), primary_key=True),
    Column("period_end", Date, primary_key=True),
    Column("published_date", Date, nullable=False),
    Column("financial_score", Float),
)
RAW_TABLES = ["suppliers", "programs", "parts", "supplier_links", "part_sources", "bom", "production_plan",
              "purchase_orders", "financials"]


@lru_cache(maxsize=4)
def engine(url: str = DATABASE_URL):
    kw = {"future": True}
    if url.startswith("sqlite"):
        kw["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kw)


def reset_schema(url: str = DATABASE_URL) -> None:
    eng = engine(url)
    with eng.begin() as c:
        for t in inspect(eng).get_table_names():
            if t not in metadata.tables:
                c.execute(text(f'DROP TABLE IF EXISTS "{t}"'))
    metadata.drop_all(eng)
    metadata.create_all(eng)


def ingest(D: dict, url: str = DATABASE_URL) -> dict:
    """Load the raw exports into the relational schema (full refresh)."""
    reset_schema(url)
    eng = engine(url)
    counts = {}
    for t in RAW_TABLES:
        df = D[t].copy()
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].dt.date
        cols = [c.name for c in metadata.tables[t].columns]
        df = df[cols].astype(object).where(df[cols].notna(), None)
        with eng.begin() as conn:
            conn.execute(metadata.tables[t].insert(), df.to_dict(orient="records"))
        counts[t] = len(df)
    return counts


def load(url: str = DATABASE_URL) -> dict[str, pd.DataFrame]:
    """Read the raw tables back as typed DataFrames (same structure as data.load.load_raw)."""
    from ..data.load import typed
    eng = engine(url)
    with eng.connect() as c:
        return {t: typed(t, pd.read_sql_table(t, c)) for t in RAW_TABLES}


def write_result(name: str, df: pd.DataFrame, url: str = DATABASE_URL) -> None:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object and df[c].map(lambda v: isinstance(v, (list, dict))).any():
            df[c] = df[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (list, dict)) else v)
    with engine(url).begin() as c:
        df.to_sql(name, c, if_exists="replace", index=False, chunksize=5000)


def read_result(name: str, url: str = DATABASE_URL, json_cols=()) -> pd.DataFrame:
    with engine(url).connect() as c:
        df = pd.read_sql_table(name, c)
    for col in json_cols:
        if col in df:
            df[col] = df[col].map(lambda v: json.loads(v) if isinstance(v, str) else v)
    return df


def has_table(name: str, url: str = DATABASE_URL) -> bool:
    return inspect(engine(url)).has_table(name)


def data_quality(D: dict) -> list[dict]:
    """Checks run at ingestion; failures are logged to the document store with the run."""
    po, sup, parts = D["purchase_orders"], D["suppliers"], D["parts"]
    checks = [
        ("po.supplier_fk", po.supplier_id.isin(sup.supplier_id).all()),
        ("po.part_fk", po.part_id.isin(parts.part_id).all()),
        ("po.promised_after_order", (po.promised_date > po.order_date).all()),
        ("po.delivered_after_order", ((po.delivered_date > po.order_date) | po.delivered_date.isna()).all()),
        ("po.qty_positive", (po.qty_ordered > 0).all()),
        ("po.accepted_le_ordered", ((po.qty_accepted <= po.qty_ordered) | po.qty_accepted.isna()).all()),
        ("sources.shares_sum_to_1", D["part_sources"].groupby("part_id").share.sum().round(6).eq(1).all()),
        ("links.no_self_loop", (D["supplier_links"].upstream_id != D["supplier_links"].customer_id).all()),
        ("bom.parts_used", parts.part_id.isin(D["bom"].part_id).all()),
    ]
    return [{"check": n, "passed": bool(ok)} for n, ok in checks]
