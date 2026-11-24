"""AeroSupply API (FastAPI)."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import AS_OF, COUNTRIES, PROGRAM_PRICE_MEUR, PROGRAMS, TIER1_COMMODITIES, TIER2_COMMODITIES, TIER3_COMMODITIES
from ..data.inventory import day_idx
from ..ml.leadtime import open_book, po_features, predict_quantiles
from ..ml.optimize import allocate
from ..pipeline.steps import qty_per_aircraft_equiv
from ..storage.docs import get_store
from .state import clean, get_state, records

@asynccontextmanager
async def lifespan(_app):
    """Load pipeline outputs once at start-up so the first request is fast."""
    try:
        get_state()
    except Exception as exc:  # pipeline not run yet: endpoints will raise until it is
        print(f"[aerosupply] state not loaded at start-up: {exc}")
    yield


app = FastAPI(title="AeroSupply API", version="1.0.0", lifespan=lifespan,
              description="Supply-chain risk intelligence for aerospace manufacturing")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

COMMODITY_LABEL = {**{k: v[0] for k, v in TIER1_COMMODITIES.items()},
                   **{k: v[0] for k, v in TIER2_COMMODITIES.items()},
                   **{k: v[0] for k, v in TIER3_COMMODITIES.items()}}
LEVEL_ORDER = {"CRITICAL": 0, "WARNING": 1, "WATCH": 2, "HEALTHY": 3}


def sup_name(S, sid):
    r = S.scores.loc[S.scores.supplier_id == sid]
    return r.name.iloc[0] if len(r) else sid


@app.get("/api/health")
def health():
    return {"status": "ok", "as_of": str(AS_OF)}


@app.get("/api/meta")
def meta():
    return {"as_of": str(AS_OF), "programs": [{"program": k, "name": v[0], "segment": v[1],
                                               "price_meur": PROGRAM_PRICE_MEUR[k]} for k, v in PROGRAMS.items()],
            "commodities": COMMODITY_LABEL,
            "countries": {k: {"name": v[0], "risk": v[1]} for k, v in COUNTRIES.items()}}


@app.get("/api/overview")
def overview():
    S = get_state()
    sc, parts = S.scores, S.parts
    t1 = sc[sc.tier == 1]
    po = S.fb.po.dropna(subset=["delivered_date"])
    month_start = pd.Timestamp(AS_OF).to_period("M").to_timestamp()   # drop the incomplete current month
    hist = po[(po.delivered_date >= "2019-01-01") & (po.delivered_date < month_start)]
    m = hist.groupby(hist.delivered_date.dt.to_period("M"))
    otd = pd.DataFrame({"month": [str(p) for p in m.on_time.mean().index], "otd": m.on_time.mean().values,
                        "delay": m.delay_days.mean().values, "deliveries": m.size().values})
    asof = pd.Timestamp(AS_OF)
    last90 = po[po.delivered_date >= asof - pd.Timedelta(days=90)]
    prev = po[(po.delivered_date < asof - pd.Timedelta(days=90)) & (po.delivered_date >= asof - pd.Timedelta(days=180))]
    cw = S.commodity.copy()
    cw["week"] = pd.to_datetime(cw.week)
    latest = cw[cw.week == cw.week.max()]
    tense = latest[latest.tension].commodity.tolist()
    at_risk = parts[parts.p_stockout_60 >= 0.5]
    prog_risk = []
    bom = S.D["bom"]
    for prog in PROGRAMS:
        pp = parts[parts.part_id.isin(bom[bom.program == prog].part_id)]
        prog_risk.append({"program": prog, "parts": int(len(pp)), "parts_p60_over_50": int((pp.p_stockout_60 >= 0.5).sum()),
                          "parts_short_now": int(pp.in_shortage_now.sum()),
                          "max_p60": float(pp.p_stockout_60.max())})
    top = t1.sort_values("exposure", ascending=False).head(8)
    eps = S.episodes[S.episodes.ongoing.astype(bool)] if len(S.episodes) else S.episodes
    sigs = list(get_store()["signals"].find({"polarity": -1, "date": {"$gte": (asof - pd.Timedelta(days=60)).date().isoformat()}},
                                            {"_id": 0}).sort("date", -1).limit(8))
    for s in sigs:
        s["supplier_name"] = sup_name(S, s["supplier_id"])
    return clean({
        "as_of": str(AS_OF),
        "kpis": {
            "suppliers": int(len(sc)), "tier1": int(len(t1)), "subtier": int((sc.tier > 1).sum()),
            "otd_90d": float(last90.on_time.mean()), "otd_prev_90d": float(prev.on_time.mean()),
            "delay_90d": float(last90.delay_days.mean()),
            "tier1_by_level": t1.risk_level.value_counts().to_dict(),
            "subtier_by_level": sc[sc.tier > 1].risk_level.value_counts().to_dict(),
            "parts": int(len(parts)), "parts_p60_over_50": int(len(at_risk)),
            "parts_short_now": int(parts.in_shortage_now.sum()),
            "open_pos": int(S.fb.po.delivered_date.isna().sum()),
            "open_pos_overdue": int(((S.fb.po.delivered_date.isna()) & (S.fb.po.promised_date < asof)).sum()),
            "hidden_spof_parts": int(S.spofs.part_id.nunique()) if len(S.spofs) else 0,
            "ongoing_anomalies": int(len(eps)),
        },
        "otd_monthly": records(otd), "tense_commodities": tense, "programs": prog_risk,
        "top_suppliers": records(top[["supplier_id", "name", "commodity", "country", "risk_prob", "risk_level",
                                      "impact_aircraft_90d", "exposure", "otd_90d", "delay_90d"]]),
        "ongoing_episodes": records(eps.merge(sc[["supplier_id", "name"]], on="supplier_id")) if len(eps) else [],
        "recent_signals": sigs,
    })


@app.get("/api/suppliers")
def suppliers(tier: int | None = None, level: str | None = None, commodity: str | None = None, q: str | None = None):
    S = get_state()
    df = S.scores.copy()
    if tier:
        df = df[df.tier == tier]
    if level:
        df = df[df.risk_level == level.upper()]
    if commodity:
        df = df[df.commodity == commodity]
    if q:
        df = df[df.name.str.contains(q, case=False) | df.supplier_id.str.contains(q, case=False)]
    df = df.assign(_o=df.risk_level.map(LEVEL_ORDER)).sort_values(["_o", "exposure", "screening_score"],
                                                                   ascending=[True, False, False]).drop(columns="_o")
    cols = ["supplier_id", "name", "tier", "commodity", "country", "lat", "lon", "size", "risk_prob", "screening_score",
            "risk_level", "otd_90d", "delay_90d", "open_overdue_share", "fin_score", "parts_dependent",
            "parts_fully_dependent", "hidden_spof_parts", "programs", "spend_12m_meur", "impact_aircraft_90d",
            "impact_value_meur_90d", "exposure", "centrality", "tier1_customers"]
    return records(df[cols])


@app.get("/api/suppliers/{sid}")
def supplier(sid: str):
    S = get_state()
    r = S.scores[S.scores.supplier_id == sid]
    if r.empty:
        raise HTTPException(404, "unknown supplier")
    row = r.iloc[0].to_dict()
    links = S.D["supplier_links"]
    names = S.scores.set_index("supplier_id")
    up = links[links.customer_id == sid].assign(name=lambda x: x.upstream_id.map(names.name),
                                               level=lambda x: x.upstream_id.map(names.risk_level),
                                               tier=lambda x: x.upstream_id.map(names.tier))
    down = links[links.upstream_id == sid].assign(name=lambda x: x.customer_id.map(names.name),
                                                 level=lambda x: x.customer_id.map(names.risk_level))
    src = S.D["part_sources"]
    supplied = src[src.supplier_id == sid].merge(S.parts[["part_id", "name", "p_stockout_60", "days_of_cover", "programs"]],
                                                on="part_id")
    dep = S.net.loss[sid]
    dep = dep[dep > 1e-9].sort_values(ascending=False)
    dependent = S.parts.set_index("part_id").loc[dep.index, ["name", "programs", "p_stockout_60"]].assign(dependence=dep.values)
    fin = S.D["financials"][S.D["financials"].supplier_id == sid].sort_values("period_end")
    sig = list(get_store()["signals"].find({"supplier_id": sid}, {"_id": 0}).sort("date", -1).limit(25))
    eps = S.episodes[S.episodes.supplier_id == sid] if len(S.episodes) else S.episodes
    lt = S.lead_times[S.lead_times.supplier_id == sid]
    hs = S.spofs[S.spofs.upstream_id == sid] if len(S.spofs) else S.spofs
    po = S.fb.po[S.fb.po.supplier_id == sid]
    open_ = po[po.delivered_date.isna()]
    return clean({
        **row, "commodity_label": COMMODITY_LABEL.get(row["commodity"]), "country_name": COUNTRIES[row["country"]][0],
        "upstream": records(up), "customers": records(down), "parts_supplied": records(supplied),
        "dependent_parts": records(dependent.reset_index().rename(columns={"index": "part_id"})),
        "hidden_spof_part_ids": hs.part_id.tolist() if len(hs) else [],
        "financials": records(fin[["period_end", "financial_score"]]), "signals": sig,
        "episodes": records(eps), "lead_time": records(lt)[0] if len(lt) else None,
        "open_pos": int(len(open_)), "open_pos_overdue": int((open_.promised_date < pd.Timestamp(AS_OF)).sum()),
    })


@app.get("/api/suppliers/{sid}/weekly")
def supplier_weekly(sid: str):
    S = get_state()
    w = S.weekly[S.weekly.supplier_id == sid]
    if w.empty:
        return []
    return records(w[["week", "n", "otd", "delay", "c_delay", "rel_delay", "score", "anomaly", "degraded"]])


@app.get("/api/parts")
def parts_list(commodity: str | None = None, min_p: float = 0.0, program: str | None = None):
    S = get_state()
    df = S.parts
    if commodity:
        df = df[df.commodity == commodity]
    if program:
        df = df[df.programs.str.contains(program)]
    df = df[df.p_stockout_90 >= min_p].sort_values(["p_stockout_60", "expected_shortage_units"], ascending=False)
    return records(df.drop(columns=[c for c in df.columns if c.endswith("_raw")]))


@app.get("/api/parts/{pid}")
def part(pid: str):
    S = get_state()
    r = S.parts[S.parts.part_id == pid]
    if r.empty:
        raise HTTPException(404, "unknown part")
    row = r.iloc[0].to_dict()
    pp = S.part_paths[S.part_paths.part_id == pid].sort_values("day")
    asof = pd.Timestamp(AS_OF)
    book = open_book(S.fb.po, asof)
    book = book[book.part_id == pid]
    pos = []
    if len(book):
        X = po_features(S.fb, book, asof)
        Q = predict_quantiles(S.leadtime["models"], X, S.leadtime["offsets"])
        for (_, b), q in zip(book.iterrows(), Q):
            pos.append({"po_id": b.po_id, "supplier_id": b.supplier_id, "supplier_name": sup_name(S, b.supplier_id),
                        "order_date": b.order_date, "qty": b.qty_ordered, "promised_date": b.promised_date,
                        "overdue_days": max(0, (asof - b.promised_date).days),
                        "eta_p10": b.promised_date + pd.Timedelta(days=float(round(q[0]))),
                        "eta_p50": b.promised_date + pd.Timedelta(days=float(round(q[1]))),
                        "eta_p90": b.promised_date + pd.Timedelta(days=float(round(q[2])))})
    a = day_idx(asof)
    # actual on-hand over the last 12 months
    from ..data.inventory import on_hand_series, receipts_daily
    rec = receipts_daily(S.fb.po[S.fb.po.part_id == pid], [pid], AS_OF)[pid]
    init = float(S.D["parts"].set_index("part_id").loc[pid, "initial_stock"])
    oh = on_hand_series(init, rec, S.demand[pid])[a - 365:a]
    days_hist = pd.date_range(asof - pd.Timedelta(days=365), periods=365, freq="D")
    src = S.D["part_sources"][S.D["part_sources"].part_id == pid].assign(
        name=lambda x: x.supplier_id.map(S.scores.set_index("supplier_id").name),
        level=lambda x: x.supplier_id.map(S.scores.set_index("supplier_id").risk_level),
        risk_prob=lambda x: x.supplier_id.map(S.scores.set_index("supplier_id").risk_prob))
    dep = S.net.loss.loc[pid]
    sub = dep[(dep > 1e-9) & (S.scores.set_index("supplier_id").tier.reindex(dep.index) > 1)].sort_values(ascending=False)
    bom = S.D["bom"][S.D["bom"].part_id == pid]
    return clean({
        **row, "bom": records(bom), "sources": records(src), "open_purchase_orders": pos,
        "subtier_dependence": [{"supplier_id": k, "name": sup_name(S, k), "dependence": float(v),
                                "level": S.scores.set_index("supplier_id").risk_level.get(k)} for k, v in sub.head(10).items()],
        "forecast": {"dates": [(asof + pd.Timedelta(days=int(d))).date().isoformat() for d in pp.day],
                     "p10": pp.p10.tolist(), "p50": pp.p50.tolist(), "p90": pp.p90.tolist(), "mrp": pp.mrp.tolist()},
        "history": {"dates": [d.date().isoformat() for d in days_hist], "on_hand": oh.tolist()},
        "daily_demand_series": S.demand[pid][a:a + 120].tolist(),
    })


@app.get("/api/network")
def network():
    S = get_state()
    pos = S.net.layout()
    sc = S.scores.set_index("supplier_id")
    pf = S.parts.set_index("part_id")
    nodes = []
    for n, d in S.net.G.nodes(data=True):
        kind, key = d["kind"], n.split(":", 1)[1]
        item = {"id": n, "key": key, "kind": kind, "label": d["label"], "x": pos[n][0], "y": pos[n][1]}
        if kind == "supplier":
            item.update(tier=d["tier"], commodity=d["commodity"], level=sc.risk_level.get(key),
                        impact=sc.impact_aircraft_90d.get(key), risk_prob=sc.risk_prob.get(key),
                        hidden_spof_parts=sc.hidden_spof_parts.get(key))
        elif kind == "part":
            item.update(commodity=d["commodity"], p60=pf.p_stockout_60.get(key))
        nodes.append(item)
    edges = [{"source": u, "target": v, **{k: x for k, x in d.items()}} for u, v, d in S.net.G.edges(data=True)]
    return clean({"nodes": nodes, "edges": edges})


@app.get("/api/network/spofs")
def spofs():
    """One row per hidden-SPOF part, with every sub-tier supplier all its sources depend on."""
    S = get_state()
    if not len(S.spofs):
        return []
    sc = S.scores.set_index("supplier_id")
    rows = []
    for pid, g in S.spofs.groupby("part_id"):
        p = S.parts.set_index("part_id").loc[pid]
        ups = sorted(g.upstream_id, key=lambda u: (sc.tier.get(u), u))
        rows.append({"part_id": pid, "name": p["name"], "programs": p.programs, "p_stockout_60": p.p_stockout_60,
                     "source_names": [sc.name.get(x, x) for x in p.sources.split(",")],
                     "common_upstream": [{"supplier_id": u, "name": sc.name.get(u), "tier": sc.tier.get(u),
                                          "country": sc.country.get(u), "level": sc.risk_level.get(u)} for u in ups]})
    rows.sort(key=lambda r: (-len(r["common_upstream"]), r["part_id"]))
    return clean(rows)


class Scenario(BaseModel):
    supplier_id: str
    duration_days: int = Field(90, ge=7, le=365)
    capacity_loss: float = Field(1.0, gt=0, le=1)
    start_in_days: int = Field(0, ge=0, le=60)
    save: bool = False
    label: str | None = None


@app.post("/api/simulate")
def simulate(sc: Scenario):
    S = get_state()
    if sc.supplier_id not in S.net.sup.index:
        raise HTTPException(404, "unknown supplier")
    start = pd.Timestamp(AS_OF) + pd.Timedelta(days=sc.start_in_days)
    horizon = 180
    paths = {p: {"p50": np.concatenate([v["p50"], np.full(max(0, horizon - len(v["p50"])), v["p50"][-1])])}
             for p, v in S.paths.items()}
    res = S.net.simulate(sc.supplier_id, paths, S.demand, sc.duration_days, sc.capacity_loss, horizon=horizon, start=start)
    names = S.parts.set_index("part_id").name
    for p in res["parts"]:
        p["name"] = names.get(p["part_id"])
    for p in res["programs"]:
        p["bottleneck_name"] = names.get(p["bottleneck_part"]) if p["bottleneck_part"] else None
    res["supplier_name"] = sup_name(S, sc.supplier_id)
    res["dates"] = [(pd.Timestamp(AS_OF) + pd.Timedelta(days=d)).date().isoformat() for d in range(horizon)]
    res = clean(res)
    if sc.save:
        doc = {**res, "label": sc.label or f"{res['supplier_name']} — {sc.duration_days} d",
               "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "scenario": sc.model_dump()}
        doc.pop("timeline", None)
        st = get_store()
        st["simulations"].insert_one(doc)
        st.flush()
    return res


@app.get("/api/simulations")
def simulations():
    docs = list(get_store()["simulations"].find({}, {"_id": 0, "parts": 0}).sort("created_at", -1).limit(20))
    return clean(docs)


@app.get("/api/mitigation")
def mitigation(budget_meur: float = Query(5.0, gt=0, le=200)):
    S = get_state()
    res = allocate(S.samples, S.D["parts"], qty_per_aircraft_equiv(S.D), budget_meur * 1e6)
    names = S.parts.set_index("part_id")
    for p in res["parts"]:
        p["name"] = names.name.get(p["part_id"])
        p["commodity"] = names.commodity.get(p["part_id"])
        p["p_stockout_60"] = names.p_stockout_60.get(p["part_id"])
    curve = []
    for b in [0.5, 1, 2, 5, 10, 20, 50]:
        r = allocate(S.samples, S.D["parts"], qty_per_aircraft_equiv(S.D), b * 1e6)
        curve.append({"budget_meur": b, "protected": r["expected_aircraft_sets_protected"]})
    res["curve"] = curve
    spofs_df = S.spofs
    res["structural"] = {"hidden_spof_parts": int(spofs_df.part_id.nunique()) if len(spofs_df) else 0}
    return clean(res)


@app.get("/api/commodities")
def commodities():
    S = get_state()
    cw = S.commodity.copy()
    out = {}
    for c, g in cw.groupby("commodity"):
        out[c] = {"label": COMMODITY_LABEL.get(c), "weeks": g.week.astype(str).tolist(), "delay": g.c_delay.tolist(),
                  "otd": g.c_otd.tolist(), "tension": g.tension.astype(bool).tolist()}
    return clean(out)


@app.get("/api/signals")
def signals(supplier_id: str | None = None, days: int = 90, negative_only: bool = False):
    q = {"date": {"$gte": (pd.Timestamp(AS_OF) - pd.Timedelta(days=days)).date().isoformat()}}
    if supplier_id:
        q["supplier_id"] = supplier_id
    if negative_only:
        q["polarity"] = -1
    docs = list(get_store()["signals"].find(q, {"_id": 0}).sort("date", -1).limit(200))
    S = get_state()
    for d in docs:
        d["supplier_name"] = sup_name(S, d["supplier_id"])
    return clean(docs)


@app.get("/api/model")
def model_card():
    S = get_state()
    runs = list(get_store()["pipeline_runs"].find({}, {"_id": 0}).sort("started_at", -1).limit(1))
    return clean({"metrics": S.metrics, "last_run": runs[0] if runs else None})


@app.post("/api/admin/reload")
def reload():
    get_state(force=True)
    return {"reloaded": True}
