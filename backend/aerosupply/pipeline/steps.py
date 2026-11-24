"""Pipeline steps. Each step is an independent function that reads its inputs from the stores
(SQL, MongoDB, model artifacts) and writes its outputs back, so the same code runs as Airflow tasks
in separate workers or sequentially from the CLI (``python -m aerosupply.pipeline.run``).

    extract -> load_sql -> load_documents -> train_models -> detect_anomalies
            -> forecast_stockouts -> network_impact -> score_suppliers -> optimize_mitigation
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from ..config import ARTIFACTS_DIR, AS_OF, RAW_DIR, TRUTH_DIR
from ..data.inventory import day_idx, part_daily_demand, program_daily_rates
from ..storage import sql
from ..storage.docs import get_store

LEVELS_T1 = [(0.80, "CRITICAL"), (0.50, "WARNING"), (0.20, "WATCH")]
LEVELS_SUB = [(55, "CRITICAL"), (40, "WARNING"), (28, "WATCH")]


def level(v, scale) -> str:
    if v is None or pd.isna(v):
        return "HEALTHY"
    for t, name in scale:
        if v >= t:
            return name
    return "HEALTHY"


# ------------------------------------------------------------------------------------ helpers
def dataset() -> dict:
    """Raw data as stored: relational tables from SQL, text signals from MongoDB."""
    D = sql.load()
    docs = list(get_store()["signals"].find({}, {"_id": 0}))
    sig = pd.DataFrame(docs)
    sig["date"] = pd.to_datetime(sig["date"])
    D["signals"] = sig
    return D


@dataclass
class Ctx:
    D: dict
    fb: object


_ctx: Ctx | None = None


def ctx(refresh: bool = False) -> Ctx:
    global _ctx
    if _ctx is None or refresh:
        from ..ml.features import FeatureBuilder
        D = dataset()
        _ctx = Ctx(D, FeatureBuilder(D))
    return _ctx


def artifact(name: str):
    return joblib.load(ARTIFACTS_DIR / f"{name}.joblib")


def metrics() -> dict:
    p = ARTIFACTS_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def qty_per_aircraft_equiv(D: dict) -> pd.Series:
    """Units of a part per 'average aircraft' (BOM quantities weighted by current production rates)."""
    rates = {p: r[day_idx(AS_OF)] for p, r in program_daily_rates(D["production_plan"]).items()}
    b = D["bom"].assign(rate=lambda x: x.program.map(rates))
    return b.groupby("part_id").apply(lambda g: (g.qty_per_aircraft * g.rate).sum() / g.rate.sum(), include_groups=False)


# -------------------------------------------------------------------------------------- steps
def extract(run_id: str, force: bool = False) -> dict:
    """In production: pull ERP / SRM exports and the news feed. Here: synthetic exports."""
    if (RAW_DIR / "purchase_orders.csv").exists() and not force:
        return {"skipped": True, "reason": "raw exports already present"}
    from ..data.generate import generate
    return generate(verbose=False)


def load_sql(run_id: str) -> dict:
    from ..data.load import load_raw
    D = load_raw()
    dq = sql.data_quality(D)
    failed = [c["check"] for c in dq if not c["passed"]]
    if failed:
        raise ValueError(f"data quality checks failed: {failed}")
    counts = sql.ingest(D)
    get_store()["pipeline_runs"].update_one({"run_id": run_id}, {"$set": {"data_quality": dq, "row_counts": counts}},
                                           upsert=True)
    get_store().flush()
    return {"rows": counts, "dq_checks": len(dq)}


def load_documents(run_id: str) -> dict:
    with open(RAW_DIR / "signals.jsonl", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    st = get_store()
    st.replace_collection("signals", docs, indexes=["supplier_id", "date", "signal_id"])
    st.flush()
    return {"signals": len(docs)}


def train_models(run_id: str, force: bool = False) -> dict:
    """Weekly in production (daily runs reuse the artifacts)."""
    if (ARTIFACTS_DIR / "risk.joblib").exists() and not force:
        return {"skipped": True, "reason": "artifacts present (use --retrain)"}
    from ..ml import anomaly, leadtime, risk, stockout
    c = ctx()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lt, lt_m = leadtime.train(c.fb)
    joblib.dump(lt, ARTIFACTS_DIR / "leadtime.joblib")
    det, weekly = anomaly.run(c.fb)
    joblib.dump(det, ARTIFACTS_DIR / "anomaly.joblib")
    an_m = {}
    if (TRUTH_DIR / "disruptions.csv").exists():   # evaluation against the generator's hidden truth
        eps = anomaly.episodes(weekly)
        cc = anomaly.common_cause(weekly, c.D["supplier_links"])
        td = pd.read_csv(TRUTH_DIR / "disruptions.csv")
        te = pd.read_csv(TRUTH_DIR / "delay_effect_weekly.csv", index_col=0)
        an_m = {"isolation_forest": anomaly.evaluate(weekly, eps, td, te, c.D["supplier_links"], cc),
                "z_score_rule": anomaly.evaluate(weekly, anomaly.episodes(weekly, "anomaly_rule"), td, te,
                                                 c.D["supplier_links"], cc, "anomaly_rule")}
    rk, rk_m, _ = risk.train(c.fb, weekly)
    joblib.dump(rk, ARTIFACTS_DIR / "risk.joblib")
    sf = stockout.StockoutForecaster(c.D, c.fb, lt)
    st_m = sf.backtest()
    m = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "as_of": str(AS_OF),
         "lead_time": lt_m, "anomaly": an_m, "risk": rk_m, "stockout": st_m,
         "training_seconds": round(time.time() - t0, 1)}
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    return {"lead_time_mae": lt_m["mae_model_p50"], "risk_auc": rk_m["test_walk_forward"]["roc_auc"],
            "stockout_auc_60": st_m["h60"]["roc_auc"]}


def detect_anomalies(run_id: str) -> dict:
    from ..ml import anomaly
    c = ctx()
    det = artifact("anomaly")
    weekly = anomaly.score(anomaly.weekly_table(c.fb), det)
    eps = anomaly.episodes(weekly)
    sysw = anomaly.systemic(weekly)
    cc = anomaly.common_cause(weekly, c.D["supplier_links"])
    keep = ["supplier_id", "week", "commodity", "n", "otd", "delay", "reject", "rel_delay", "c_delay", "c_otd",
            "z_delay", "z_otd", "z_reject", "score", "anomaly", "degraded"]
    sql.write_result("anomaly_weekly", weekly[keep])
    sql.write_result("anomaly_episodes", eps)
    sql.write_result("commodity_weekly", sysw[["commodity", "week", "c_delay", "c_otd", "z", "tension"]])
    sql.write_result("common_cause", cc)
    lt = __import__("aerosupply.ml.leadtime", fromlist=["x"]).lead_time_table(c.fb).reset_index()
    sql.write_result("lead_times", lt)
    return {"episodes": len(eps), "ongoing": int(eps.ongoing.sum()) if len(eps) else 0, "common_cause_weeks": len(cc)}


def forecast_stockouts(run_id: str) -> dict:
    from ..ml import stockout
    c = ctx()
    lt = artifact("leadtime")
    sf = stockout.StockoutForecaster(c.D, c.fb, lt)
    f, paths, samples = sf.forecast(AS_OF, keep_samples=True)
    f = stockout.calibrate(f, metrics().get("stockout", {}).get("calibrators"))
    parts = c.D["parts"][["part_id", "part_number", "name", "commodity", "unit_cost"]]
    f = parts.merge(f, on="part_id")
    progs = c.D["bom"].groupby("part_id").program.apply(lambda s: ",".join(sorted(s)))
    f["programs"] = f.part_id.map(progs)
    nsrc = c.D["part_sources"].groupby("part_id").supplier_id.apply(lambda s: ",".join(s))
    f["sources"] = f.part_id.map(nsrc)
    sql.write_result("part_forecasts", f)
    rows = []
    for pid, p in paths.items():
        for d in range(len(p["p50"])):
            rows.append((pid, d, p["p10"][d], p["p50"][d], p["p90"][d], p["mrp"][d]))
    sql.write_result("part_paths", pd.DataFrame(rows, columns=["part_id", "day", "p10", "p50", "p90", "mrp"]))
    joblib.dump(samples, ARTIFACTS_DIR / "shortage_samples.joblib")
    return {"parts": len(f), "p60_over_50pct": int((f.p_stockout_60 >= 0.5).sum()),
            "in_shortage_now": int(f.in_shortage_now.sum())}


def base_paths() -> dict:
    pp = sql.read_result("part_paths")
    return {pid: {"p50": g.sort_values("day").p50.to_numpy()} for pid, g in pp.groupby("part_id")}


def network_impact(run_id: str) -> dict:
    from ..graph.network import Network
    c = ctx()
    net = Network(c.D)
    po = c.fb.po
    recent = po[po.order_date >= pd.Timestamp(AS_OF) - pd.Timedelta(days=365)]
    spend = (recent.qty_ordered * recent.unit_price).groupby(recent.supplier_id).sum()
    m = net.supplier_metrics(spend)
    paths = base_paths()
    demand = part_daily_demand(c.D["bom"], c.D["production_plan"])
    impact = {}
    for sid in m.index:
        r = net.simulate(sid, paths, demand, duration_days=90, capacity_loss=1.0)
        impact[sid] = (r["aircraft_at_risk"], r["value_at_risk_meur"], r["parts_short"])
    m["impact_aircraft_90d"] = [impact[s][0] for s in m.index]
    m["impact_value_meur_90d"] = [impact[s][1] for s in m.index]
    m["impact_parts_short_90d"] = [impact[s][2] for s in m.index]
    sql.write_result("network_metrics", m.reset_index())
    hs = net.hidden_spofs()
    sql.write_result("hidden_spofs", hs)
    return {"suppliers": len(m), "hidden_spof_pairs": len(hs),
            "max_impact_aircraft": float(m.impact_aircraft_90d.max())}


def score_suppliers(run_id: str) -> dict:
    from ..ml import risk
    c = ctx()
    bundle = artifact("risk")
    weekly = sql.read_result("anomaly_weekly")
    weekly["week"] = pd.to_datetime(weekly.week)
    weekly["anomaly"] = weekly.anomaly.astype(bool)
    net = sql.read_result("network_metrics").set_index("supplier_id")
    sup = c.D["suppliers"].set_index("supplier_id")
    t1 = sup.index[sup.tier == 1]
    P = risk.panel(c.fb, weekly, [pd.Timestamp(AS_OF)])
    prob = risk.predict(bundle, P)
    drv = risk.drivers(bundle, P)
    cc = sql.read_result("common_cause")
    cc_recent = pd.Series(dtype=float)
    if len(cc):
        cc["week"] = pd.to_datetime(cc.week)
        cc_recent = cc[cc.week >= pd.Timestamp(AS_OF) - pd.Timedelta(weeks=13)].groupby("upstream_id").week.nunique() / 13
    scr = risk.screening_score(c.fb, sup.index[sup.tier > 1], cc_recent).set_index("supplier_id")
    rows, snaps = [], []
    feats = P.set_index("supplier_id")
    for i, sid in enumerate(P.supplier_id):
        f = feats.loc[sid]
        rows.append({"supplier_id": sid, "risk_prob": float(prob[i]), "risk_level": level(prob[i], LEVELS_T1),
                     "screening_score": None, "drivers": drv[i],
                     "otd_90d": f.d90_otd, "delay_90d": f.d90_delay, "otd_12m": f.d365_otd,
                     "open_overdue_share": f.open_overdue_share, "anomaly_weeks_13": f.anomaly_weeks_13,
                     "fin_score": f.fin_score, "neg_signals_90": f.neg_signals_90})
    for sid, r in scr.iterrows():
        drivers = [{"label": k, "value": None if pd.isna(v) else float(v)} for k, v in
                   [("Country risk", r.country_risk), ("Financial score", r.fin_score),
                    ("Financial score change (2 quarters)", r.fin_delta_2q),
                    ("Negative signals (6 m)", r.neg_signals_180),
                    ("Tier-1 customers degrading together (share of last 13 weeks)", r.common_cause_weeks_13)]]
        rows.append({"supplier_id": sid, "risk_prob": None, "screening_score": float(r.score),
                     "risk_level": level(r.score, LEVELS_SUB), "drivers": drivers, "otd_90d": None, "delay_90d": None,
                     "otd_12m": None, "open_overdue_share": None, "anomaly_weeks_13": None, "fin_score": r.fin_score,
                     "neg_signals_90": None})
    S = pd.DataFrame(rows).set_index("supplier_id")
    S = sup.join(S).join(net[["parts_dependent", "parts_fully_dependent", "hidden_spof_parts", "programs",
                              "centrality", "tier1_customers", "spend_12m_meur", "impact_aircraft_90d",
                              "impact_value_meur_90d", "impact_parts_short_90d"]])
    # exposure index = P(severe failure in 90 d) x aircraft at risk if the supplier stopped 90 d.
    # A ranking device (probability x impact, as in a risk matrix), not an expected loss.
    S["exposure"] = (S.risk_prob * S.impact_aircraft_90d).round(1)
    S = S.reset_index()
    sql.write_result("supplier_scores", S)
    for r in S.to_dict(orient="records"):
        snaps.append({"run_id": run_id, "as_of": str(AS_OF), **{k: (None if isinstance(v, float) and np.isnan(v) else v)
                                                               for k, v in r.items()}})
    st = get_store()
    st["risk_snapshots"].delete_many({"run_id": run_id})
    st["risk_snapshots"].insert_many(snaps)
    st["risk_snapshots"].create_index([("supplier_id", 1), ("as_of", -1)])
    st.flush()
    return {"tier1_scored": int(len(t1)), "critical": int((S.risk_level == "CRITICAL").sum()),
            "warning": int((S.risk_level == "WARNING").sum())}


def optimize_mitigation(run_id: str, budget_eur: float = 5e6) -> dict:
    from ..ml.optimize import allocate
    c = ctx()
    samples = joblib.load(ARTIFACTS_DIR / "shortage_samples.joblib")
    res = allocate(samples, c.D["parts"], qty_per_aircraft_equiv(c.D), budget_eur)
    plan = pd.DataFrame(res["parts"])
    if len(plan):
        plan = plan.merge(c.D["parts"][["part_id", "name", "commodity"]], on="part_id")
    sql.write_result("mitigation_plan", plan if len(plan) else pd.DataFrame(columns=["part_id"]))
    st = get_store()
    st["pipeline_runs"].update_one({"run_id": run_id}, {"$set": {"mitigation": {k: v for k, v in res.items() if k != "parts"}}},
                                   upsert=True)
    st.flush()
    return {k: v for k, v in res.items() if k != "parts"}


STEPS = [
    ("extract", extract),
    ("load_sql", load_sql),
    ("load_documents", load_documents),
    ("train_models", train_models),
    ("detect_anomalies", detect_anomalies),
    ("forecast_stockouts", forecast_stockouts),
    ("network_impact", network_impact),
    ("score_suppliers", score_suppliers),
    ("optimize_mitigation", optimize_mitigation),
]


def log_step(run_id: str, name: str, seconds: float, result: dict) -> None:
    st = get_store()
    st["pipeline_runs"].update_one(
        {"run_id": run_id},
        {"$set": {f"steps.{name}": {"seconds": round(seconds, 2), "result": result,
                                    "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}},
         "$setOnInsert": {"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}},
        upsert=True)
    st.flush()
