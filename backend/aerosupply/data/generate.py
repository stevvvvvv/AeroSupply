"""Synthetic — but mechanistic — aerospace supply chain.

Why synthetic: real OEM purchase-order histories are confidential and public supply-chain datasets
(retail / e-commerce) have neither multi-tier aerospace structure nor labelled disruptions. The
generator therefore *simulates* the mechanisms the platform has to detect, so every model can be
evaluated against a known ground truth:

* a 3-tier network (raw materials -> sub-tier -> tier-1 -> 4 aircraft programs) with skewed
  concentration, including "hidden" single points of failure (two tier-1 sources sharing one
  forging house);
* a periodic-review order-up-to MRP policy that orders on *quoted* lead times;
* delays driven by supplier-specific disruptions (financial distress, quality escape, strike,
  disaster, ramp-up capacity, cyber attack), by systemic crises (COVID 2020, electronics shortage
  2021-23, castings/forgings crunch 2022-24, titanium sanctions 2022) and by upstream propagation
  through the network (Leontief: the scarcest input limits output, 30-day lag);
* quarterly financial scores that degrade *before* financial distress, and noisy text signals
  (news, audits, quality notifications), including false leads.

Supplier names are fictional on purpose. Outputs:
  data/raw/*.csv + signals.jsonl   what an OEM would actually have (ERP / SRM exports, news feed)
  data/truth/*.csv                 hidden ground truth, used only to *evaluate* models
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta

import numpy as np
import pandas as pd

from ..config import (
    AS_OF, COUNTRIES, PROGRAMS, RAW_DIR, SEED, SIM_START, TIER1_COMMODITIES, TIER2_COMMODITIES,
    TIER3_COMMODITIES, TRUTH_DIR, UPSTREAM, UPSTREAM_LAG_DAYS, UPSTREAM_PASS_THROUGH,
)
from .inventory import DAYS, N_DAYS, day_idx, monthly_plan, part_daily_demand

AS_OF_IDX = day_idx(AS_OF)

PREFIXES = [
    "Aubrac", "Vernier", "Halden", "Corvel", "Mistral", "Lumen", "Brennus", "Altair", "Norvik", "Castel",
    "Orvane", "Sorel", "Kestrel", "Valmont", "Arden", "Pyrène", "Tarn", "Iroise", "Solano", "Borea",
    "Celtis", "Quercy", "Vanta", "Meridian", "Novaro", "Ostra", "Pelican", "Rhéa", "Sable", "Talon",
    "Vesta", "Ysel", "Zenit", "Garonne", "Hexa", "Isard", "Jura", "Kairo", "Loire", "Morvan", "Nacre",
    "Opale", "Pavin", "Ravel", "Saône", "Tellis", "Verdon", "Aldra", "Belcour", "Cévenne", "Dorval",
    "Esterel", "Faucon", "Gévaudan", "Hautbois", "Ivrel", "Lyra", "Marlin", "Nérac", "Orion", "Pertuis",
    "Quimper", "Ronce", "Sirius", "Trivane", "Ubaye", "Vauclair", "Wendel", "Xaintes", "Yonne", "Zéphyr",
    "Argos", "Bastide", "Cobalt", "Draco", "Elba", "Fjord", "Granit", "Helio", "Indigo", "Kobalt",
    "Lancet", "Mosaic", "Nordis", "Osmond", "Polaris", "Quartz", "Rivage", "Strato", "Tessel", "Umbra",
    "Vector", "Wolfram", "Axial", "Boreal", "Cirrus", "Dynamo", "Equinox", "Ferrum", "Gallium", "Halcyon",
    "Ionis", "Kinetic", "Luxor", "Magnis", "Neutron", "Onyx", "Prism", "Radian", "Scalar", "Titanis",
    "Ultra", "Vortex", "Atlas", "Bryce", "Cygnus", "Delos", "Etna", "Forge", "Gaia", "Hydra",
]
SUFFIX = {
    "aerostructures": "Aerostructures", "machined_parts": "Précision", "landing_gear": "Landing Systems",
    "hydraulics": "Hydraulics", "avionics": "Avionics", "electrical": "Câblage", "interiors": "Interiors",
    "fasteners": "Fixations", "composites": "Composites", "nacelle": "Nacelles", "castings": "Fonderie",
    "forgings": "Forge", "prepreg": "Prepreg", "electronic_components": "Electronics",
    "aluminium_semi": "Aluminium", "rubber_seals": "Seals", "titanium": "Titanium",
    "nickel_superalloy": "Alloys", "aluminium_ingot": "Smelting", "carbon_fiber": "Carbon",
    "semiconductors": "Semicon",
}
N_SUPPLIERS = {
    "aerostructures": 8, "machined_parts": 14, "landing_gear": 4, "hydraulics": 6, "avionics": 8,
    "electrical": 7, "interiors": 8, "fasteners": 6, "composites": 7, "nacelle": 5,
    "castings": 6, "forgings": 6, "prepreg": 4, "electronic_components": 8, "aluminium_semi": 5,
    "rubber_seals": 4, "titanium": 5, "nickel_superalloy": 3, "aluminium_ingot": 3, "carbon_fiber": 3,
    "semiconductors": 4,
}
N_PARTS = {
    "aerostructures": 22, "machined_parts": 45, "landing_gear": 10, "hydraulics": 18, "avionics": 22,
    "electrical": 20, "interiors": 22, "fasteners": 24, "composites": 22, "nacelle": 12,
}
PART_NAMES = {
    "aerostructures": ["Fuselage section S", "Wing rib set R", "Floor beam assembly F", "Wing skin panel W",
                       "Pressure bulkhead B", "Keel beam K"],
    "machined_parts": ["Frame fitting ", "Pylon bracket ", "Door hinge fitting ", "Actuator lug ", "Seat track "],
    "landing_gear": ["Main gear leg MLG-", "Nose gear strut NLG-", "Gear brace assembly ", "Wheel & brake kit "],
    "hydraulics": ["Hydraulic pump HP-", "Servo actuator SA-", "Accumulator AC-", "Valve manifold VM-"],
    "avionics": ["Flight control computer FCC-", "Display unit DU-", "Air data module ADM-",
                 "Radio management unit RMU-", "Inertial reference unit IRU-"],
    "electrical": ["Wiring harness WH-", "Power distribution box PDB-", "Connector kit CK-"],
    "interiors": ["Passenger seat row PS-", "Galley module G-", "Lavatory unit L-", "Overhead bin set OB-",
                  "Sidewall panel kit SW-"],
    "fasteners": ["Hi-Lok pin HL", "Titanium bolt TB", "Blind rivet BR", "Lockbolt LB", "Collar C"],
    "composites": ["Flap track fairing FT-", "Belly fairing panel BF-", "Radome RD-", "Spoiler SP-",
                   "Vertical fin skin VF-"],
    "nacelle": ["Thrust reverser TR-", "Fan cowl FC-", "Inlet cowl IC-", "Exhaust nozzle EN-"],
}
QTY_RANGE = {
    "aerostructures": (1, 2), "machined_parts": (2, 16), "landing_gear": (1, 2), "hydraulics": (1, 4),
    "avionics": (1, 3), "electrical": (2, 20), "interiors": (1, 30), "fasteners": (200, 3000),
    "composites": (1, 4), "nacelle": (2, 2),
}
COUNTRY_MIX = {
    1: {"FR": .22, "DE": .12, "GB": .10, "ES": .08, "US": .12, "CA": .05, "MA": .06, "TN": .04, "MX": .05,
        "IN": .05, "PL": .04, "RO": .03, "CZ": .04},
    2: {"FR": .12, "DE": .10, "GB": .07, "IT": .07, "US": .18, "JP": .10, "CN": .12, "TR": .05, "IN": .06,
        "ES": .05, "PL": .04, "CZ": .04},
}
TIER3_COUNTRIES = {
    "titanium": ["RU", "US", "JP", "KZ", "FR"], "nickel_superalloy": ["US", "GB", "FR"],
    "aluminium_ingot": ["NO", "CA", "CN"], "carbon_fiber": ["JP", "US", "FR"],
    "semiconductors": ["CN", "JP", "US", "DE"],
}

# disruption type: (prob, ramp days, plateau days, recovery days, K = delay at full intensity)
DISRUPTION_TYPES = {
    "financial_distress": (0.25, (90, 180), (60, 240), 90, (30, 80)),
    "quality_escape": (0.25, (5, 10), (30, 120), 30, (20, 50)),
    "strike": (0.15, (1, 3), (10, 40), 20, (15, 40)),
    "natural_disaster": (0.10, (1, 3), (60, 200), 60, (50, 120)),
    "capacity_rampup": (0.15, (60, 120), (90, 240), 90, (15, 45)),
    "cyber_attack": (0.10, (1, 3), (14, 45), 20, (20, 45)),
}
EVENT_RATE = {1: 0.07, 2: 0.08, 3: 0.06}   # idiosyncratic disruptions per supplier-year

SIGNAL_TEMPLATES = {
    "financial_distress": [
        ("news", "{name} enters safeguard procedure", "Local press reports that {name} requested a safeguard procedure after two loss-making years."),
        ("credit", "Payment incidents reported for {name}", "Credit insurer flags late payments to {name}'s own suppliers."),
        ("supplier_survey", "{name} requests price revision and advance payments", "Account manager reports cash-flow tension at {name}."),
        ("news", "Job cuts announced at {name}", "{name} announced a restructuring plan and job cuts at its main plant."),
    ],
    "quality_escape": [
        ("quality", "Quality notification: nonconformity on {name} deliveries", "Incoming inspection rejected lots from {name}; containment requested."),
        ("audit", "EN 9100 major finding at {name}", "Surveillance audit raised a major finding on process control at {name}."),
    ],
    "strike": [
        ("news", "Strike at {name} production site", "Unions at {name} called an indefinite strike over wages."),
        ("supplier_survey", "{name} warns of delivery slippage (social conflict)", "Supplier notified a social conflict impacting output."),
    ],
    "natural_disaster": [
        ("news", "Fire at {name} plant", "A fire damaged part of {name}'s production hall; extent of damage under assessment."),
        ("news", "Flooding disrupts {name} operations", "Severe flooding forced {name} to halt production."),
    ],
    "capacity_rampup": [
        ("supplier_survey", "{name} struggles to follow rate increase", "{name} reports labour shortage and machine capacity limits."),
        ("audit", "Capacity assessment: {name} at risk", "Capacity review shows {name} loaded above 110 % for next two quarters."),
    ],
    "cyber_attack": [
        ("news", "Ransomware attack on {name}", "{name} shut down IT systems after a ransomware attack."),
    ],
    "sanctions": [
        ("news", "Sanctions affect {name} exports", "New sanctions and export restrictions affect shipments from {name}."),
    ],
}
NOISE_NEGATIVE = [
    ("news", "Management change at {name}", "{name} appointed a new CEO."),
    ("supplier_survey", "{name} flags raw material price increase", "Supplier announced a surcharge on raw materials."),
    ("news", "Minor incident reported at {name}", "A minor incident was reported at a {name} site; no impact announced."),
    ("credit", "Rating outlook revised for {name}", "Rating agency moved {name}'s outlook to negative."),
]
NEUTRAL_POSITIVE = [
    ("news", "{name} wins new aerospace contract", "{name} announced a multi-year contract with another OEM.", 1),
    ("audit", "{name} EN 9100 recertification passed", "Recertification audit closed without major findings.", 1),
    ("news", "{name} invests in new production line", "{name} announced a capacity investment.", 1),
    ("supplier_survey", "{name} quarterly business review", "Routine QBR, no particular issue raised.", 0),
]


class Gen:
    def __init__(self, seed: int = SEED):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------------------- network
    def suppliers(self) -> pd.DataFrame:
        rng = self.rng
        names = list(rng.permutation(PREFIXES))
        rows = []
        for tier, comms in ((1, TIER1_COMMODITIES), (2, TIER2_COMMODITIES), (3, TIER3_COMMODITIES)):
            for c in comms:
                for k in range(N_SUPPLIERS[c]):
                    if tier == 3:
                        ctry = TIER3_COUNTRIES[c][k % len(TIER3_COUNTRIES[c])]
                    else:
                        mix = COUNTRY_MIX[tier]
                        ctry = rng.choice(list(mix), p=np.array(list(mix.values())) / sum(mix.values()))
                    _, risk, lat, lon = COUNTRIES[ctry]
                    size = rng.choice(["SME", "Mid-cap", "Large"], p=[0.45, 0.35, 0.20] if tier > 1 else [0.35, 0.40, 0.25])
                    emp = {"SME": rng.integers(60, 250), "Mid-cap": rng.integers(250, 2500), "Large": rng.integers(2500, 20000)}[size]
                    if not names:
                        names = list(rng.permutation(PREFIXES))
                    rows.append({
                        "supplier_id": f"SUP-T{tier}-{len([r for r in rows if r['tier'] == tier]) + 1:03d}",
                        "name": f"{names.pop()} {SUFFIX[c]}",
                        "tier": tier, "commodity": c, "country": ctry,
                        "lat": round(lat + rng.normal(0, 1.5), 3), "lon": round(lon + rng.normal(0, 2.0), 3),
                        "size": size, "employees": int(emp),
                        "revenue_meur": round(float(emp) * rng.uniform(0.12, 0.25), 1),
                        "en9100": bool(rng.random() < (0.97 if tier == 1 else 0.85 if tier == 2 else 0.6)),
                        # hidden behaviour (truth only)
                        "_lt_factor": rng.uniform(0.8, 1.2), "_mu": rng.uniform(-3.0, 1.5),
                        "_sd": rng.uniform(1.0, 3.5), "_reject": rng.uniform(0.002, 0.02),
                        "_popularity": rng.pareto(1.2) + 0.3,
                    })
        return pd.DataFrame(rows)

    def links(self, sup: pd.DataFrame) -> pd.DataFrame:
        """Supplier -> customer supplier edges (sub-tier map) with share of the input commodity."""
        rng = self.rng
        rows = []
        for _, s in sup[sup.tier < 3].iterrows():
            for ic in UPSTREAM[s.commodity]:
                cands = sup[sup.commodity == ic]
                p = cands._popularity / cands._popularity.sum()
                k = 1 if rng.random() < 0.6 or len(cands) == 1 else 2
                chosen = rng.choice(cands.supplier_id.to_numpy(), size=k, replace=False, p=p)
                shares = [1.0] if k == 1 else ([0.7, 0.3] if rng.random() < 0.6 else [0.5, 0.5])
                for u, sh in zip(chosen, shares):
                    rows.append({"upstream_id": u, "customer_id": s.supplier_id, "input_commodity": ic, "share": sh})
        return pd.DataFrame(rows)

    def parts(self, sup: pd.DataFrame):
        rng = self.rng
        parts, sources, bom = [], [], []
        n = 0
        for c, (label, _, cost) in TIER1_COMMODITIES.items():
            cands = sup[sup.commodity == c]
            for k in range(N_PARTS[c]):
                n += 1
                pid = f"P-{n:04d}"
                tmpl = PART_NAMES[c][k % len(PART_NAMES[c])]
                parts.append({
                    "part_id": pid, "part_number": f"ASP-{rng.integers(10000, 99999)}",
                    "name": f"{tmpl}{rng.integers(10, 99)}", "commodity": c,
                    "unit_cost": round(float(cost * rng.lognormal(0, 0.35)), 2),
                    "safety_stock_days": int(rng.integers(12, 40)),
                    "order_interval_days": int(rng.choice([14, 21, 28, 35])),
                })
                ns = rng.choice([1, 2, 3], p=[0.55, 0.38, 0.07])
                ns = min(ns, len(cands))
                p = cands._popularity / cands._popularity.sum()
                chosen = rng.choice(cands.supplier_id.to_numpy(), size=ns, replace=False, p=p)
                shares = {1: [[1.0]], 2: [[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]], 3: [[0.5, 0.3, 0.2]]}[ns]
                sh = shares[rng.integers(len(shares))]
                for s_id, share in zip(chosen, sh):
                    sources.append({"part_id": pid, "supplier_id": s_id, "share": share})
                progs = [pg for pg, pr in zip(PROGRAMS, [0.7, 0.4, 0.25, 0.2]) if rng.random() < pr] or ["NB-200"]
                lo, hi = QTY_RANGE[c]
                for pg in progs:
                    q = int(rng.integers(lo, hi + 1))
                    if c == "interiors" and pg in ("RJ-72", "BJ-6"):
                        q = max(1, q // 4)
                    bom.append({"program": pg, "part_id": pid, "qty_per_aircraft": q})
        return pd.DataFrame(parts), pd.DataFrame(sources), pd.DataFrame(bom)

    # ---------------------------------------------------------------------------- disruptions
    @staticmethod
    def profile(onset: int, ramp: int, plateau: int, recovery: int) -> np.ndarray:
        t = np.arange(N_DAYS)
        up = np.clip((t - onset) / max(ramp, 1), 0, 1)
        down = np.clip(1 - (t - (onset + ramp + plateau)) / max(recovery, 1), 0, 1)
        return np.where(t < onset, 0.0, np.minimum(up, down))

    def disruptions(self, sup: pd.DataFrame, links: pd.DataFrame, parts_src: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        rows = []
        types = list(DISRUPTION_TYPES)
        probs = np.array([DISRUPTION_TYPES[t][0] for t in types])
        horizon_end = AS_OF_IDX + 60
        years = horizon_end / 365.25

        def add(supplier_id, commodity, scope, dtype, onset, ramp, plateau, recovery, k, reject=0.0):
            rows.append({
                "disruption_id": f"D-{len(rows) + 1:03d}", "supplier_id": supplier_id, "commodity": commodity,
                "scope": scope, "type": dtype, "onset_idx": int(onset), "ramp": int(ramp), "plateau": int(plateau),
                "recovery": int(recovery), "delay_k": float(k), "reject_add": float(reject),
            })

        for _, s in sup.iterrows():
            n = rng.poisson(EVENT_RATE[s.tier] * years)
            for _ in range(n):
                dt = types[rng.choice(len(types), p=probs / probs.sum())]
                _, rr, pr, rec, kr = DISRUPTION_TYPES[dt]
                onset = int(rng.integers(60, horizon_end))
                add(s.supplier_id, s.commodity, "supplier", dt, onset, rng.integers(*rr), rng.integers(*pr), rec,
                    rng.uniform(*kr), rng.uniform(0.10, 0.30) if dt == "quality_escape" else 0.0)

        # systemic crises (one row per affected commodity)
        def systemic(name, commodities, start, ramp, plateau, recovery, k):
            for c in commodities:
                add(None, c, "systemic", name, day_idx(start), ramp, plateau, recovery, k)

        systemic("covid_19", list(TIER1_COMMODITIES), "2020-03-15", 30, 200, 120, 9)
        systemic("electronics_shortage", ["electronic_components", "semiconductors"], "2021-05-01", 180, 360, 240, 28)
        systemic("castings_forgings_crunch", ["castings", "forgings"], "2022-06-01", 150, 360, 240, 16)
        systemic("titanium_market_tension", ["titanium"], "2022-03-01", 30, 300, 200, 15)
        ru = sup[(sup.commodity == "titanium") & (sup.country == "RU")]
        for sid in ru.supplier_id:
            add(sid, "titanium", "supplier", "sanctions", day_idx("2022-03-01"), 20, 330, 180, 120)

        # --- live scenarios around AS_OF (so the dashboard has something to manage today)
        cust_count = links.groupby("upstream_id").customer_id.nunique()
        forges = sup[sup.commodity == "forgings"].supplier_id
        hub = cust_count.reindex(forges).fillna(0).idxmax()
        add(hub, "forgings", "supplier", "financial_distress", AS_OF_IDX - 110, 150, 200, 90, 70)
        part_count = parts_src.groupby("supplier_id").part_id.nunique()
        av = sup[sup.commodity == "avionics"].supplier_id
        add(part_count.reindex(av).fillna(0).idxmax(), "avionics", "supplier", "quality_escape",
            AS_OF_IDX - 40, 7, 90, 30, 40, 0.2)
        st = sup[sup.commodity == "aerostructures"].supplier_id
        add(part_count.reindex(st).fillna(0).idxmax(), "aerostructures", "supplier", "strike",
            AS_OF_IDX - 12, 2, 30, 20, 30)

        d = pd.DataFrame(rows)
        d["severity"] = pd.cut(d.delay_k, [0, 25, 50, 1e9], labels=["low", "medium", "high"]).astype(str)
        d["onset_date"] = DAYS[d.onset_idx].date
        d["full_date"] = DAYS[np.minimum(d.onset_idx + d.ramp, N_DAYS - 1)].date
        d["end_date"] = DAYS[np.minimum(d.onset_idx + d.ramp + d.plateau + d.recovery, N_DAYS - 1)].date
        return d

    def effects(self, sup, links, dis):
        """Daily delay effect (days) per supplier incl. systemic + upstream propagation, and reject add."""
        own = {s: np.zeros(N_DAYS) for s in sup.supplier_id}
        rej = {s: np.zeros(N_DAYS) for s in sup.supplier_id}
        sys = {}
        for _, d in dis.iterrows():
            prof = self.profile(d.onset_idx, d.ramp, d.plateau, d.recovery)
            if d.scope == "systemic":
                sys[d.commodity] = sys.get(d.commodity, np.zeros(N_DAYS)) + prof * d.delay_k
            else:
                own[d.supplier_id] += prof * d.delay_k
                rej[d.supplier_id] += prof * d.reject_add
        eff = {}
        for tier in (3, 2, 1):
            for _, s in sup[sup.tier == tier].iterrows():
                e = own[s.supplier_id] + sys.get(s.commodity, 0.0)
                ups = links[links.customer_id == s.supplier_id]
                if len(ups):
                    by_input = []
                    for _, g in ups.groupby("input_commodity"):
                        x = sum(sh * eff[u] for u, sh in zip(g.upstream_id, g.share))
                        by_input.append(x)
                    up = np.max(by_input, axis=0)
                    up = np.concatenate([np.zeros(UPSTREAM_LAG_DAYS), up[:-UPSTREAM_LAG_DAYS]])
                    e = e + UPSTREAM_PASS_THROUGH * up
                eff[s.supplier_id] = e
        return eff, rej, own

    # ----------------------------------------------------------------------- orders / stock
    def orders(self, sup, parts, sources, bom, plan, eff, rej):
        rng = self.rng
        demand = part_daily_demand(bom, plan)
        sup_i = sup.set_index("supplier_id")
        base_lt = {c: v[1] for c, v in TIER1_COMMODITIES.items()}
        price_drift = 1.0 + 0.03 * (np.arange(N_DAYS) / 365.25)
        # trailing 90-day mean effect -> suppliers quote longer lead times when they are late
        trail = {s: pd.Series(e).rolling(90, min_periods=1).mean().to_numpy() for s, e in eff.items()}
        pos, init_stock = [], {}
        start_hist = day_idx(SIM_START)
        for _, p in parts.iterrows():
            d = demand[p.part_id]
            src = sources[sources.part_id == p.part_id]
            qlt0 = {s: int(round(base_lt[p.commodity] * sup_i.loc[s, "_lt_factor"])) for s in src.supplier_id}
            L, R = max(qlt0.values()), int(p.order_interval_days)
            init = d[: L + R].sum() + p.safety_stock_days * d[:30].mean()
            init_stock[p.part_id] = float(np.ceil(init))
            receipts = np.zeros(N_DAYS)
            open_orders: list[tuple[int, float]] = []   # (delivery idx, ordered qty)
            dem_cum = np.concatenate([[0.0], np.cumsum(d)])
            unit_price = {s: p.unit_cost * rng.uniform(0.92, 1.08) for s in src.supplier_id}
            for t in range(start_hist, AS_OF_IDX, R):
                on_hand = init + receipts[:t].sum() - dem_cum[t]
                open_orders = [(i, q) for i, q in open_orders if i >= t]
                position = on_hand + sum(q for _, q in open_orders)
                qlt = {s: int(qlt0[s] + 0.5 * max(0.0, trail[s][t])) for s in qlt0}
                Lt = max(qlt.values())
                target = dem_cum[min(t + Lt + R, N_DAYS)] - dem_cum[t] + p.safety_stock_days * d[t:t + 30].mean()
                Q = target - position
                if Q <= 0.5:
                    continue
                for s_id, share in zip(src.supplier_id, src.share):
                    q = float(np.ceil(Q * share))
                    if q <= 0:
                        continue
                    s = sup_i.loc[s_id]
                    prom = t + qlt[s_id]
                    e = eff[s_id][min(prom, N_DAYS - 1)]
                    delay = int(round(rng.normal(s._mu + e, s._sd + 0.25 * e)))
                    delay = max(delay, -7)
                    deliv = max(prom + delay, t + 5)
                    rate = min(0.9, s._reject + rej[s_id][min(deliv, N_DAYS - 1)])
                    rejected = int(rng.binomial(int(q), rate))
                    if deliv < N_DAYS:
                        receipts[deliv] += q - rejected
                    open_orders.append((deliv, q))
                    known = deliv <= AS_OF_IDX
                    pos.append({
                        "part_id": p.part_id, "supplier_id": s_id, "order_date": DAYS[t].date(),
                        "qty_ordered": q, "unit_price": round(unit_price[s_id] * price_drift[t], 2),
                        "quoted_lead_time": qlt[s_id], "promised_date": DAYS[min(prom, N_DAYS - 1)].date(),
                        "delivered_date": DAYS[deliv].date() if known else None,
                        "qty_accepted": q - rejected if known else None,
                        "qty_rejected": rejected if known else None,
                        "_true_delivery": DAYS[min(deliv, N_DAYS - 1)].date(),
                    })
        po = pd.DataFrame(pos).sort_values(["order_date", "part_id"]).reset_index(drop=True)
        po.insert(0, "po_id", [f"PO-{i + 1:06d}" for i in range(len(po))])
        po["status"] = np.where(po.delivered_date.isna(), "open", "delivered")
        return po, init_stock

    # -------------------------------------------------------------------------- financials
    def financials(self, sup, dis):
        rng = self.rng
        quarters = pd.period_range("2018Q1", pd.Period(AS_OF, "Q") - 1, freq="Q")
        rows = []
        fin = dis[dis.type.isin(["financial_distress", "sanctions"])]
        for _, s in sup.iterrows():
            base = rng.uniform(52, 88)
            x = 0.0
            for q in quarters:
                end = q.end_time.normalize()
                x = 0.7 * x + rng.normal(0, 2.5)
                v = base + x
                if pd.Timestamp("2020-04-01") <= end <= pd.Timestamp("2021-06-30"):
                    v -= 8
                ei = day_idx(end)
                for _, d in fin[fin.supplier_id == s.supplier_id].iterrows():
                    lead = 180 if d.type == "financial_distress" else 0
                    w = np.clip((ei - (d.onset_idx - lead)) / (d.ramp + lead), 0, 1)
                    if ei > d.onset_idx + d.ramp + d.plateau:
                        w *= 0.5
                    v -= w * (25 + 0.2 * d.delay_k)
                published = end + timedelta(days=45)
                if published.date() > AS_OF:
                    continue
                rows.append({"supplier_id": s.supplier_id, "period_end": end.date(),
                             "published_date": published.date(), "financial_score": round(float(np.clip(v, 1, 99)), 1)})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------------------ signals
    def signals(self, sup, dis):
        rng = self.rng
        name = sup.set_index("supplier_id").name
        docs = []

        def doc(sid, day, src, title, body, polarity, severity, truth=None):
            if day < 0 or day > AS_OF_IDX:
                return
            docs.append({
                "date": DAYS[day].date().isoformat(), "supplier_id": sid, "source": src,
                "title": title.format(name=name[sid]), "body": body.format(name=name[sid]),
                "polarity": polarity, "severity": severity, "_truth_disruption_id": truth,
            })

        for _, d in dis[dis.scope == "supplier"].iterrows():
            n = rng.choice([0, 1, 2, 3], p=[0.25, 0.35, 0.25, 0.15])
            tmpl = SIGNAL_TEMPLATES[d.type]
            for _ in range(n):
                src, title, body = tmpl[rng.integers(len(tmpl))]
                off = rng.integers(-60, 45) if d.type in ("financial_distress", "capacity_rampup") else rng.integers(0, 20)
                sev = 3 if d.severity == "high" else 2 if d.severity == "medium" else 1
                doc(d.supplier_id, int(d.onset_idx + off), src, title, body, -1, sev, d.disruption_id)
        years = AS_OF_IDX / 365.25
        for sid in sup.supplier_id:
            for _ in range(rng.poisson(0.25 * years)):
                src, title, body = NOISE_NEGATIVE[rng.integers(len(NOISE_NEGATIVE))]
                doc(sid, int(rng.integers(0, AS_OF_IDX)), src, title, body, -1, int(rng.integers(1, 3)))
            for _ in range(rng.poisson(0.6 * years)):
                src, title, body, pol = NEUTRAL_POSITIVE[rng.integers(len(NEUTRAL_POSITIVE))]
                doc(sid, int(rng.integers(0, AS_OF_IDX)), src, title, body, pol, 1)
        docs.sort(key=lambda x: x["date"])
        for i, x in enumerate(docs):
            x["signal_id"] = f"SIG-{i + 1:05d}"
        return docs


def prune(sup: pd.DataFrame, links: pd.DataFrame, sources: pd.DataFrame):
    """Keep only suppliers that are actually in the OEM's supply chain (tier-1 with parts, and
    sub-tier suppliers with at least one remaining customer)."""
    keep = set(sources.supplier_id)
    tier = sup.set_index("supplier_id").tier
    for t in (2, 3):
        live = links[links.customer_id.isin(keep)]
        keep |= set(live[live.upstream_id.map(tier) == t].upstream_id)
    links = links[links.customer_id.isin(keep) & links.upstream_id.isin(keep)].reset_index(drop=True)
    return sup[sup.supplier_id.isin(keep)].reset_index(drop=True), links


def generate(seed: int = SEED, out_raw=RAW_DIR, out_truth=TRUTH_DIR, verbose: bool = True) -> dict:
    g = Gen(seed)
    sup = g.suppliers()
    links = g.links(sup)
    parts, sources, bom = g.parts(sup)
    sup, links = prune(sup, links, sources)
    plan = monthly_plan()
    dis = g.disruptions(sup, links, sources)
    eff, rej, _ = g.effects(sup, links, dis)
    po, init = g.orders(sup, parts, sources, bom, plan, eff, rej)
    parts["initial_stock"] = parts.part_id.map(init)
    fin = g.financials(sup, dis)
    sig = g.signals(sup, dis)

    out_raw.mkdir(parents=True, exist_ok=True)
    out_truth.mkdir(parents=True, exist_ok=True)
    hidden = [c for c in sup.columns if c.startswith("_")]
    sup.drop(columns=hidden).to_csv(out_raw / "suppliers.csv", index=False)
    links.to_csv(out_raw / "supplier_links.csv", index=False)
    parts.to_csv(out_raw / "parts.csv", index=False)
    sources.to_csv(out_raw / "part_sources.csv", index=False)
    bom.to_csv(out_raw / "bom.csv", index=False)
    plan.to_csv(out_raw / "production_plan.csv", index=False)
    pd.DataFrame([{"program": k, "name": v[0], "segment": v[1]} for k, v in PROGRAMS.items()]).to_csv(
        out_raw / "programs.csv", index=False)
    po.drop(columns=["_true_delivery"]).to_csv(out_raw / "purchase_orders.csv", index=False)
    fin.to_csv(out_raw / "financials.csv", index=False)
    with open(out_raw / "signals.jsonl", "w", encoding="utf-8") as f:
        for d in sig:
            f.write(json.dumps({k: v for k, v in d.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")

    dis.to_csv(out_truth / "disruptions.csv", index=False)
    sup[["supplier_id"] + hidden].to_csv(out_truth / "supplier_params.csv", index=False)
    po[["po_id", "_true_delivery"]].to_csv(out_truth / "po_true_delivery.csv", index=False)
    pd.DataFrame([{"signal_id": d["signal_id"], "disruption_id": d["_truth_disruption_id"]} for d in sig]).to_csv(
        out_truth / "signal_links.csv", index=False)
    eff_week = pd.DataFrame({s: e for s, e in eff.items()}, index=DAYS).resample("W-SUN").mean()
    eff_week.round(2).to_csv(out_truth / "delay_effect_weekly.csv")

    stats = {"suppliers": len(sup), "links": len(links), "parts": len(parts), "purchase_orders": len(po),
             "open_orders": int(po.delivered_date.isna().sum()), "disruptions": len(dis), "signals": len(sig)}
    if verbose:
        print("generated:", stats)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    generate(ap.parse_args().seed)
