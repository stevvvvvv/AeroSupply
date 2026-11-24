"""Global configuration: paths, dates, thresholds and the static industrial reference data."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("AEROSUPPLY_DATA", BACKEND_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"            # "ERP exports" written by the generator
TRUTH_DIR = DATA_DIR / "truth"        # ground truth, used for evaluation only
ARTIFACTS_DIR = Path(os.getenv("AEROSUPPLY_ARTIFACTS", BACKEND_DIR / "artifacts"))

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BACKEND_DIR / 'aerosupply.db'}")
MONGO_URL = os.getenv("MONGO_URL")    # unset -> embedded mongomock persisted to DATA_DIR/docstore.json
MONGO_DB = os.getenv("MONGO_DB", "aerosupply")

SEED = 7

# Timeline -------------------------------------------------------------------------------------
SIM_START = date(2018, 1, 1)          # warm-up year (orders already in flight in 2019)
HISTORY_START = date(2019, 1, 1)
AS_OF = date(2026, 11, 1)             # "today" for the whole platform (reproducible)
SIM_END = date(2027, 12, 31)          # production plan / disruption profiles extend past AS_OF

# Temporal splits (no row of the future is ever seen by a model trained on the past)
TRAIN_END = date(2024, 6, 30)
TEST_START = date(2024, 10, 1)        # 3-month gap = label horizon, avoids overlap leakage

# Business rules ------------------------------------------------------------------------------
OTD_TOLERANCE_DAYS = 3                # delivered <= promised + 3 days counts as on time
RISK_HORIZON_DAYS = 90
SEVERE_OTD = 0.60                     # "severe delivery failure" = next-90-day OTD below 60 %
SEVERE_DELAY_DAYS = 30                # ... or mean delay above 30 days
STOCKOUT_HORIZONS = (30, 60, 90)
MC_RUNS = 300                         # Monte-Carlo scenarios for the stock-out forecast

# Static reference data -------------------------------------------------------------------------
PROGRAMS = {
    # code: (name, segment)
    "NB-200": ("Single-aisle NB-200", "Narrowbody"),
    "WB-350": ("Twin-aisle WB-350", "Widebody"),
    "RJ-72": ("Regional turboprop RJ-72", "Regional"),
    "BJ-6": ("Business jet BJ-6", "Business"),
}

# monthly production rate per year (aircraft / month). 2020 is split: Jan-Mar pre-COVID rate.
RATE_PLAN = {
    "NB-200": {2018: 45, 2019: 50, 2020: 30, 2021: 38, 2022: 44, 2023: 50, 2024: 52, 2025: 56, 2026: 60, 2027: 63},
    "WB-350": {2018: 9, 2019: 9, 2020: 5, 2021: 5, 2022: 6, 2023: 7, 2024: 7, 2025: 8, 2026: 9, 2027: 10},
    "RJ-72": {2018: 6, 2019: 6, 2020: 3, 2021: 3, 2022: 4, 2023: 4, 2024: 5, 2025: 5, 2026: 5, 2027: 6},
    "BJ-6": {2018: 3, 2019: 3, 2020: 2, 2021: 2, 2022: 3, 2023: 3, 2024: 4, 2025: 4, 2026: 4, 2027: 4},
}

# Commodities by tier: (label, base quoted lead time in days, typical unit cost EUR)
TIER1_COMMODITIES = {
    "aerostructures": ("Aerostructures", 180, 60_000),
    "machined_parts": ("Precision machined parts", 120, 2_500),
    "landing_gear": ("Landing gear", 240, 45_000),
    "hydraulics": ("Hydraulic systems", 150, 12_000),
    "avionics": ("Avionics", 150, 30_000),
    "electrical": ("Electrical harnesses", 90, 4_000),
    "interiors": ("Cabin interiors", 120, 8_000),
    "fasteners": ("Fasteners", 60, 4),
    "composites": ("Composite parts", 150, 9_000),
    "nacelle": ("Nacelle components", 200, 35_000),
}
TIER2_COMMODITIES = {
    "castings": ("Castings", 160),
    "forgings": ("Forgings", 200),
    "prepreg": ("Carbon prepreg", 120),
    "electronic_components": ("Electronic components", 140),
    "aluminium_semi": ("Aluminium plate & extrusions", 110),
    "rubber_seals": ("Elastomer seals", 80),
}
TIER3_COMMODITIES = {
    "titanium": ("Titanium sponge & ingot", 180),
    "nickel_superalloy": ("Nickel superalloys", 150),
    "aluminium_ingot": ("Aluminium ingot", 90),
    "carbon_fiber": ("Carbon fiber", 120),
    "semiconductors": ("Semiconductors", 160),
}

# Bill of materials of each commodity: which upstream commodities it consumes.
# Inputs are complements (Leontief: the scarcest input limits output); suppliers of the same
# input commodity are substitutes (their effects are averaged by share).
UPSTREAM = {
    "aerostructures": ["aluminium_semi", "forgings", "prepreg"],
    "machined_parts": ["aluminium_semi", "forgings", "castings"],
    "landing_gear": ["forgings", "castings", "rubber_seals"],
    "hydraulics": ["castings", "rubber_seals"],
    "avionics": ["electronic_components"],
    "electrical": ["electronic_components"],
    "interiors": ["prepreg", "aluminium_semi"],
    "fasteners": ["forgings", "aluminium_semi"],
    "composites": ["prepreg"],
    "nacelle": ["prepreg", "castings", "forgings"],
    "castings": ["nickel_superalloy", "titanium"],
    "forgings": ["titanium", "nickel_superalloy", "aluminium_ingot"],
    "prepreg": ["carbon_fiber"],
    "electronic_components": ["semiconductors"],
    "aluminium_semi": ["aluminium_ingot"],
    "rubber_seals": [],
}
UPSTREAM_PASS_THROUGH = 0.5   # share of an upstream delay passed on (sub-tier buffers absorb the rest)
UPSTREAM_LAG_DAYS = 30        # upstream problems hit customers ~1 month later

# Illustrative country risk index (0 = low, 1 = high). Approximate values inspired by public
# governance / logistics indicators; replace with World Bank WGI + LPI for production use.
COUNTRIES = {
    # code: (name, risk, lat, lon)
    "FR": ("France", 0.15, 46.6, 2.4),
    "DE": ("Germany", 0.12, 51.1, 10.4),
    "GB": ("United Kingdom", 0.17, 52.8, -1.6),
    "ES": ("Spain", 0.20, 40.3, -3.7),
    "IT": ("Italy", 0.25, 42.8, 12.5),
    "PL": ("Poland", 0.25, 52.1, 19.4),
    "CZ": ("Czechia", 0.20, 49.8, 15.5),
    "RO": ("Romania", 0.33, 45.9, 24.9),
    "US": ("United States", 0.18, 39.8, -98.6),
    "CA": ("Canada", 0.12, 56.1, -106.3),
    "MX": ("Mexico", 0.45, 23.6, -102.6),
    "MA": ("Morocco", 0.40, 31.8, -7.1),
    "TN": ("Tunisia", 0.50, 33.9, 9.5),
    "IN": ("India", 0.45, 21.1, 78.0),
    "CN": ("China", 0.55, 35.9, 104.2),
    "JP": ("Japan", 0.12, 36.2, 138.3),
    "TR": ("Turkey", 0.55, 39.0, 35.2),
    "NO": ("Norway", 0.08, 60.5, 8.5),
    "KZ": ("Kazakhstan", 0.60, 48.0, 66.9),
    "RU": ("Russia", 0.85, 61.5, 105.3),
}

# Illustrative list price per aircraft (EUR m) — only used to express deliveries at risk in value.
PROGRAM_PRICE_MEUR = {"NB-200": 55, "WB-350": 140, "RJ-72": 20, "BJ-6": 30}

# Disruption simulation assumptions
SUBTIER_BUFFER_DAYS = {1: 0, 2: 30, 3: 60}   # stock held downstream delays the impact of a sub-tier failure
SURGE_START_DAYS = 60                          # alternative qualified sources need time to ramp up
SURGE_CAPACITY = 0.5                           # ... and can absorb up to 50 % of their current share extra
