"""Read-side state of the API: pipeline outputs + the objects needed for live what-if analysis.

Reloaded at most every ``TTL`` seconds so results written by Airflow runs show up without restart.
"""
from __future__ import annotations

import json
import math
import threading
import time

import joblib
import numpy as np
import pandas as pd

from ..config import ARTIFACTS_DIR
from ..data.inventory import part_daily_demand
from ..storage import sql
from ..storage.docs import get_store

TTL = 300
_lock = threading.Lock()
_state = None


class State:
    def __init__(self):
        from ..graph.network import Network
        from ..ml.features import FeatureBuilder
        from ..pipeline.steps import base_paths, dataset
        get_store().reload()
        self.D = dataset()
        self.fb = FeatureBuilder(self.D)
        self.net = Network(self.D)
        self.demand = part_daily_demand(self.D["bom"], self.D["production_plan"])
        self.paths = base_paths()
        self.leadtime = joblib.load(ARTIFACTS_DIR / "leadtime.joblib")
        self.samples = joblib.load(ARTIFACTS_DIR / "shortage_samples.joblib")
        self.metrics = json.loads((ARTIFACTS_DIR / "metrics.json").read_text())
        self.scores = sql.read_result("supplier_scores", json_cols=["drivers", "programs"])
        self.parts = sql.read_result("part_forecasts")
        self.part_paths = sql.read_result("part_paths")
        self.weekly = sql.read_result("anomaly_weekly")
        self.episodes = sql.read_result("anomaly_episodes")
        self.commodity = sql.read_result("commodity_weekly")
        self.common_cause = sql.read_result("common_cause")
        self.lead_times = sql.read_result("lead_times")
        self.spofs = sql.read_result("hidden_spofs")
        self.mitigation = sql.read_result("mitigation_plan")
        self.loaded_at = time.time()


def get_state(force: bool = False) -> State:
    global _state
    with _lock:
        if force or _state is None or time.time() - _state.loaded_at > TTL:
            _state = State()
        return _state


def clean(v):
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) or math.isinf(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.date().isoformat() if v == v.normalize() else v.isoformat()
    if v is pd.NaT:
        return None
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return clean(df.to_dict(orient="records"))
