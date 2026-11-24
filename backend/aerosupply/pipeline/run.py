"""Run the whole pipeline sequentially (same steps as the Airflow DAG).

    python -m aerosupply.pipeline.run                 # daily run (reuses trained models)
    python -m aerosupply.pipeline.run --retrain       # + retrain and re-evaluate every model
    python -m aerosupply.pipeline.run --regenerate    # + regenerate the synthetic exports
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from .steps import STEPS, extract, log_step, train_models


def run(retrain: bool = False, regenerate: bool = False, only: list[str] | None = None) -> str:
    run_id = "cli__" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    for name, fn in STEPS:
        if only and name not in only:
            continue
        t = time.time()
        if fn is extract:
            res = fn(run_id, force=regenerate)
        elif fn is train_models:
            res = fn(run_id, force=retrain or regenerate)
        else:
            res = fn(run_id)
        dt = time.time() - t
        log_step(run_id, name, dt, res)
        print(f"[{name:<20}] {dt:6.1f}s  {res}")
    return run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    run(a.retrain, a.regenerate, a.only)
