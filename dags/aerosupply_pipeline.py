"""Airflow DAG — daily supply-chain risk refresh.

The task list is generated from ``aerosupply.pipeline.steps.STEPS`` so the DAG and the CLI
(``python -m aerosupply.pipeline.run``) can never diverge. Each task is an isolated process that
exchanges data only through PostgreSQL, MongoDB and the model artifacts volume.

Models are retrained on Mondays (or when ``retrain`` is set in the run config); other days reuse
the latest artifacts.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow.sdk import dag, get_current_context, task

from aerosupply.pipeline.steps import STEPS, log_step

DEFAULT_ARGS = {"owner": "supply-chain-analytics", "retries": 2, "retry_delay": timedelta(minutes=5)}


def make_task(name, fn):
    @task(task_id=name, execution_timeout=timedelta(minutes=30))
    def _run():
        import time

        ctx = get_current_context()
        run_id = f"airflow__{ctx['run_id']}"
        conf = ctx["dag_run"].conf or {}
        kwargs = {}
        if name == "train_models":
            kwargs["force"] = bool(conf.get("retrain")) or ctx["logical_date"].weekday() == 0
        if name == "extract":
            kwargs["force"] = bool(conf.get("regenerate"))
        t = time.time()
        res = fn(run_id, **kwargs)
        log_step(run_id, name, time.time() - t, res)
        return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in res.items()}

    return _run()


@dag(
    dag_id="aerosupply_daily_risk",
    description="Ingest ERP/SRM exports and news signals, score supplier risk, forecast stock-outs",
    schedule="0 5 * * *",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["aerosupply", "supply-chain", "risk"],
)
def aerosupply_daily_risk():
    previous = None
    for name, fn in STEPS:
        current = make_task(name, fn)
        if previous is not None:
            previous >> current
        previous = current


aerosupply_daily_risk()
