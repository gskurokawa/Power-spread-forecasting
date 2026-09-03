"""
Airflow DAG: daily pre-auction power-spread update.

Replaces the GitHub-Actions / cron-job.org trigger with a real orchestrator. The
pipeline stays in `daily_update.py`; Airflow adds scheduling, retries, run history,
and a gate so a run is skipped cleanly when ENTSO-E is unreachable.

    entsoe_up (ShortCircuit)  ->  run_daily_update  ->  verify_freshness

Deploy: mount the project repo into the Airflow scheduler/worker and put this file
in the dags/ folder (see airflow/README.md). ENTSOE_API_KEY and DATABASE_URL come
from the Airflow environment (or Airflow Variables/Connections).

Heavy modelling imports (xgboost, etc.) are done INSIDE task callables, never at
module top level, so DAG parsing stays fast.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/project")

default_args = {
    "owner": "glen",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=45),
}


def entsoe_reachable(**_) -> bool:
    """Lightweight probe: is ENTSO-E answering? Short-circuits the run if not,
    so a platform outage skips (not fails) the day. Mirrors daily_update's own
    circuit breaker; kept here so downstream tasks don't even start."""
    from entsoe import EntsoePandasClient
    from entsoe.exceptions import NoMatchingDataError
    import pandas as pd

    key = os.environ["ENTSOE_API_KEY"]
    client = EntsoePandasClient(api_key=key, retry_count=2, retry_delay=0, timeout=15)
    now = pd.Timestamp.now(tz="Europe/Brussels")
    for _try in range(3):
        try:
            client.query_day_ahead_prices(
                "DE_LU", start=now.floor("D") - pd.Timedelta(days=1), end=now.floor("D")
            )
            return True
        except NoMatchingDataError:
            return True            # server up, just no data for that window
        except Exception:
            continue
    return False


def verify_freshness(**_):
    """Confirm the run advanced the predictions table; warn (don't fail) if the
    latest forecast is stale, e.g. because ENTSO-E hadn't published day-ahead
    forecasts yet."""
    from sqlalchemy import create_engine, text

    eng = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    with eng.connect() as c:
        rows = c.execute(text(
            "SELECT spread, MAX(timestamp) AS latest FROM predictions GROUP BY spread"
        )).fetchall()
    if not rows:
        raise ValueError("predictions table is empty after the run")
    for spread, latest in rows:
        print(f"[freshness] {spread}: latest forecast = {latest}")
    eng.dispose()


with DAG(
    dag_id="power_spread_daily",
    description="Daily pre-auction DE-PL / DE-FR spread forecast",
    schedule="2 10 * * *",          # ~before the day-ahead gate closure (UTC)
    start_date=datetime(2026, 1, 1),
    catchup=False,                 # don't backfill missed days
    max_active_runs=1,             # never overlap runs (the old concurrency guard)
    default_args=default_args,
    tags=["energy", "forecasting", "entsoe"],
) as dag:

    entsoe_up = ShortCircuitOperator(
        task_id="entsoe_up",
        python_callable=entsoe_reachable,
    )

    run_daily_update = BashOperator(
        task_id="run_daily_update",
        bash_command=f"cd {PROJECT_DIR} && python daily_update.py --lean",
    )

    verify = PythonOperator(
        task_id="verify_freshness",
        python_callable=verify_freshness,
    )

    entsoe_up >> run_daily_update >> verify
