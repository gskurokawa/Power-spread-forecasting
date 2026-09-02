"""
Power-spread forecast API (FastAPI).

Serves the day-ahead spread forecasts that the pipeline writes to Postgres, plus
the SHAP driver ranking. Reads the same `predictions` table the Streamlit app and
`daily_update.py` use (columns: timestamp, spread, actual, predicted).

Run locally:
    export DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/db
    uvicorn api.main:app --reload --port 8000
    # then open http://localhost:8000/docs

With no DATABASE_URL set it falls back to predictions.csv, so it runs anywhere.
"""
from __future__ import annotations

import csv
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
except Exception:  # sqlalchemy optional when running purely off the CSV fallback
    create_engine = None  # type: ignore
    Engine = object       # type: ignore

# spread code <-> short label used by the dashboard
SPREADS = {"DE-PL": "DE-PL", "DE-FR": "DE-FR"}
CODE_BY_DB = {v: k for k, v in SPREADS.items()}

DATABASE_URL = os.environ.get("DATABASE_URL")
PRED_CSV = Path(os.environ.get("PREDICTIONS_CSV", "predictions.csv"))
SHAP_CSV = Path(os.environ.get("SHAP_RANKING_CSV", "shap_ranking_preauction.csv"))

_state: dict = {"engine": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if DATABASE_URL and create_engine is not None:
        _state["engine"] = create_engine(DATABASE_URL, pool_pre_ping=True)
    yield
    eng = _state.get("engine")
    if eng is not None:
        eng.dispose()


app = FastAPI(
    title="European Power Price Spread Forecast API",
    version="1.0.0",
    summary="Day-ahead DE-PL and DE-FR spread forecasts and their SHAP drivers.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"]
)


# ---- response models ---------------------------------------------------------
class ForecastPoint(BaseModel):
    timestamp: str
    spread: str
    predicted: Optional[float] = None
    actual: Optional[float] = None


class Driver(BaseModel):
    rank: int
    driver: str
    mean_abs_shap: Optional[float] = None


# ---- data access (DB, with CSV fallback) -------------------------------------
def _rows_from_db(db_spread: str, limit: int, order_desc: bool):
    eng: Engine = _state["engine"]
    order = "DESC" if order_desc else "ASC"
    sql = text(
        f"SELECT timestamp, spread, actual, predicted FROM predictions "
        f"WHERE spread = :s ORDER BY timestamp {order} LIMIT :n"
    )
    with eng.connect() as c:
        return [dict(r._mapping) for r in c.execute(sql, {"s": db_spread, "n": limit})]


def _rows_from_csv(db_spread: str, limit: int, order_desc: bool):
    if not PRED_CSV.exists():
        raise HTTPException(503, f"No database and no {PRED_CSV} fallback found.")
    rows = []
    with PRED_CSV.open() as f:
        for r in csv.DictReader(f):
            if r.get("spread") == db_spread:
                rows.append(r)
    rows.sort(key=lambda r: r["timestamp"], reverse=order_desc)
    out = []
    for r in rows[:limit]:
        out.append({
            "timestamp": r["timestamp"], "spread": r["spread"],
            "actual": _f(r.get("actual")), "predicted": _f(r.get("predicted")),
        })
    return out


def _f(v):
    try:
        return float(v) if v not in (None, "", "NA", "NaN") else None
    except (TypeError, ValueError):
        return None


def _fetch(code: str, limit: int, order_desc: bool):
    if code not in SPREADS:
        raise HTTPException(404, f"Unknown spread '{code}'. Try one of {list(SPREADS)}.")
    db_spread = SPREADS[code]
    rows = (_rows_from_db(db_spread, limit, order_desc)
            if _state.get("engine") is not None
            else _rows_from_csv(db_spread, limit, order_desc))
    return [
        ForecastPoint(
            timestamp=str(r["timestamp"]), spread=CODE_BY_DB.get(r["spread"], r["spread"]),
            predicted=_f(r.get("predicted")), actual=_f(r.get("actual")),
        )
        for r in rows
    ]


# ---- endpoints ---------------------------------------------------------------
@app.get("/", tags=["meta"])
def root():
    return {
        "service": "power-spread-forecast-api",
        "source": "postgres" if _state.get("engine") is not None else "csv-fallback",
        "spreads": list(SPREADS),
        "docs": "/docs",
    }


@app.get("/health", tags=["meta"])
def health():
    if _state.get("engine") is None:
        return {"status": "ok", "db": "not-configured (csv fallback)"}
    try:
        with _state["engine"].connect() as c:
            c.execute(text("SELECT 1"))
        return {"status": "ok", "db": "reachable"}
    except Exception as e:  # pragma: no cover
        raise HTTPException(503, f"database unreachable: {e}")


@app.get("/spreads", tags=["forecast"])
def list_spreads():
    return [{"code": k, "db_name": v} for k, v in SPREADS.items()]


@app.get("/forecast/{spread}", response_model=ForecastPoint, tags=["forecast"])
def latest_forecast(spread: str):
    """Most recent forecast for a spread (the next day's, once published)."""
    rows = _fetch(spread, limit=1, order_desc=True)
    if not rows:
        raise HTTPException(404, f"No forecasts stored for {spread} yet.")
    return rows[0]


@app.get("/forecast/{spread}/series", response_model=list[ForecastPoint], tags=["forecast"])
def forecast_series(
    spread: str,
    hours: int = Query(168, ge=1, le=8760, description="How many recent hours to return."),
):
    """Recent actual-vs-predicted series for charting."""
    rows = _fetch(spread, limit=hours, order_desc=True)
    return list(reversed(rows))


@app.get("/drivers", response_model=list[Driver], tags=["interpretation"])
def drivers(top: int = Query(10, ge=1, le=50)):
    """Top SHAP drivers of the DE-PL forecast (from the interpretation stage)."""
    if not SHAP_CSV.exists():
        raise HTTPException(503, f"{SHAP_CSV} not found.")
    out = []
    with SHAP_CSV.open() as f:
        reader = list(csv.reader(f))
    header = reader[0] if reader else []
    # tolerate either (driver, mean_abs_shap) or an indexed first column
    for i, row in enumerate(reader[1:top + 1], start=1):
        if not row:
            continue
        name = row[1] if len(row) > 2 else row[0]
        val = _f(row[-1])
        out.append(Driver(rank=i, driver=str(name), mean_abs_shap=val))
    return out
