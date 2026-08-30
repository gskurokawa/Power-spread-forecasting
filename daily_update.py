"""
daily_update.py — Phase 2 (option a): "latest-available-day refresh".

WHAT IT DOES, ONCE PER DAY
--------------------------
1. Look at the cloud database (Neon) and find the newest hour we already have.
2. Re-pull a SHORT recent window from ENTSO-E (a few days back, through
   tomorrow) — short because ENTSO-E keeps revising the last few days
   (load actuals, outages, generation), and because tomorrow's day-ahead
   auction result appears around 12:45 CET.
3. Rebuild those hours into the exact modelling-table schema, reusing the
   SAME transform logic as the historical build (build_modelling_table.build).
4. UPSERT that window into Neon: replace the overlapping recent rows (to pick
   up ENTSO-E's revisions) and append the genuinely new ones. Older history
   is left untouched.
5. Retrain the tuned XGBoost on the trailing two years and write fresh
   predictions for the new rows into the `predictions` table, so the
   dashboard's "latest available day" is always current.

WHY "option a": we predict whatever the freshest published day-ahead data is.
Because the day-ahead auction for tomorrow clears today, tomorrow's spread is
already known once the auction publishes — so every new row has both an
`actual` and a `predicted`. Option (b) — a genuine pre-auction forward
forecast — is a later refinement; the fetch/transform/upsert plumbing here is
exactly what (b) would reuse.

RUN IT LOCALLY FIRST (never straight into the scheduled job):

    python daily_update.py            # normal run
    python daily_update.py --dry-run  # fetch + transform + predict, but DON'T write to Neon
    python daily_update.py --buffer-days 14   # widen the revision refetch window

SECRETS — both come from the environment (a local .env here; GitHub Actions
secrets in Phase 3). Nothing is hard-coded:

    ENTSOE_API_KEY=your-entsoe-token
    DATABASE_URL=postgresql://USER:PASS@HOST/DB?sslmode=require

REQUIREMENTS
    python -m pip install entsoe-py pandas sqlalchemy psycopg2-binary holidays python-dotenv xgboost
"""

import os
import sys
import argparse
import tempfile
import datetime as dt

import pandas as pd
from sqlalchemy import create_engine, text

# reuse the well-tested transform + the model/feature definitions -----------
import build_modelling_table as bmt
from spread_forecasting import (
    num_features, add_lags, xgb_estimator, load_data,
    BIN, CAT, BEST_JSON, DIFF_MAP,
)
import json

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TZ = "Europe/Berlin"
ZONES = {"DE_LU": "Germany", "FR": "France", "PL": "Poland"}
DE_NEIGHBOURS = ["FR", "NL", "BE", "AT", "CH", "PL", "CZ",
                 "DK_1", "DK_2", "SE_4", "NO_2"]

# the two forecasting targets and the short labels the dashboard uses
SPREADS = ["spread_DE_PL", "spread_DE_FR"]
SHORT_LABEL = {"spread_DE_PL": "DE-PL", "spread_DE_FR": "DE-FR"}

DEFAULT_BUFFER_DAYS = 10   # how far back to refetch, to absorb ENTSO-E revisions
FORWARD_DAYS = 2           # fetch through tomorrow (+ a margin) for the new day-ahead


# ---------------------------------------------------------------------------
# 0. secrets / env
# ---------------------------------------------------------------------------
def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.environ.get("ENTSOE_API_KEY")
    db_url = os.environ.get("DATABASE_URL")
    missing = [n for n, v in [("ENTSOE_API_KEY", api_key), ("DATABASE_URL", db_url)] if not v]
    if missing:
        raise SystemExit(f"Missing environment variable(s): {', '.join(missing)}. "
                         "Put them in a local .env or set them in the shell.")
    return api_key, db_url


# ---------------------------------------------------------------------------
# 1. where does the cloud table currently end?
# ---------------------------------------------------------------------------
def latest_timestamp(engine):
    """Newest hour already in Neon's modelling_table, as a tz-aware UTC Timestamp,
    or None if the table is empty / absent."""
    try:
        with engine.connect() as c:
            row = c.execute(text("SELECT MAX(timestamp) FROM modelling_table")).scalar()
    except Exception as e:
        print(f"  [note] could not read modelling_table max(timestamp): {e}")
        return None
    if row is None:
        return None
    return pd.Timestamp(row).tz_localize("UTC") if pd.Timestamp(row).tzinfo is None \
        else pd.Timestamp(row).tz_convert("UTC")


# ---------------------------------------------------------------------------
# 2. fetch a SHORT recent window from ENTSO-E
#    (same query calls as entsoe_data_collect.py, parameterised by start/end
#     and written to a throwaway folder that build() then reads)
# ---------------------------------------------------------------------------
def fetch_window(api_key, start, end, out_dir, lean=False):
    """Pull the series build_modelling_table.build() consumes, for the window
    [start, end]. Saves CSVs into out_dir with the filenames build() expects.
    One failed pull is logged but does not stop the run.

    lean=True skips the pulls that are NOT model features — load/generation
    actuals and physical cross-border flows (all realised quantities, excluded
    from the model as leakage) — which cuts ~10 of ~38 calls for a faster run.
    The model inputs (prices, day-ahead forecasts, scheduled exchanges, net
    positions, outages) are always fetched."""
    from entsoe import EntsoePandasClient
    # retry_count/retry_delay: ENTSO-E's server throws intermittent 503s, so
    # every pull is retried a few times with a pause before we give up on it.
    client = EntsoePandasClient(api_key=api_key, retry_count=6, retry_delay=10, timeout=60)
    os.makedirs(out_dir, exist_ok=True)

    import time

    def safe(label, fn, tries=4, pause=10, spacing=1.5):
        """Run a pull; on top of the client's own retries, retry the whole call a
        few more times on transient server errors (503 / timeouts) before giving up.
        `spacing` puts a small gap BEFORE each call so we never burst ENTSO-E's API
        (bursting is what triggers 503s even when the backend is healthy)."""
        time.sleep(spacing)
        last = None
        for attempt in range(1, tries + 1):
            try:
                r = fn()
                print(f"    [ok]   {label}: {0 if r is None else len(r)} rows")
                return r
            except Exception as e:
                last = e
                msg = str(e)
                transient = ("503" in msg or "502" in msg or "504" in msg
                             or "timed out" in msg.lower() or "connection" in msg.lower())
                if transient and attempt < tries:
                    print(f"    [retry {attempt}/{tries - 1}] {label}: {type(e).__name__} — waiting {pause}s")
                    time.sleep(pause)
                    continue
                break
        print(f"    [FAIL] {label}: {type(last).__name__}: {last}")
        return None

    def to_hourly(o):
        return None if o is None else o.resample("1h").mean()

    def save(o, name):
        if o is None or len(o) == 0:
            return
        o.to_csv(os.path.join(out_dir, f"{name}.csv"))

    # -- per zone -----------------------------------------------------------
    for z in ZONES:
        print(f"  === {z} ({ZONES[z]}) ===")
        save(safe(f"{z} day-ahead prices",
                  lambda z=z: client.query_day_ahead_prices(z, start=start, end=end)),
             f"{z}_day_ahead_prices")
        if not lean:   # load/generation ACTUALS are not model features (leakage) -> skip in lean mode
            save(to_hourly(safe(f"{z} load (actual)",
                                lambda z=z: client.query_load(z, start=start, end=end))),
                 f"{z}_load_actual")
            save(to_hourly(safe(f"{z} generation by type",
                                lambda z=z: client.query_generation(z, start=start, end=end, psr_type=None))),
                 f"{z}_generation_by_type")
        save(to_hourly(safe(f"{z} DA load forecast",
                            lambda z=z: client.query_load_forecast(z, start=start, end=end))),
             f"{z}_load_forecast_da")
        save(to_hourly(safe(f"{z} DA wind & solar forecast",
                            lambda z=z: client.query_wind_and_solar_forecast(z, start=start, end=end, psr_type=None))),
             f"{z}_wind_solar_forecast_da")
        save(to_hourly(safe(f"{z} DA generation forecast",
                            lambda z=z: client.query_generation_forecast(z, start=start, end=end))),
             f"{z}_generation_forecast_da")
        # A80 outages is ENTSO-E's flakiest, heaviest endpoint (paginated) -> extra
        # patience. If it still fails, build() falls back to zero and the next
        # run's 10-day overlap re-pulls it, so a transient miss self-heals.
        # A80 outages is ENTSO-E's flakiest, heaviest endpoint (paginated). A few
        # tries is enough: if it still fails, build() falls back to zero and the
        # next run's 10-day overlap re-pulls it, so a transient miss self-heals.
        save(safe(f"{z} generation outages",
                  lambda z=z: client.query_unavailability_of_generation_units(z, start=start, end=end),
                  tries=3, pause=12, spacing=2),
             f"{z}_outages_generation")

    # -- cross-border physical flows: realised -> NOT model features (leakage).
    # Kept for a complete stored table; skipped in lean mode for speed.
    if not lean:
        print("  === cross-border flows (non-model) ===")
        for nb in ("FR", "PL"):
            fwd = safe(f"flow DE_LU->{nb}",
                       lambda nb=nb: client.query_crossborder_flows("DE_LU", nb, start=start, end=end))
            rev = safe(f"flow {nb}->DE_LU",
                       lambda nb=nb: client.query_crossborder_flows(nb, "DE_LU", start=start, end=end))
            if fwd is not None and rev is not None:
                net = (to_hourly(fwd) - to_hourly(rev)).rename(f"net_flow_DE_{nb}")
                save(net, f"interconnector_DE_{nb}")

    # -- DA scheduled commercial exchanges on Germany's borders -------------
    sched = {}
    for nb in DE_NEIGHBOURS:
        s = safe(f"sched DA DE_LU<->{nb}",
                 lambda nb=nb: client.query_scheduled_exchanges("DE_LU", nb, start=start, end=end, dayahead=True))
        if s is not None:
            sched[nb] = to_hourly(s)
    if sched:
        save(pd.DataFrame(sched), "DE_LU_scheduled_exchanges_da")

    # -- FR / PL day-ahead net positions -----------------------------------
    for z in ("FR", "PL"):
        save(to_hourly(safe(f"{z} net position (DA)",
                            lambda z=z: client.query_net_position(z, start=start, end=end, dayahead=True))),
             f"{z}_net_position_da")


# ---------------------------------------------------------------------------
# 3. transform the fetched window into the modelling-table schema
#    (reuses build_modelling_table.build() by pointing its DATA_DIR at our tmp)
# ---------------------------------------------------------------------------
def transform_window(data_dir):
    bmt.DATA_DIR = data_dir            # build() reads this module global
    table = bmt.build()               # exact same logic as the historical table
    table.index = pd.to_datetime(table.index, utc=True)
    table.index.name = "timestamp"
    return table.sort_index()


# ---------------------------------------------------------------------------
# 4. upsert the window into Neon (replace overlap, append new)
# ---------------------------------------------------------------------------
def align_to_table(engine, table_name, df):
    """Return df reindexed to the live table's column order, so append matches."""
    try:
        live_cols = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 0", engine).columns.tolist()
    except Exception:
        return df                      # table doesn't exist yet -> create as-is
    live_feature_cols = [c for c in live_cols if c != "timestamp"]
    for c in live_feature_cols:
        if c not in df.columns:
            df[c] = pd.NA              # keep schema stable if a pull was empty today
    return df[[c for c in live_feature_cols if c in df.columns]]


def merge_with_existing(engine, window, window_start):
    """Fill EMPTY cells in the fresh window from the rows already in Neon for the
    same range. A fresh (non-null) value always wins — so ENTSO-E revisions are
    picked up — but where a pull came back empty this run, the existing value is
    kept. This makes a failed/503'd pull unable to overwrite good data with nulls."""
    try:
        existing = pd.read_sql(
            text("SELECT * FROM modelling_table WHERE timestamp >= :ws"),
            engine, params={"ws": window_start.to_pydatetime()}, index_col="timestamp")
    except Exception as e:
        print(f"  [note] no existing overlap read ({e}); writing fresh window as-is")
        return window
    if existing.empty:
        return window
    existing.index = pd.to_datetime(existing.index, utc=True)
    existing = align_to_table(engine, "modelling_table", existing)
    merged = window.combine_first(existing)          # window's non-null values win
    filled = int(window.isna().sum().sum() - merged.loc[window.index].isna().sum().sum())
    if filled > 0:
        print(f"  [merge] preserved {filled} existing cell(s) where this run's pull was empty")
    return merged.reindex(columns=window.columns)


def upsert_window(engine, window, window_start, dry_run):
    """Delete modelling_table rows >= window_start, then append the (merged) window.
    Everything before window_start is left exactly as it was."""
    window = align_to_table(engine, "modelling_table", window.copy())
    if dry_run:
        print(f"  [dry-run] would replace modelling_table rows >= {window_start} "
              f"with {len(window)} fresh rows")
        return
    window = merge_with_existing(engine, window, window_start)   # never null-out good data
    with engine.begin() as conn:      # single transaction: delete + append atomic
        conn.execute(text("DELETE FROM modelling_table WHERE timestamp >= :ws"),
                     {"ws": window_start.to_pydatetime()})
        window.to_sql("modelling_table", conn, if_exists="append",
                      index=True, index_label="timestamp", chunksize=2000, method="multi")
    print(f"  modelling_table: replaced rows >= {window_start.date()} "
          f"({len(window)} rows written)")


# ---------------------------------------------------------------------------
# 5. retrain on trailing 2y and predict the new rows
# ---------------------------------------------------------------------------
def predict_new_rows(full_df, target, best, pred_from):
    """Train the tuned XGBoost on the two years ending at `pred_from` (early-stop
    on the last 8 weeks — identical to the walk-forward backtest) and predict
    every row from `pred_from` onward. Returns a Series indexed by timestamp."""
    d, lags = add_lags(full_df, target)
    num = num_features(full_df) + lags
    feats = num + BIN + CAT
    for c in CAT:
        d[c] = d[c].astype("category")

    ws = pred_from - pd.DateOffset(years=2)
    win = d[(d.index >= ws) & (d.index < pred_from)].dropna(subset=feats + [target])
    if len(win) < 2000:
        print(f"    [skip] {target}: only {len(win)} training rows before {pred_from.date()}")
        return pd.Series(dtype=float)

    cut = pred_from - pd.Timedelta(weeks=8)
    fit, esv = win[win.index < cut], win[win.index >= cut]
    if len(esv) < 200:
        fit, esv = win, win

    est = xgb_estimator(best)
    est.fit(fit[feats], fit[target], eval_set=[(esv[feats], esv[target])], verbose=False)

    ch = d[d.index >= pred_from].dropna(subset=feats)     # target may be NaN; features must be present
    if ch.empty:
        return pd.Series(dtype=float)
    return pd.Series(est.predict(ch[feats]), index=ch.index, name="predicted")


def build_prediction_rows(full_df, best, pred_from):
    """Tidy predictions frame (spread, timestamp, actual, predicted) for the new
    window, one block per spread, using the winning model (XGBoost direct)."""
    blocks = []
    for target in SPREADS:
        pred = predict_new_rows(full_df, target, best, pred_from)
        if pred.empty:
            continue
        actual = full_df[target].reindex(pred.index)
        blk = pd.DataFrame({"spread": SHORT_LABEL[target],
                            "actual": actual.values,
                            "predicted": pred.values}, index=pred.index)
        blk.index.name = "timestamp"
        blocks.append(blk)
        n_act = int(actual.notna().sum())
        print(f"    {SHORT_LABEL[target]}: {len(blk)} new predictions "
              f"({n_act} with a published actual)")
    return pd.concat(blocks) if blocks else pd.DataFrame()


def upsert_predictions(engine, preds, pred_from, dry_run):
    if preds.empty:
        print("  no new predictions to write")
        return
    if dry_run:
        print(f"  [dry-run] would replace predictions rows >= {pred_from} "
              f"with {len(preds)} rows")
        return
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM predictions WHERE timestamp >= :pf"),
                     {"pf": pred_from.to_pydatetime()})
        preds.reset_index().to_sql("predictions", conn, if_exists="append",
                                   index=False, chunksize=2000, method="multi")
    print(f"  predictions: replaced rows >= {pred_from.date()} ({len(preds)} rows written)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Daily latest-available-day refresh")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch, transform and predict but do NOT write to Neon")
    ap.add_argument("--buffer-days", type=int, default=DEFAULT_BUFFER_DAYS,
                    help="how many days back to refetch (absorbs ENTSO-E revisions)")
    ap.add_argument("--keep-data", action="store_true",
                    help="keep the temporary fetched CSVs instead of deleting them")
    ap.add_argument("--lean", action="store_true",
                    help="skip non-model pulls (load/generation actuals, physical flows) "
                         "for a faster run; the stored table leaves those columns null")
    args = ap.parse_args()

    api_key, db_url = load_env()
    engine = create_engine(db_url, pool_pre_ping=True)

    # -- window bounds ------------------------------------------------------
    now = pd.Timestamp.now(tz=TZ)
    latest = latest_timestamp(engine)
    if latest is not None:
        window_start = (latest.tz_convert(TZ) - pd.Timedelta(days=args.buffer_days)).normalize()
        print(f"Neon modelling_table currently ends at {latest.tz_convert(TZ)}")
    else:
        # empty DB: this script is for incremental top-ups, not the first bulk load
        window_start = (now - pd.Timedelta(days=args.buffer_days)).normalize()
        print("modelling_table is empty — run the one-time load_to_postgres.py first "
              "for the full history; this run will only add a recent window.")
    window_end = (now + pd.Timedelta(days=FORWARD_DAYS)).normalize()
    print(f"Fetch window: {window_start.date()} -> {window_end.date()}  (buffer {args.buffer_days}d)")

    # -- fetch + transform --------------------------------------------------
    import time as _time
    t0 = _time.time()
    tmp = tempfile.mkdtemp(prefix="entsoe_daily_")
    try:
        print(f"\n[1/4] fetching recent ENTSO-E window ...{'  (lean mode)' if args.lean else ''}")
        fetch_window(api_key, window_start, window_end, tmp, lean=args.lean)
        print(f"  fetch took {(_time.time() - t0) / 60:.1f} min")

        print("\n[2/4] transforming into the modelling-table schema ...")
        window = transform_window(tmp)
        if window.empty:
            raise SystemExit("Transform produced no rows — aborting (nothing written).")
        # only keep rows at/after window_start (build() may include a trimmed tail)
        window = window[window.index >= window_start.tz_convert("UTC")]
        print(f"  fresh window: {len(window)} rows, "
              f"{window.index.min()} -> {window.index.max()}")

        # guard: the holiday flags are real model features. If the `holidays`
        # package isn't installed, build() omits them silently and every new row
        # ends up incomplete -> no predictions. Fail loudly BEFORE writing, so the
        # unattended job can't quietly degrade.
        need = ["is_holiday_DE", "is_holiday_FR", "is_holiday_PL"]
        bad = [c for c in need if c not in window.columns or window[c].isna().all()]
        if bad:
            raise SystemExit(
                f"Holiday feature(s) missing/empty: {', '.join(bad)}. "
                "Install the 'holidays' package (python -m pip install holidays) and re-run — "
                "without it the fresh rows are incomplete and no predictions can be made.")

        print("\n[3/4] upserting modelling_table into Neon ...")
        upsert_window(engine, window, window_start.tz_convert("UTC"), args.dry_run)

        # -- read the FULL (now-updated) table back for lag-aware prediction
        print("\n[4/4] retraining on trailing 2y and predicting new rows ...")
        if args.dry_run:
            # in a dry run Neon is unchanged, so stitch locally: DB history + fresh window
            try:
                hist = pd.read_sql("SELECT * FROM modelling_table", engine,
                                   index_col="timestamp")
                hist.index = pd.to_datetime(hist.index, utc=True)
                full = pd.concat([hist[hist.index < window.index.min()], window]).sort_index()
            except Exception:
                full = window
        else:
            full = pd.read_sql("SELECT * FROM modelling_table", engine, index_col="timestamp")
            full.index = pd.to_datetime(full.index, utc=True)
            full = full.sort_index()

        best = json.load(open(BEST_JSON))
        pred_from = window.index.min()          # re-predict everything in the refreshed window
        preds = build_prediction_rows(full, best, pred_from)
        upsert_predictions(engine, preds, pred_from, args.dry_run)

    finally:
        if not args.keep_data:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nDone in {(_time.time() - t0) / 60:.1f} min."
          + ("  (dry run — nothing written to Neon)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
