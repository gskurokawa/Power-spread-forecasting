#%%
"""
Merge the per-series ENTSO-E CSVs into ONE aligned hourly modelling table.

Handles: mixed DST offsets in timestamps, outage EVENTS -> hourly "MW unavailable"
(real ENTSO-E format AND the synthetic format), and per-country holiday flags.
"""

import os
import glob
import ast
import pandas as pd

DATA_DIR = "entsoe_data"               # real data folder (synthetic_entsoe_data for the fake set)
OUT = "modelling_table.csv"
TZ = "Europe/Berlin"
ZONES = ["DE_LU", "FR", "PL"]
SHORT = {"DE_LU": "DE", "FR": "FR", "PL": "PL"}


def resolve_data_dir(name):
    for c in [name, os.path.join(name, name), os.path.join(os.getcwd(), name)]:
        if os.path.isdir(c) and glob.glob(os.path.join(c, "*.csv")):
            print(f"Using data folder: {os.path.abspath(c)}")
            return c
    print(f"ERROR: no CSVs found. Working dir: {os.getcwd()}")
    print(f"  Folders here: {[d for d in os.listdir('.') if os.path.isdir(d)]}")
    raise SystemExit("Set DATA_DIR to the folder that holds the CSVs.")


def _header_rows(path):
    """1 if the CSV has a normal single header, 2 if it has a MultiIndex header
    (ENTSO-E generation files: production type on row 1, Aggregated/Consumption on row 2)."""
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        second_first_cell = f.readline().split(",")[0].strip()
    try:
        ts = pd.Timestamp(second_first_cell)          # 2nd row starts with a timestamp?
        return 1 if pd.notna(ts) else 2               # NaT (empty cell) -> 2-row header
    except Exception:
        return 2                                       # non-timestamp string -> 2-row header


def _load(name, col=None):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        print(f"  [skip] {name} (not found)")
        return None
    nh = _header_rows(path)
    df = pd.read_csv(path, index_col=0, header=[0, 1] if nh == 2 else 0, low_memory=False)
    if nh == 2:
        # flatten MultiIndex columns -> keep the production type (level 0),
        # drop "Actual Consumption" columns (storage), keep "Actual Aggregated"
        keep = {}
        for c in df.columns:
            top, sub = (c[0], c[1]) if isinstance(c, tuple) else (c, "")
            if "Consumption" in str(sub):
                continue
            keep[top] = df[c]
        df = pd.DataFrame(keep)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(TZ)   # mixed DST offsets
    return df[col] if col else df


def _1d(obj):
    return None if obj is None else (obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj)


def _split_col(c):
    """Return (production_type, subtype) from a column that may be a real tuple,
    a stringified tuple  "('Wind Onshore', 'Actual Aggregated')", or a plain name."""
    if isinstance(c, tuple):
        return c[0], (c[1] if len(c) > 1 else "")
    s = str(c)
    if s.startswith("(") and s.endswith(")"):
        try:
            t = ast.literal_eval(s)
            if isinstance(t, tuple):
                return t[0], (t[1] if len(t) > 1 else "")
        except Exception:
            pass
    return s, ""


def _load_gen(name):
    """Load a generation-by-type file robustly (any header quirk). Returns a frame
    keyed by production type, numeric, with 'Actual Consumption' columns dropped."""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return None
    nh = _header_rows(path)
    df = pd.read_csv(path, index_col=0, header=[0, 1] if nh == 2 else 0, low_memory=False)
    keep = {}
    for c in df.columns:
        top, sub = _split_col(c)
        if "Consumption" in str(sub):
            continue
        col = pd.to_numeric(df[c], errors="coerce")
        keep[top] = keep[top] + col if top in keep else col   # sum any dup types
    out = pd.DataFrame(keep)
    out.index = pd.to_datetime(out.index, utc=True).tz_convert(TZ)
    return out


def load_outages_hourly(zone, index):
    """Expand generation-unit outage EVENTS into an hourly 'MW unavailable' series
    aligned to `index`. Works on real ENTSO-E files (nominal_power/avail_qty) and
    on the synthetic files (unavailable_MW)."""
    path = os.path.join(DATA_DIR, f"{zone}_outages_generation.csv")
    if not os.path.exists(path):
        return pd.Series(0.0, index=index)
    ev = pd.read_csv(path)
    if ev.empty:
        return pd.Series(0.0, index=index)

    for c in ("start", "end"):                       # tz-aware, mixed offsets
        ev[c] = pd.to_datetime(ev[c], utc=True, errors="coerce").dt.tz_convert(TZ)

    if {"mrid", "revision"} <= set(ev.columns):      # keep latest revision per outage
        ev = ev.sort_values("revision").drop_duplicates("mrid", keep="last")

    if "unavailable_MW" in ev.columns:               # synthetic format
        ev["_un"] = pd.to_numeric(ev["unavailable_MW"], errors="coerce")
    elif {"nominal_power", "avail_qty"} <= set(ev.columns):   # real ENTSO-E format
        ev["_un"] = (pd.to_numeric(ev["nominal_power"], errors="coerce")
                     - pd.to_numeric(ev["avail_qty"], errors="coerce")).clip(lower=0)
    else:
        print(f"  [note] {zone} outages: unrecognised columns, skipping")
        return pd.Series(0.0, index=index)

    ev = ev.dropna(subset=["start", "end", "_un"])
    hourly = pd.Series(0.0, index=index)
    for _, r in ev.iterrows():
        hourly.loc[(index >= r["start"]) & (index < r["end"])] += r["_un"]
    return hourly


# --- Polish price currency fix -------------------------------------------------
# ENTSO-E reports Polish day-ahead prices in PLN before 2019-11-20 and in EUR from
# 2019-11-20 onward (a reporting change, not a real price jump). Convert the PLN
# period to EUR using monthly-average ECB EUR/PLN reference rates.
PL_FX_SWITCH = pd.Timestamp("2019-11-20 00:00", tz="UTC")
EURPLN = {
    "2018-12": 4.2899, "2019-01": 4.2945, "2019-02": 4.3158, "2019-03": 4.2968,
    "2019-04": 4.2839, "2019-05": 4.2960, "2019-06": 4.2603, "2019-07": 4.2573,
    "2019-08": 4.3515, "2019-09": 4.3439, "2019-10": 4.3010, "2019-11": 4.2755,
}


def correct_pl_fx(price_pl):
    s = price_pl.copy()
    mask = s.index < PL_FX_SWITCH
    if not mask.any():
        return s
    rates = pd.Series(s.index.strftime("%Y-%m"), index=s.index).map(EURPLN)
    conv = mask & rates.notna()
    s.loc[conv] = s.loc[conv] / rates.loc[conv]
    print(f"  PL FX fix: converted {int(conv.sum())} pre-2019-11-20 hours PLN->EUR")
    return s


def build():
    cols = {}

    # prices + targets
    prices = {}
    for z in ZONES:
        s = _1d(_load(f"{z}_day_ahead_prices"))
        if s is not None:
            if z == "PL":
                s = correct_pl_fx(s)          # PLN -> EUR for the pre-2019-11-20 period
            prices[z] = s
            cols[f"price_{SHORT[z]}"] = s
    if "DE_LU" in prices and "FR" in prices:
        cols["spread_DE_FR"] = prices["DE_LU"] - prices["FR"]
    if "DE_LU" in prices and "PL" in prices:
        cols["spread_DE_PL"] = prices["DE_LU"] - prices["PL"]

    # per-zone features + actuals
    for z in ZONES:
        s = SHORT[z]
        cols[f"load_fc_{s}"] = _1d(_load(f"{z}_load_forecast_da"))
        cols[f"load_act_{s}"] = _1d(_load(f"{z}_load_actual"))
        cols[f"gen_fc_{s}"] = _1d(_load(f"{z}_generation_forecast_da"))
        ws = _load(f"{z}_wind_solar_forecast_da")
        if ws is not None:
            cols[f"wind_solar_fc_{s}"] = ws.sum(axis=1)
        gen = _load_gen(f"{z}_generation_by_type")
        if gen is not None:
            wind = [c for c in gen.columns if "Wind" in str(c)]     # onshore + offshore
            solar = [c for c in gen.columns if "Solar" in str(c)]
            if wind:  cols[f"wind_{s}"] = gen[wind].sum(axis=1)
            if solar: cols[f"solar_{s}"] = gen[solar].sum(axis=1)
            ren = list(dict.fromkeys(wind + solar))
            if ren:   cols[f"renewable_act_{s}"] = gen[ren].sum(axis=1)   # combined, symmetric

    for pair in ("DE_FR", "DE_PL"):
        cols[f"flow_{pair}"] = _1d(_load(f"interconnector_{pair}"))     # physical (realised) -> EDA only, NOT a model feature

    # day-ahead scheduled commercial exchanges on Germany's borders
    # (leakage-safe: known at gate closure) -> one column per neighbour
    sched = _load("DE_LU_scheduled_exchanges_da")
    if sched is not None:
        if isinstance(sched, pd.Series):
            sched = sched.to_frame()
        for nb in sched.columns:
            cols[f"sched_exch_DE_{nb}"] = pd.to_numeric(sched[nb], errors="coerce")

    # FR / PL day-ahead net positions (leakage-safe), if the files exist
    for z in ("FR", "PL"):
        cols[f"net_pos_{z}"] = _1d(_load(f"{z}_net_position_da"))

    cols = {k: v for k, v in cols.items() if v is not None}
    if not cols:
        raise SystemExit("No data columns loaded — check DATA_DIR.")
    df = pd.DataFrame(cols).sort_index()

    # collapse to a clean HOURLY grid. From 1 Oct 2025 EU day-ahead prices switched
    # to 15-min resolution, so prices arrive quarter-hourly while other series stay
    # hourly -> averaging each hour's quarters realigns everything onto one grid.
    df = df.resample("1h").mean()

    # trim to the DE-LU era (Germany only exists from Oct 2018; earlier rows have no spread)
    if "price_DE" in df.columns and df["price_DE"].first_valid_index() is not None:
        start = df["price_DE"].first_valid_index()
        before = len(df)
        df = df.loc[start:]
        print(f"Trimmed to DE-LU era: {start.date()} onward ({before - len(df)} earlier rows dropped)")

    # outages: expand events -> hourly, aligned to the final index
    for z in ZONES:
        df[f"outage_{SHORT[z]}"] = load_outages_hourly(z, df.index)

    # calendar + per-country holidays
    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    try:
        import holidays
        yrs = range(int(idx.year.min()), int(idx.year.max()) + 1)
        for z, code in [("DE", "DE"), ("FR", "FR"), ("PL", "PL")]:
            hol = set(holidays.country_holidays(code, years=yrs).keys())
            df[f"is_holiday_{z}"] = pd.Series(idx.date, index=idx).isin(hol).astype(int)
    except Exception as e:
        print(f"  [note] holidays not added ({e}); pip install holidays")

    return df


if __name__ == "__main__":
    DATA_DIR = resolve_data_dir(DATA_DIR)
    table = build()
    table.to_csv(OUT)
    print(f"\nModelling table: {table.shape[0]} rows x {table.shape[1]} cols -> {OUT}")
    print(f"Range: {table.index.min()} -> {table.index.max()}")
    print("Columns:", list(table.columns))
    print("\nOutage columns (should now be populated, not all-NaN):")
    print(table[[f"outage_{s}" for s in SHORT.values()]].describe().loc[["mean", "max"]])
# %%
