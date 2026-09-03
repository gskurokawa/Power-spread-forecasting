"""
Streamlit dashboard for the spread-forecasting project.
Shows actual vs predicted spreads and the model results interactively.
Reads predictions.csv, the SHAP images, and the data file that
spread_forecasting.py produces. No heavy computation runs here.

Run locally:   python -m streamlit run app.py
Deploy free:   push to GitHub, then share.streamlit.io -> "New app" -> point at app.py
"""
import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Power Spread Forecasting", page_icon="⚡", layout="wide")
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------- data source: Neon cloud Postgres, CSV fallback
# When a [connections.neon] secret is set (Streamlit Secrets), the app reads live
# data from Neon. Otherwise it falls back to the committed CSVs, so it always works.
def load_predictions():
    try:
        conn = st.connection("neon", type="sql")
        df = conn.query("SELECT timestamp, spread, actual, predicted FROM predictions", ttl="1h")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return _norm_spread(df), "Neon · cloud Postgres"
    except Exception:
        if os.path.exists("predictions.csv"):
            return _norm_spread(pd.read_csv("predictions.csv", parse_dates=["timestamp"])), "local CSV (fallback)"
        return None, None

# accept either the short dashboard labels ("DE-PL") or the raw column names
# ("spread_DE_PL") in the predictions table, so old and new data both chart.
def _norm_spread(df):
    df["spread"] = df["spread"].replace({"spread_DE_PL": "DE-PL", "spread_DE_FR": "DE-FR"})
    return df

def modelling_table_bytes():
    try:
        conn = st.connection("neon", type="sql")
        data = conn.query("SELECT * FROM modelling_table", ttl="1h").to_csv(index=False).encode()
        return data, "Neon · cloud Postgres"
    except Exception:
        if os.path.exists("modelling_table.csv"):
            with open("modelling_table.csv", "rb") as f:
                return f.read(), "local file"
        return None, None

# actual day-ahead price LEVELS for DE, FR, PL (from modelling_table), for the
# price-levels overview chart. Neon first, committed CSV as a fallback.
def load_prices():
    cols = 'timestamp, "price_DE", "price_FR", "price_PL"'   # mixed-case cols need quoting
    try:
        conn = st.connection("neon", type="sql")
        # only the range the charts use (predictions start in 2025) — avoids
        # pulling the full 2018-onward history on every load
        df = conn.query(f"SELECT {cols} FROM modelling_table WHERE timestamp >= '2024-01-01'", ttl="1h")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp").sort_index(), "Neon · cloud Postgres"
    except Exception:
        if os.path.exists("modelling_table.csv"):
            df = pd.read_csv("modelling_table.csv", index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
            return df[["price_DE", "price_FR", "price_PL"]].sort_index(), "local CSV (fallback)"
        return None, None

# ----------------------------------------------------------------- results (operational pre-auction walk-forward, 2025+)
RESULTS = pd.DataFrame(
    [("DE-PL", "LASSO", "direct", 16.39, 25.82),
     ("DE-PL", "LASSO", "differenced", 17.10, 26.03),
     ("DE-PL", "XGBoost", "direct", 15.39, 26.76),
     ("DE-PL", "XGBoost", "differenced", 16.26, 25.72),
     ("DE-FR", "LASSO", "direct", 19.42, 26.83),
     ("DE-FR", "LASSO", "differenced", 19.30, 27.16),
     ("DE-FR", "XGBoost", "direct", 19.28, 27.20),
     ("DE-FR", "XGBoost", "differenced", 19.98, 27.92)],
    columns=["spread", "model", "form", "MAE", "RMSE"])

PERSISTENCE = {"DE-PL": 22.03, "DE-FR": 24.35}
VERDICT = {
    "DE-PL": ("**Congestion-prone border — modelling choices matter.** The tuned XGBoost "
              "on the spread directly is the best operational model (15.39 €/MWh). It beats "
              "LASSO by 1.00 €/MWh and differencing by 0.87 €/MWh, both significant "
              "(Diebold–Mariano, robust to a two-week HAC bandwidth). An oracle that could also "
              "see the auction's own outputs would reach 14.40, so forecasting ahead of the "
              "auction costs about €1/MWh."),
    "DE-FR": ("**Well-coupled border — nothing wins.** All four approaches land near 19 €/MWh, "
              "and no difference is statistically significant. Differencing two clean price "
              "forecasts loses nothing here."),
}

# ----------------------------------------------------------------- header
st.title("⚡ Forecasts of Selected European Power Price Spreads")
st.markdown(
    "Day-ahead Germany–Poland (DE-PL) and Germany–France (DE-FR) price spreads, forecast before each day's auction with walk-forward machine-learning models."
)

# ----------------------------------------------------------------- charts
# each spread is paired with its two actual price levels (Germany + the neighbour)
PAIR = {"DE-PL": ("price_DE", "price_PL", "DE", "PL"),
        "DE-FR": ("price_DE", "price_FR", "DE", "FR")}

def price_pair(prices, sp, since, until=None):
    """The two actual day-ahead price levels for a spread (e.g. DE and PL), from
    `since` to `until` (hourly). Capping at `until` keeps the price chart's right
    edge aligned with where the spread's actual ends, rather than running a day
    ahead of it whenever prices publish before the forecast inputs do."""
    a, b, la, lb = PAIR[sp]
    d = prices[prices.index >= since]
    if until is not None and pd.notna(until):
        d = d[d.index <= until]
    return d[[a, b]].rename(columns={a: la, b: lb})

# ----------------------------------------------------------------- spread: actual vs forecast
WINDOW_DAYS = 7            # recent history shown in every chart, at hourly resolution

def spread_series(pred_df, sp):
    d = pred_df[pred_df["spread"] == sp].set_index("timestamp").sort_index()
    last_day = d.index.normalize().max()
    is_last = d.index.normalize() == last_day
    latest_mean = float(d.loc[is_last, "predicted"].mean())
    latest_has_actual = bool(d.loc[is_last, "actual"].notna().any())
    actual_end = d.index[d["actual"].notna()].max()   # last hour with a realised actual
    cutoff = d.index.max() - pd.Timedelta(days=WINDOW_DAYS)
    d = d[d.index >= cutoff]                       # last WINDOW_DAYS, hourly
    chart = d[["actual", "predicted"]].rename(columns={"predicted": "forecast"})
    return chart, latest_mean, last_day, latest_has_actual, cutoff, actual_end

pred_df, pred_src = load_predictions()
prices, prices_src = load_prices()

# Each row: the spread (actual vs forecast) on the left, the two actual price
# levels on the right. No slider — every chart pans and zooms directly.
if pred_df is not None:
    for sp in ["DE-PL", "DE-FR"]:
        a, b, la, lb = PAIR[sp]
        chart, latest, last_day, has_actual, cutoff, actual_end = spread_series(pred_df, sp)
        fmade = (last_day - pd.Timedelta(days=1)).date()   # forecast is made the day before delivery

        st.subheader(f"{sp}  ·  €/MWh")
        scol, pcol = st.columns(2)
        with scol:
            st.markdown(f"**Power price spread ({la} minus {lb}) — actual vs forecast "
                        f"(midnight-to-midnight of the day after {fmade})**")
            st.markdown(f"There are frequent issues downloading entsoe data.")
            st.markdown(f"Straight lines are used where data downloads are missing.")
            st.markdown(f"If last forecast date = last actual date, today's data was not available.")         
            # columns are ["actual", "forecast"] -> light grey actual, red forecast
            st.line_chart(chart, height=280, color=["#b0b0b0", "#d62728"])
        with pcol:
            st.markdown(f"**{la} and {lb} price levels — actual**")
            if prices is not None:
                st.line_chart(price_pair(prices, sp, cutoff, actual_end), height=280)
            else:
                st.info("Price levels unavailable (no modelling_table found).")
    st.caption(f"Data source: ENTSOE — predictions from **{pred_src}**"
               + (f", prices from **{prices_src}**" if prices is not None else ""))
else:
    st.info("No predictions available yet — load `predictions` into Neon "
            "(or generate `predictions.csv`), then reload.")

st.divider()

# ----------------------------------------------------------------- SHAP interpretation (DE-PL)
st.header("Machine Learning Drivers Summary")

# plain-English description of each model feature, shown next to its SHAP value
DRIVER_DESC = {
    "wind_solar_fc_DE": "Germany day-ahead wind + solar generation forecast",
    "wind_solar_fc_FR": "France day-ahead wind + solar generation forecast",
    "wind_solar_fc_PL": "Poland day-ahead wind + solar generation forecast",
    "load_fc_DE": "Germany day-ahead load (demand) forecast",
    "load_fc_FR": "France day-ahead load forecast",
    "load_fc_PL": "Poland day-ahead load forecast",
    "gen_fc_DE": "Germany day-ahead total generation forecast",
    "gen_fc_FR": "France day-ahead total generation forecast",
    "gen_fc_PL": "Poland day-ahead total generation forecast",
    "net_pos_FR": "France day-ahead net position (net exports)",
    "net_pos_PL": "Poland day-ahead net position (net exports)",
    "outage_DE": "Germany generation capacity unavailable (outages)",
    "outage_FR": "France generation capacity unavailable (outages)",
    "outage_PL": "Poland generation capacity unavailable (outages)",
    "sched_exch_DE_FR": "Scheduled day-ahead exchange, Germany–France",
    "sched_exch_DE_NL": "Scheduled day-ahead exchange, Germany–Netherlands",
    "sched_exch_DE_BE": "Scheduled day-ahead exchange, Germany–Belgium",
    "sched_exch_DE_AT": "Scheduled day-ahead exchange, Germany–Austria",
    "sched_exch_DE_CH": "Scheduled day-ahead exchange, Germany–Switzerland",
    "sched_exch_DE_PL": "Scheduled day-ahead exchange, Germany–Poland",
    "sched_exch_DE_CZ": "Scheduled day-ahead exchange, Germany–Czechia",
    "sched_exch_DE_DK_1": "Scheduled day-ahead exchange, Germany–Denmark (DK1)",
    "sched_exch_DE_DK_2": "Scheduled day-ahead exchange, Germany–Denmark (DK2)",
    "sched_exch_DE_SE_4": "Scheduled day-ahead exchange, Germany–Sweden (SE4)",
    "sched_exch_DE_NO_2": "Scheduled day-ahead exchange, Germany–Norway (NO2)",
    "spread_DE_PL_lag24": "DE-PL spread 24 h earlier (previous day, same hour)",
    "spread_DE_PL_lag48": "DE-PL spread 48 h earlier",
    "spread_DE_PL_lag168": "DE-PL spread 168 h earlier (one week before)",
    "is_weekend": "Weekend indicator (1 = Sat/Sun)",
    "is_holiday_DE": "German public-holiday indicator",
    "is_holiday_FR": "French public-holiday indicator",
    "is_holiday_PL": "Polish public-holiday indicator",
    "hour": "Hour of day (0–23)",
    "dayofweek": "Day of week (0 = Monday)",
    "month": "Month of year (1–12)",
}

def show_image(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.caption(f"⚠️ `{path}` not found — run `python spread_forecasting.py shap --pre-auction` to generate it.")

if os.path.exists("shap_ranking_preauction.csv"):
    rank = pd.read_csv("shap_ranking_preauction.csv", index_col=0).head(10)
    tbl = pd.DataFrame({
        "Driver": rank.index,
        "What it is": [DRIVER_DESC.get(f, "—") for f in rank.index],
    })                                        # magnitude column dropped — the bar chart shows it
    tcol, bcol = st.columns([5, 2])          # table on the left, bar chart on the right
    with tcol:
        st.subheader("Top drivers — based on early-2025 to mid-2026 OOS data with training window 2023-2024")
        st.dataframe(tbl, hide_index=True, use_container_width=True)
    with bcol:
        show_image("shap_bar_preauction.png", "Global driver importance (mean |SHAP|).")
else:
    st.caption("SHAP ranking not found — run `python spread_forecasting.py shap --pre-auction` to generate it.")

st.divider()

# ----------------------------------------------------------------- data download
st.header("Downloads")
data_bytes, data_src = modelling_table_bytes()
if data_bytes is not None:
    st.download_button("⬇ Download all data (.csv)", data_bytes,
                       file_name="modelling_table.csv", mime="text/csv")
else:
    st.caption("No data available to download yet.")

DOC_FILE = "Power_spreads_paper.pdf"
if os.path.exists(DOC_FILE):
    with open(DOC_FILE, "rb") as f:      # read the markdown itself, not the data CSV
        st.download_button("⬇ Download model paper (.pdf)", f.read(),
                           file_name="Power_spreads_paper.pdf", mime="application/PDF")
else:
    st.caption(f"`{DOC_FILE}` not found — add your write-up to the repo under that name.")

st.caption("Github page: https://github.com/gskurokawa/Power-spread-forecasting")

