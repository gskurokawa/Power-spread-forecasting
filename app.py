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
st.title("⚡ DE-PL and DE-FR Day-Ahead Power Price Spread Forecasts")
st.markdown(
    "Day-ahead Germany–Poland (DE-PL) and Germany–France (DE-FR) price spreads, forecast before each day's auction with walk-forward machine-learning models."
)

# ----------------------------------------------------------------- TOP: actual vs predicted + latest forecast
def spread_series(pred_df, sp, days):
    d = pred_df[pred_df["spread"] == sp].set_index("timestamp").sort_index()
    d = d[d.index >= d.index.max() - pd.Timedelta(days=days)]
    last_day = d.index.normalize().max()
    is_last = d.index.normalize() == last_day
    chart = pd.DataFrame(index=d.index)
    chart["actual"] = d["actual"]
    chart["predicted"] = d["predicted"].where(~is_last)
    chart["forecast (latest day)"] = d["predicted"].where(is_last)
    latest_mean = float(d.loc[is_last, "predicted"].mean())
    latest_has_actual = bool(d.loc[is_last, "actual"].notna().any())
    return chart, latest_mean, last_day, latest_has_actual

pred_df, pred_src = load_predictions()
if pred_df is not None:
    days = st.slider("Select days of history to show using the slider:", min_value=14, max_value=180, value=60, step=7)
    for sp in ["DE-PL", "DE-FR"]:
        chart, latest, last_day, has_actual = spread_series(pred_df, sp, days)
        st.subheader(f"{sp} spread  ·  €/MWh")
        mcol, ccol = st.columns([1, 5])
        mcol.metric(f"Forecast for {last_day.date()}", f"{latest:+.1f} €/MWh")
        if has_actual:
            mcol.caption(f"Midnight-to-midnight mean predicted spread for {last_day.date()} "
                         "(the auction has cleared, so the actual is shown alongside)")
        else:
            mcol.caption(f"Pre-auction forecast for {last_day.date()}; the actual fills in "
                         "once the day-ahead auction clears")
        ccol.line_chart(chart, height=260)
    st.caption(f"Data source: ENTSOE from **{pred_src}**")
else:
    st.info("No predictions available yet — load `predictions` into Neon (or generate "
            "`predictions.csv` with `python spread_forecasting.py compare`), then reload.")

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
        "mean |SHAP| (€/MWh)": rank.iloc[:, 0].to_numpy(),
    })
    tcol, bcol = st.columns([5, 2])          # narrow table on the left, bar chart on the right
    with tcol:
        st.subheader("Top drivers")
        st.dataframe(
            tbl, hide_index=True, use_container_width=True,
            column_config={
                "mean |SHAP| (€/MWh)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
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

DOC_FILE = "Documentation.md"
if os.path.exists(DOC_FILE):
    with open(DOC_FILE, "rb") as f:      # read the markdown itself, not the data CSV
        st.download_button("⬇ Download model documentation (.md)", f.read(),
                           file_name="Documentation.md", mime="text/markdown")
else:
    st.caption(f"`{DOC_FILE}` not found — add your write-up to the repo under that name.")

st.caption("Github page: https://github.com/gskurokawa/Power-spread-forecasting")

