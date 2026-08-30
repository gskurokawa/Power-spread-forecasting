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

# ----------------------------------------------------------------- results (final walk-forward, 2025+)
RESULTS = pd.DataFrame(
    [("DE-PL", "LASSO", "direct", 16.24, 25.59, 15.52),
     ("DE-PL", "LASSO", "differenced", 17.07, 26.12, 14.34),
     ("DE-PL", "XGBoost", "direct", 14.40, 25.43, 15.75),
     ("DE-PL", "XGBoost", "differenced", 16.63, 25.99, 15.32),
     ("DE-FR", "LASSO", "direct", 19.21, 26.59, 24.19),
     ("DE-FR", "LASSO", "differenced", 18.85, 26.30, 24.08),
     ("DE-FR", "XGBoost", "direct", 19.17, 27.52, 24.01),
     ("DE-FR", "XGBoost", "differenced", 20.59, 28.78, 21.65)],
    columns=["spread", "model", "form", "MAE", "RMSE", "Sharpe"])

PERSISTENCE = {"DE-PL": 22.05, "DE-FR": 24.37}
VERDICT = {
    "DE-PL": ("**Congestion-prone border — modelling choices matter.** The tuned XGBoost "
              "on the spread directly is the best model (14.40 €/MWh). It beats LASSO by "
              "1.84 €/MWh and beats differencing two price forecasts by 2.23 €/MWh — both "
              "significant at p < 0.001 (Diebold–Mariano, robust to a two-week HAC bandwidth)."),
    "DE-FR": ("**Well-coupled border — nothing wins.** All four approaches land near 19 €/MWh, "
              "and no difference is statistically significant (p ≈ 0.9 for XGBoost vs LASSO). "
              "Differencing two clean price forecasts loses nothing here."),
}

# ----------------------------------------------------------------- header
st.title("⚡ Forecasting European Power Price Spreads")
st.markdown(
    "Day-ahead German–French and German–Polish price spreads, forecast with a leakage-safe, "
    "walk-forward model. Below: the model's predictions against what actually happened, and its "
    "forecast for the latest available day."
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
    return chart, latest_mean, last_day

pred_df, pred_src = load_predictions()
if pred_df is not None:
    st.caption(f"Data source: **{pred_src}**")
    days = st.slider("Days of history to show", min_value=14, max_value=180, value=60, step=7)
    for sp in ["DE-PL", "DE-FR"]:
        chart, latest, last_day = spread_series(pred_df, sp, days)
        st.subheader(f"{sp} spread  ·  €/MWh")
        mcol, ccol = st.columns([1, 5])
        mcol.metric("Latest daily forecast", f"{latest:+.1f} €/MWh")
        mcol.caption(f"mean predicted spread for {last_day.date()} (most recent day)")
        ccol.line_chart(chart, height=260)
    st.caption(
        "The **forecast (latest day)** segment is the model's prediction for the most recent day "
        "in the data. Once the live daily ENTSO-E pull is running, this point becomes tomorrow's "
        "genuine day-ahead forecast — the rest of the code is unchanged."
    )
else:
    st.info("No predictions available yet — load `predictions` into Neon (or generate "
            "`predictions.csv` with `python spread_forecasting.py compare`), then reload.")

st.divider()

# ----------------------------------------------------------------- headline metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Best DE-PL model", "14.40 €/MWh", "XGBoost · direct")
c2.metric("vs LASSO", "−1.84 €/MWh", "p < 0.001", delta_color="inverse")
c3.metric("vs differencing", "−2.23 €/MWh", "p < 0.001", delta_color="inverse")
c4.metric("Test window", "2025–26", "walk-forward, out-of-sample")

st.divider()

# ----------------------------------------------------------------- interactive comparison
st.header("Model comparison")
spread = st.radio("Choose a spread", ["DE-PL", "DE-FR"], horizontal=True)

sub = RESULTS[RESULTS["spread"] == spread].copy()
sub["approach"] = sub["model"] + " · " + sub["form"]

left, right = st.columns([3, 2])
with left:
    disp = sub.set_index("approach")[["MAE", "RMSE", "Sharpe"]]
    styler = disp.style.format("{:.2f}").highlight_min(
        subset=["MAE"], color="rgba(224,145,46,0.35)")
    st.dataframe(styler, use_container_width=True)
    st.caption(f"Persistence baseline (t−24h): {PERSISTENCE[spread]:.2f} €/MWh · lower MAE is better")
with right:
    st.bar_chart(sub.set_index("approach")["MAE"], height=280)

st.info(VERDICT[spread])

st.divider()

# ----------------------------------------------------------------- SHAP interpretation (DE-PL)
st.header("Why the DE-PL model works — SHAP")
st.markdown(
    "The **renewable forecasts** set the price level smoothly (merit order), while the "
    "**cross-border coupling** variables act as sharp **thresholds** — the structure a linear "
    "model cannot capture, and the reason the tree wins on DE-PL."
)

if os.path.exists("shap_ranking.csv"):
    rank = pd.read_csv("shap_ranking.csv", index_col=0)
    rank.columns = ["mean |SHAP| (€/MWh)"]
    st.subheader("Top drivers")
    st.dataframe(rank.head(10).style.format("{:.2f}"), use_container_width=True)

def show_image(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.caption(f"⚠️ `{path}` not found — run `python spread_forecasting.py shap` to generate it.")

ic1, ic2 = st.columns(2)
with ic1:
    show_image("shap_bar.png", "Global driver importance (mean |SHAP|).")
with ic2:
    show_image("shap_beeswarm.png", "Per-hour contributions, coloured by feature value.")

st.subheader("Two thresholds")
tc1, tc2 = st.columns(2)
with tc1:
    show_image("shap_dependence_net_pos_PL.png",
               "Polish net position: a discontinuous jump as Poland turns net exporter.")
with tc2:
    show_image("shap_dependence_wind_solar_fc_DE.png",
               "German renewables: a cliff near 55 GW, where German prices turn negative.")

st.divider()

# ----------------------------------------------------------------- data download
st.header("Get the data")
data_bytes, data_src = modelling_table_bytes()
if data_bytes is not None:
    st.download_button("⬇ Download the modelling table (CSV)", data_bytes,
                       file_name="modelling_table.csv", mime="text/csv")
    st.caption(f"~66,000 hourly rows · leakage-safe features for three bidding zones · source: {data_src}.")
else:
    st.caption("No data available to download yet.")

with st.expander("Method — how it was built"):
    st.markdown(
        "- **Data**: ENTSO-E day-ahead prices, forecasts, scheduled exchanges, net positions and "
        "outages for DE-LU, FR, PL. PLN→EUR break and the 15-minute switch corrected.\n"
        "- **Features**: leakage-safe only — day-ahead forecasts, scheduled flows, calendar, and "
        "24/48/168h target lags. No realised prices, actuals or physical flows.\n"
        "- **Models**: LASSO benchmark vs XGBoost tuned by Optuna (rolling-window CV, tracked in MLflow).\n"
        "- **Evaluation**: rolling two-year window, retrained monthly, tested once on 2025+; "
        "MAE / RMSE / Sharpe with Diebold–Mariano significance tests.")

st.caption("Portfolio project · data © ENTSO-E Transparency Platform · not investment advice.")
