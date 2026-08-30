# ⚡ Forecasting Selected European Power Price Spreads

A leakage-safe, walk-forward study of the German–French (DE–FR) and German–Polish (DE–PL)
day-ahead electricity price spreads, deployed as a live, self-updating dashboard. It asks a
specific question: **is an inter-zonal price spread forecast better by modelling it directly,
or by differencing two independent zonal forecasts?** It answers this out-of-sample, with
significance testing and SHAP interpretation.

**Headline result.** For the congestion-prone DE–PL border, a tuned gradient-boosting model
applied to the spread *directly* is the best forecast (14.40 €/MWh MAE), significantly beating
both a LASSO benchmark (−1.84 €/MWh) and the differenced-forecast approach (−2.23 €/MWh), each
at p < 0.001. For the well-coupled DE–FR border, no approach significantly beats any other,
exactly where the coupling/decoupling mechanism predicts spread-specific structure should be absent.

**Live dashboard:** <https://power-spread-forecasting.streamlit.app/>

## The app

The Streamlit dashboard (`app.py`) shows:

- **actual vs predicted** spread for both borders over the test period, plus the forecast for the next day;
- the **SHAP drivers**: a ranked table of what moves the forecast, with plain-English descriptions, and the importance bar chart;
- **downloads** for the underlying data and the project write-up.

## Live system

The dashboard is not a static snapshot; it updates itself daily:

- `daily_update.py` refetches the latest ENTSO-E data, retrains the model, and writes fresh forecasts to a cloud Postgres SQL database.
- A **GitHub Actions** workflow (`.github/workflows/daily-update.yml`) runs it automatically each weekday, using repository secrets for the ENTSO-E token and database URL.
- The Streamlit app reads the predictions and data live from the database on each load.
- Everything runs on free service tiers, at zero standing cost.

## Run it locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Reproduce the analysis

All modelling lives in one script with stage subcommands:

```bash
python spread_forecasting.py lasso        # LASSO walk-forward benchmark
python spread_forecasting.py xgb-manual   # single-split XGBoost fit (intuition)
python spread_forecasting.py optuna       # Optuna tuning (rolling-window CV)  [--mlflow]
python spread_forecasting.py compare      # final LASSO vs XGBoost  -> predictions.csv
python spread_forecasting.py dm           # Diebold–Mariano significance tests
python spread_forecasting.py shap         # SHAP interpretation -> shap_*.png
```

The deployment scripts (`load_to_postgres.py` for the one-time load, `daily_update.py` for the
daily refresh) use `requirements-daily.txt`.

## Method

- **Data**: ENTSO-E Transparency Platform (day-ahead prices, load / wind-solar / generation
  forecasts, scheduled exchanges, net positions, outages) for DE-LU, FR, PL: hourly, from
  Dec 2018 and updated daily, with the PLN→EUR currency break and the Oct-2025 15-minute
  settlement switch corrected.
- **Features**: leakage-safe only: information known at day-ahead gate closure, plus 24/48/168h
  target lags. No realised prices, actuals, or physical flows.
- **Models**: LASSO (LEAR-style) benchmark vs XGBoost tuned with Optuna over a rolling-window
  cross-validation, tracked in MLflow.
- **Evaluation**: rolling two-year window, retrained monthly, tested out-of-sample; MAE / RMSE /
  Sharpe with Diebold–Mariano tests under a Newey–West HAC variance.
