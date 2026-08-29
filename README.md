---
title: Power Spread Forecasting
emoji: ⚡
colorFrom: yellow
colorTo: gray
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
---

# ⚡ Forecasting European Power Price Spreads

A leakage-safe, walk-forward study of the German–French (DE–FR) and German–Polish (DE–PL)
day-ahead electricity price spreads. It asks a specific question: **is an inter-zonal price
spread forecast better by modelling it directly, or by differencing two independent zonal
forecasts?** — and answers it out-of-sample, with significance testing and SHAP interpretation.

**Headline result.** For the congestion-prone DE–PL border, a tuned gradient-boosting model
applied to the spread *directly* is the best forecast (14.40 €/MWh MAE). It significantly beats
both a LASSO benchmark (−1.84 €/MWh) and the differenced-forecast approach (−2.23 €/MWh), each at
p < 0.001. For the well-coupled DE–FR border, no approach significantly beats any other — exactly
where the coupling/decoupling mechanism predicts spread-specific structure should be absent.

## The app

The Streamlit dashboard shows, for both spreads:
- **actual vs predicted** spread over the walk-forward test period, plus the latest-day forecast;
- the **LASSO vs XGBoost** comparison (MAE / RMSE / Sharpe) with Diebold–Mariano significance verdicts;
- the **SHAP** interpretation — driver importance, per-hour contributions, and the two threshold
  effects (Polish net position, German renewables) that explain the model's edge;
- a **download** button for the underlying data.

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
python spread_forecasting.py compare      # final LASSO vs XGBoost on 2025+  -> predictions.csv
python spread_forecasting.py dm           # Diebold–Mariano significance tests
python spread_forecasting.py shap         # SHAP interpretation -> shap_*.png
```

## Method

- **Data**: ENTSO-E Transparency Platform (day-ahead prices, load / wind-solar / generation
  forecasts, scheduled exchanges, net positions, outages) for DE-LU, FR, PL — ~66,000 hourly
  rows, Dec 2018 – Jul 2026, with the PLN→EUR currency break and the Oct-2025 15-minute
  settlement switch corrected.
- **Features**: leakage-safe only — information known at day-ahead gate closure, plus 24/48/168h
  target lags. No realised prices, actuals, or physical flows.
- **Models**: LASSO (LEAR-style) benchmark vs XGBoost tuned with Optuna over a rolling-window
  cross-validation, tracked in MLflow.
- **Evaluation**: rolling two-year window, retrained monthly, tested once on 2025+; MAE / RMSE /
  Sharpe with Diebold–Mariano tests under a Newey–West HAC variance.

*Portfolio project. Data © ENTSO-E Transparency Platform. Not investment advice.*
