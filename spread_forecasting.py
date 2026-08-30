#!/usr/bin/env python
"""
Inter-zonal electricity price spread forecasting -- consolidated pipeline.
DE-FR and DE-PL day-ahead spreads; LASSO benchmark vs tuned XGBoost;
walk-forward evaluation; Diebold-Mariano significance tests.

All stages share one set of feature / model / metric definitions below.
Pick a stage:

  python spread_forecasting.py lasso        LASSO walk-forward benchmark table
                                            (persistence / static / WF no-lags / WF lags)
  python spread_forecasting.py xgb-manual   single dev-split XGBoost fit for intuition
                                            [--max-depth 4 --min-child-weight 1 ...]
  python spread_forecasting.py optuna       Optuna tuning on dev (rolling-window CV)
                                            [--trials 60] [--mlflow]  -> writes best-params JSON
  python spread_forecasting.py compare      final LASSO vs tuned-XGBoost walk-forward on 2025+
                                            (MAE/RMSE/Sharpe) -> writes wf_errors.csv
  python spread_forecasting.py dm           all Diebold-Mariano tests from wf_errors.csv
  python spread_forecasting.py shap         SHAP interpretation of the winning DE-PL model
                                            (bar, beeswarm, dependence plots, ranking CSV)

Common options:
  --csv PATH       default modelling_table.csv, the full 51-column table
  --pre-auction    operational model: drop the auction-outcome features (scheduled
                   exchanges and net positions, which publish ~12:45 with the price),
                   so the model forecasts BEFORE the auction clears. Every stage then
                   reads/writes '*_preauction' artifacts, leaving the full-model files
                   untouched.

Heavy libraries are imported lazily inside the stages that need them, so the
`lasso` and `dm` stages run with only pandas/numpy/scipy/scikit-learn installed;
`xgb-manual`/`optuna`/`compare` additionally need xgboost (and optuna/mlflow),
and `shap` needs xgboost + shap + matplotlib.
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ configuration
BEST_JSON  = "xgb_best_params_DEPL.json"
ERRORS_CSV = "wf_errors.csv"
SPREADS = ["spread_DE_FR", "spread_DE_PL"]
DIFF_MAP = {"spread_DE_FR": ("price_DE", "price_FR"),
            "spread_DE_PL": ("price_DE", "price_PL")}
ALL_TARGETS = SPREADS + ["price_DE", "price_FR", "price_PL"]
LAG_HOURS = (24, 48, 168)
TEST_START = "2025-01-01"           # untouched final test period start

FC = ["load_fc_DE", "load_fc_FR", "load_fc_PL", "wind_solar_fc_DE", "wind_solar_fc_FR",
      "wind_solar_fc_PL", "gen_fc_DE", "gen_fc_FR", "gen_fc_PL"]
BIN = ["is_weekend", "is_holiday_DE", "is_holiday_FR", "is_holiday_PL"]
CAT = ["hour", "dayofweek", "month"]

XGB_FIXED = dict(objective="reg:squarederror", eval_metric="mae", tree_method="hist",
                 n_jobs=-1, random_state=0)

# ---- pre-auction (operational) mode --------------------------------------------
# Scheduled cross-border exchanges (sched_exch_DE_*) and net positions (net_pos_*)
# are day-ahead market-COUPLING outputs: they publish together with the price
# (~12:45), so any model using them can never forecast AHEAD of the auction.
# --pre-auction drops them, leaving only inputs known before gate closure
# (load / wind-solar / generation forecasts, outages, target lags, calendar). This
# is the operational model; the full model keeps them as a same-time benchmark.
AUCTION_OUTCOME_PREFIXES = ("sched_exch_DE_", "net_pos_")
PRE_AUCTION = False        # set by --pre-auction in main()

def _tag():
    return "_preauction" if PRE_AUCTION else ""

def out_name(base):
    """Tag output filenames in pre-auction mode so they never clobber the
    full-model artifacts (wf_errors.csv, predictions.csv, shap_*.png, ...)."""
    root, dot, ext = base.rpartition(".")
    return f"{root}{_tag()}{dot}{ext}" if dot else base + _tag()

def best_params_path():
    return f"xgb_best_params_DEPL{_tag()}.json"

def load_best():
    """XGBoost hyperparameters. In pre-auction mode prefer a pre-auction-tuned file
    if present, else reuse the full-model params (Section 6 found a hard accuracy
    floor, so re-tuning barely moves the result)."""
    path = best_params_path()
    if PRE_AUCTION and not os.path.exists(path):
        print(f"  [pre-auction] {path} not found -> reusing full-model params '{BEST_JSON}'")
        path = BEST_JSON
    return json.load(open(path))

# ------------------------------------------------------------------ data / features
def load_data(csv):
    df = pd.read_csv(csv, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()

def num_features(df):
    sched = [c for c in df.columns if c.startswith("sched_exch_DE_")]
    feats = FC + ["outage_DE", "outage_FR", "outage_PL"] + sched + ["net_pos_FR", "net_pos_PL"]
    if PRE_AUCTION:                       # operational model: drop auction-outcome inputs
        feats = [c for c in feats if not c.startswith(AUCTION_OUTCOME_PREFIXES)]
    return feats

def add_lags(df, target):
    d = df.copy(); lags = []
    for L in LAG_HOURS:
        c = f"{target}_lag{L}"; d[c] = d[target].shift(L); lags.append(c)
    return d, lags

# ------------------------------------------------------------------ models
def lasso_pipe(num):
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import TimeSeriesSplit
    pre = ColumnTransformer([("n", StandardScaler(), num), ("b", "passthrough", BIN),
                             ("c", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
    return Pipeline([("p", pre), ("l", LassoCV(alphas=np.logspace(-1, 2, 15),
                                               cv=TimeSeriesSplit(3), max_iter=2000))])

def xgb_estimator(best, early_stop=True):
    import xgboost as xgb
    params = dict(best)
    if early_stop:
        params.setdefault("early_stopping_rounds", 50)
    else:
        params.pop("early_stopping_rounds", None)
    return xgb.XGBRegressor(enable_categorical=True, **params, **XGB_FIXED)

# ------------------------------------------------------------------ walk-forward
def walk_forward(df, target, model, best=None, use_lags=True):
    """Rolling 2-year window, retrained monthly, predicting each month of 2025+."""
    d, lags = add_lags(df, target) if use_lags else (df.copy(), [])
    num = num_features(df) + lags
    feats = num + BIN + CAT
    if model == "xgb":
        for c in CAT:
            d[c] = d[c].astype("category")
    months = pd.date_range(pd.Timestamp(TEST_START, tz="UTC"), d.index.max(), freq="MS")
    out = []
    for m in months:
        ws = m - pd.DateOffset(years=2)
        win = d[(d.index >= ws) & (d.index < m)].dropna(subset=feats + [target])
        ch = d[(d.index >= m) & (d.index < m + pd.DateOffset(months=1))].dropna(subset=feats + [target])
        if len(win) < 2000 or len(ch) == 0:
            continue
        if model == "lasso":
            p = lasso_pipe(num).fit(win[feats], win[target]).predict(ch[feats])
        else:
            cut = m - pd.Timedelta(weeks=8)
            fit = win[win.index < cut]; esv = win[win.index >= cut]
            if len(esv) < 200:
                fit, esv = win, win
            est = xgb_estimator(best)
            est.fit(fit[feats], fit[target], eval_set=[(esv[feats], esv[target])], verbose=False)
            p = est.predict(ch[feats])
        out.append(pd.Series(p, index=ch.index))
    return pd.concat(out) if out else pd.Series(dtype=float)

def static_lasso(df, target, use_lags=True):
    """Train once on <=2023, predict 2025+ (documents the regime-shift failure)."""
    d, lags = add_lags(df, target) if use_lags else (df.copy(), [])
    num = num_features(df) + lags
    feats = num + BIN + CAT
    tr = d[d.index.year <= 2023].dropna(subset=feats + [target])
    te = d[d.index >= pd.Timestamp(TEST_START, tz="UTC")].dropna(subset=feats + [target])
    p = lasso_pipe(num).fit(tr[feats], tr[target]).predict(te[feats])
    return pd.Series(p, index=te.index)

# ------------------------------------------------------------------ metrics
def mae(a, b):  return float(np.abs(np.asarray(a) - np.asarray(b)).mean())
def rmse(a, b): return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).mean()))
def sharpe(yt, yp, cost=1.0, thr=5.0):
    yt = pd.Series(np.asarray(yt), index=yt.index)
    pos = np.where(np.asarray(yp) > thr, 1, np.where(np.asarray(yp) < -thr, -1, 0))
    dd = pd.Series(pos * yt.values - cost * np.abs(pos), index=yt.index).resample("1D").sum()
    return np.nan if dd.std() == 0 else float(dd.mean() / dd.std() * np.sqrt(365))

# ------------------------------------------------------------------ Diebold-Mariano
def hac_var(d, bw):
    d = np.asarray(d, float); n = len(d); dc = d - d.mean()
    s = (dc @ dc) / n
    for k in range(1, bw + 1):
        s += 2.0 * (1.0 - k / (bw + 1.0)) * (dc[k:] @ dc[:-k]) / n
    return s / n

def dm_sweep(e_a, e_b, label_a, label_b):
    """|e_a| - |e_b|;  d<0 => model A better. Bandwidth sweep on a HAC variance."""
    d = np.abs(e_a) - np.abs(e_b); n = len(d)
    print(f"   d = |{label_a}| - |{label_b}|   mean {d.mean():+.3f}  "
          f"(MAE {np.abs(e_a).mean():.2f} vs {np.abs(e_b).mean():.2f})")
    for bw in [24, 72, 168, 241, 336]:
        stat = d.mean() / np.sqrt(hac_var(d, bw)) * np.sqrt((n - 1) / n)
        p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
        who = label_a if d.mean() < 0 else label_b
        verdict = "no sig. diff" if p > 0.05 else f"{who} better"
        print(f"      bw={bw:>4}h ({bw//24:>2}d):  DM {stat:+6.2f}  p {p:6.3f}   {verdict}")

# ================================================================== stages
def stage_lasso(df):
    print("LASSO walk-forward benchmark  (final test 2025+, MAE / RMSE EUR/MWh)")
    print("=" * 66)
    rows = {}
    for sp in SPREADS:
        te_idx = df[df.index >= pd.Timestamp(TEST_START, tz="UTC")].index
        per = df[sp].shift(24)
        wf0 = walk_forward(df, sp, "lasso", use_lags=False)
        wf1 = walk_forward(df, sp, "lasso", use_lags=True)
        st = static_lasso(df, sp, use_lags=False)   # documented no-lags static failure
        idx = wf1.index
        act = df.loc[idx, sp]
        rows[sp] = {
            "persistence": mae(act.loc[idx], per.loc[idx]),
            "static LASSO": mae(df.loc[st.index, sp], st),
            "WF no-lags": mae(df.loc[wf0.index, sp], wf0),
            "WF with-lags": mae(act, wf1),
        }
    print(f"\n{'approach':16}{'DE-FR':>10}{'DE-PL':>10}")
    for k in ["persistence", "static LASSO", "WF no-lags", "WF with-lags"]:
        print(f"{k:16}{rows['spread_DE_FR'][k]:10.2f}{rows['spread_DE_PL'][k]:10.2f}")


def stage_xgb_manual(df, args):
    best = dict(n_estimators=args.n_estimators, learning_rate=args.learning_rate,
                max_depth=args.max_depth, min_child_weight=args.min_child_weight,
                subsample=args.subsample, colsample_bytree=args.colsample,
                reg_lambda=args.reg_lambda, reg_alpha=args.reg_alpha, gamma=args.gamma)
    target = args.target
    d, lags = add_lags(df, target)
    feats = num_features(df) + lags + BIN + CAT
    for c in CAT:
        d[c] = d[c].astype("category")
    tr = d[(d.index >= "2021-01-01") & (d.index < "2024-01-01")].dropna(subset=feats + [target])
    va = d[(d.index >= "2024-01-01") & (d.index < "2025-01-01")].dropna(subset=feats + [target])
    print(f"{target}  train {len(tr)} (2021-23)  valid {len(va)} (2024)   params: {best}")
    est = xgb_estimator(best)
    est.fit(tr[feats], tr[target], eval_set=[(va[feats], va[target])], verbose=False)
    print(f"early stopping chose {est.best_iteration} trees (ceiling {args.n_estimators})")
    ptr, pva = est.predict(tr[feats]), est.predict(va[feats])
    print(f"  train  MAE {mae(tr[target],ptr):6.2f}  RMSE {rmse(tr[target],ptr):6.2f}")
    print(f"  valid  MAE {mae(va[target],pva):6.2f}  RMSE {rmse(va[target],pva):6.2f}")
    print(f"  gap (valid-train MAE) = {mae(va[target],pva)-mae(tr[target],ptr):+.2f}")
    imp = pd.Series(est.feature_importances_, index=feats).sort_values(ascending=False)
    print("  top 12 features:", ", ".join(f"{n}({v:.2f})" for n, v in imp.head(12).items()))


def stage_optuna(df, args):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    target = args.target
    d, lags = add_lags(df, target)
    feats = num_features(df) + lags + BIN + CAT
    for c in CAT:
        d[c] = d[c].astype("category")
    # rolling 2-year train -> next 6 months validate, three recent post-crisis folds
    FOLDS = [("2021-07-01", "2023-07-01", "2023-07-01", "2024-01-01"),
             ("2022-01-01", "2024-01-01", "2024-01-01", "2024-07-01"),
             ("2022-07-01", "2024-07-01", "2024-07-01", "2025-01-01")]
    def slc(a, b):
        s = d[(d.index >= a) & (d.index < b)].dropna(subset=feats + [target])
        return s[feats], s[target]
    fixed = dict(n_estimators=2000, early_stopping_rounds=50)
    mlflow = None
    if args.mlflow:
        import mlflow as _mlflow; mlflow = _mlflow
        mlflow.set_experiment(f"xgb_{target}")

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 300, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-3, 5.0, log=True))
        maes = []
        for ts, te, vs, ve in FOLDS:
            Xtr, ytr = slc(ts, te); Xva, yva = slc(vs, ve)
            est = xgb_estimator({**params, **fixed})
            est.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            maes.append(mae(yva, est.predict(Xva)))
        val = float(np.mean(maes))
        if mlflow:
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(params); mlflow.log_metric("cv_mae", val)
        print(f"  trial {trial.number:>3}  MAE {val:6.3f}  best {min(val, getattr(objective,'b',val)):6.3f}")
        objective.b = min(val, getattr(objective, "b", val))
        return val

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    if mlflow:
        with mlflow.start_run(run_name=f"optuna_{target}"):
            study.optimize(objective, n_trials=args.trials)
    else:
        study.optimize(objective, n_trials=args.trials)
    print(f"\nbest CV MAE {study.best_value:.3f}   params {study.best_params}")
    json.dump({**study.best_params, "n_estimators": 2000, "early_stopping_rounds": 50},
              open(best_params_path(), "w"), indent=2)
    print(f"saved -> {best_params_path()}")


def stage_compare(df):
    best = load_best()
    print(f"tuned XGBoost params: {best}\nwalk-forward (both models, 5 targets) ...")
    pred = {}
    for model in ["lasso", "xgb"]:
        for t in ALL_TARGETS:
            pred[(model, t)] = walk_forward(df, t, model, best=best)
            print(f"  done: {model:5} {t}")
    print("\n" + "=" * 74)
    print("LASSO vs XGBoost   walk-forward  final test 2025+")
    print("=" * 74)
    saved = {}
    for sp in SPREADS:
        a, b = DIFF_MAP[sp]
        print(f"\n{sp}")
        print(f"   {'model / form':22}{'MAE':>7}{'RMSE':>8}{'Sharpe':>8}")
        cols = {}
        for model in ["lasso", "xgb"]:
            direct = pred[(model, sp)]
            diff = (pred[(model, a)] - pred[(model, b)]).dropna()
            idx = direct.index.intersection(diff.index); idx = idx[idx.year >= 2025]
            act = df.loc[idx, sp]
            for form, ser in [("DIRECT", direct), ("DIFFERENCED", diff)]:
                print(f"   {model.upper()+' '+form:22}{mae(ser.loc[idx],act):7.2f}"
                      f"{rmse(ser.loc[idx],act):8.2f}{sharpe(act, ser.loc[idx]):8.2f}")
                cols[f"e_{model}_{form.lower()[:6]}"] = ser.loc[idx].to_numpy() - act.to_numpy()
            cols["_idx"] = idx; cols["_act"] = act.to_numpy()
        saved[sp] = pd.DataFrame({k: v for k, v in cols.items() if not k.startswith("_")},
                                 index=cols["_idx"])
    pd.concat(saved, names=["spread"]).to_csv(out_name(ERRORS_CSV))
    print(f"\n[saved {out_name(ERRORS_CSV)} for the DM stage]")

    # tidy actual-vs-predicted series for the dashboard (winning model: XGBoost direct)
    short = {"spread_DE_FR": "DE-FR", "spread_DE_PL": "DE-PL"}   # labels the dashboard filters on
    prows = []
    for sp in SPREADS:
        s = pred[("xgb", sp)]
        idx = s.index[s.index.year >= 2025]
        prows.append(pd.DataFrame({"spread": short[sp], "actual": df.loc[idx, sp].values,
                                   "predicted": s.loc[idx].values}, index=idx))
    out = pd.concat(prows); out.index.name = "timestamp"
    out.to_csv(out_name("predictions.csv"))
    print(f"[saved {out_name('predictions.csv')} for the dashboard]")


def stage_shap(df, args):
    """Interpret the winning model. One explanatory model trained on 2023-2024,
    SHAP attributed on the 2025+ test set. Calendar features as integers (not
    native categoricals) for TreeExplainer robustness -- does not change the ranking."""
    import shap, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xgboost as xgb
    target = args.target
    best = {k: v for k, v in load_best().items() if k != "early_stopping_rounds"}
    d, lags = add_lags(df, target)
    feats = num_features(df) + lags + BIN + CAT          # CAT kept as integers here
    tr = d[(d.index >= "2023-01-01") & (d.index < "2024-11-01")].dropna(subset=feats + [target])
    es = d[(d.index >= "2024-11-01") & (d.index < "2025-01-01")].dropna(subset=feats + [target])
    te = d[(d.index >= "2025-01-01")].dropna(subset=feats + [target])
    print(f"train {len(tr)}  early-stop {len(es)}  explain {len(te)}")
    model = xgb.XGBRegressor(**best, early_stopping_rounds=50, eval_metric="mae",
                             objective="reg:squarederror", tree_method="hist",
                             enable_categorical=False, n_jobs=-1, random_state=0)
    model.fit(tr[feats], tr[target], eval_set=[(es[feats], es[target])], verbose=False)
    hint = "  (pre-auction model)" if PRE_AUCTION else "  (walk-forward ~14.4)"
    print(f"explanatory model test MAE {mae(te[target], model.predict(te[feats])):.2f}{hint}")

    expl = shap.TreeExplainer(model)
    sv = expl(te[feats])
    rank = pd.Series(np.abs(sv.values).mean(0), index=feats).sort_values(ascending=False)
    rank.to_csv(out_name("shap_ranking.csv"), header=["mean_abs_shap"])
    print("\ntop 12 drivers (mean |SHAP|, EUR/MWh):")
    for nm, v in rank.head(12).items():
        print(f"  {v:6.3f}  {nm}")

    plt.figure(); shap.plots.bar(sv, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(out_name("shap_bar.png"), dpi=140, bbox_inches="tight"); plt.close()
    plt.figure(); shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(out_name("shap_beeswarm.png"), dpi=140, bbox_inches="tight"); plt.close()
    for feat in rank.head(4).index:
        plt.figure(); shap.plots.scatter(sv[:, feat], show=False)
        plt.tight_layout()
        plt.savefig(out_name(f"shap_dependence_{feat.replace('/', '_')}.png"), dpi=140, bbox_inches="tight"); plt.close()
    print(f"\nsaved: {out_name('shap_bar.png')}, {out_name('shap_beeswarm.png')}, "
          f"{out_name('shap_dependence_*.png')}, {out_name('shap_ranking.csv')}")


def stage_dm(df=None):
    err = pd.read_csv(out_name(ERRORS_CSV)).rename(columns={"Unnamed: 1": "ts"})
    print("Diebold-Mariano tests  (HAC bandwidth sweep; d<0 => first model better)")
    print("=" * 74)
    for sp in SPREADS:
        sub = err[err["spread"] == sp]
        ed_l, ef_l = sub["e_lasso_direct"].values, sub["e_lasso_differ"].values
        ed_x, ef_x = sub["e_xgb_direct"].values, sub["e_xgb_differ"].values
        print(f"\n{sp}  (n={len(sub)})")
        print("  [LASSO] direct vs differenced")
        dm_sweep(ed_l, ef_l, "lasso_direct", "lasso_diff")
        print("  [XGBoost] direct vs differenced")
        dm_sweep(ed_x, ef_x, "xgb_direct", "xgb_diff")
        print("  [direct] XGBoost vs LASSO")
        dm_sweep(ed_x, ed_l, "xgb_direct", "lasso_direct")


# ================================================================== entry point
def main():
    ap = argparse.ArgumentParser(description="Spread forecasting pipeline")
    ap.add_argument("stage", choices=["lasso", "xgb-manual", "optuna", "compare", "dm", "shap"])
    ap.add_argument("--csv", default="modelling_table.csv")
    ap.add_argument("--pre-auction", dest="pre_auction", action="store_true",
                    help="operational mode: drop auction-outcome features (scheduled "
                         "exchanges, net positions) so the model forecasts BEFORE the "
                         "auction clears; outputs are written to '*_preauction' files")
    ap.add_argument("--target", default="spread_DE_PL")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--max-depth", dest="max_depth", type=int, default=4)
    ap.add_argument("--min-child-weight", dest="min_child_weight", type=int, default=1)
    ap.add_argument("--learning-rate", dest="learning_rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample", type=float, default=0.8)
    ap.add_argument("--reg-lambda", dest="reg_lambda", type=float, default=0.0)
    ap.add_argument("--reg-alpha", dest="reg_alpha", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--n-estimators", dest="n_estimators", type=int, default=1000)
    args = ap.parse_args()

    global PRE_AUCTION
    PRE_AUCTION = args.pre_auction
    if PRE_AUCTION:
        print("=" * 66)
        print("PRE-AUCTION (operational) mode: dropping auction-outcome features")
        print(f"  excluded prefixes : {', '.join(AUCTION_OUTCOME_PREFIXES)}")
        print("  outputs tagged '_preauction' (full-model files left untouched)")
        print("=" * 66)

    if args.stage == "dm":
        stage_dm(); return
    df = load_data(args.csv)
    if args.stage == "lasso":
        stage_lasso(df)
    elif args.stage == "xgb-manual":
        stage_xgb_manual(df, args)
    elif args.stage == "optuna":
        stage_optuna(df, args)
    elif args.stage == "compare":
        stage_compare(df)
    elif args.stage == "shap":
        stage_shap(df, args)


if __name__ == "__main__":
    main()
