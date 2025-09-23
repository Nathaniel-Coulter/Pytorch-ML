
#!/usr/bin/env python3
"""
Machine-learning enhancement for ES-vs-Model forecasting of realized SPX at expiry.

What this does
--------------
- Loads your SPX, ES, 3M T-bill, and dividend yield series (same paths as before).
- Builds a per-day panel (as-of dates up to the contract expiry) with engineered features:
  * Core: S0 (SPX), ES price, r, d, T (days/years to expiry), Model_F (carry formula), bases (ES-Model, ES-SPX, Model-SPX)
  * Momentum: 5d/10d/20d deltas of SPX, ES, Model_F
  * Term structure-ish: r - d, r*d interaction
- Target: realized SPX at the contract expiry (constant Y across as-of dates).
- Walk-forward evaluation (TimeSeriesSplit) with 3 models:
  1) Ridge (linear)
  2) RandomForestRegressor
  3) GradientBoostingRegressor
- Baselines for comparison:
  * Baseline_ES = ES price (as-of)
  * Baseline_Model = Model_F (as-of carry formula)
- Outputs:
  - ml_per_day_predictions.csv
  - ml_summary_mae.csv
  - ml_mae_bar.png

Usage
-----
python ml_predict_realized_spx.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# ------------ File paths (your Arch paths) ------------
SPX_CSV = "/home/snowden/Desktop/quant_portfolio_scaffold/data/spx.csv"
ES_CONTRACT_CSV = "/home/snowden/Desktop/quant_portfolio_scaffold/data/es_target.csv"
DGS3MO_CSV = "/home/snowden/Desktop/quant_portfolio_scaffold/data/fred_dgs3mo.csv"
DIV_YLD_CSV = "/home/snowden/Desktop/quant_portfolio_scaffold/data/spx_div_yield.csv"

DAY_COUNT = 365.0

def third_friday(y: int, m: int) -> date:
    d = date(y, m, 1)
    while d.weekday() != 4:  # Friday
        d += timedelta(days=1)
    return d + timedelta(days=14)

def nearest_quarterly_month(d: date) -> int:
    m = ((d.month - 1)//3 + 1) * 3  # 3,6,9,12
    return m if m in (3,6,9,12) else 12

def load_series(path: str, date_col="date"):
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    return df

def compute_forward_discrete(S0, r_pct, d_pct, T_years):
    r = (r_pct or 0.0)/100.0
    d = (d_pct or 0.0)/100.0
    return S0 * ((1.0 + r - d) ** T_years)

def build_panel():
    spx = load_series(SPX_CSV, "date")[["date", "close"]].rename(columns={"close":"spx_close"})
    es = load_series(ES_CONTRACT_CSV, "date")[["date", "settle"]].rename(columns={"settle":"es_settle"})
    r3m = load_series(DGS3MO_CSV, "date")[["date", "value"]].rename(columns={"value":"r3m_pct"})
    div = load_series(DIV_YLD_CSV, "date")[["date", "value"]].rename(columns={"value":"div_pct"})

    last_es_date = es["date"].max()
    expiry_month = nearest_quarterly_month(last_es_date)
    expiry = third_friday(last_es_date.year, expiry_month)

    panel = (
        es.merge(spx, on="date", how="left")
          .merge(r3m, on="date", how="left")
          .merge(div, on="date", how="left")
    )
    panel = panel[panel["date"] <= expiry].dropna(subset=["spx_close", "es_settle", "r3m_pct", "div_pct"]).copy()

    panel["T_days"] = (pd.to_datetime(expiry) - pd.to_datetime(panel["date"])).dt.days
    panel = panel[panel["T_days"] >= 0].copy()
    panel["T_years"] = panel["T_days"] / DAY_COUNT
    panel["model_F"] = panel.apply(
        lambda r: compute_forward_discrete(r["spx_close"], r["r3m_pct"], r["div_pct"], r["T_years"]), axis=1
    )

    # Realized SPX at expiry
    spx_expiry_row = spx[spx["date"] <= expiry].sort_values("date").iloc[-1]
    realized = float(spx_expiry_row["spx_close"])
    panel["realized_spx_at_expiry"] = realized

    # Bases
    panel["basis_es_minus_model"] = panel["es_settle"] - panel["model_F"]
    panel["basis_es_minus_spx"] = panel["es_settle"] - panel["spx_close"]
    panel["basis_model_minus_spx"] = panel["model_F"] - panel["spx_close"]

    # Momentum features (simple differences)
    panel = panel.sort_values("date")
    for col in ["spx_close","es_settle","model_F"]:
        for wnd in [5,10,20]:
            panel[f"d{wnd}_{col}"] = panel[col].diff(wnd)

    # Term-structure-ish
    panel["r_minus_d"] = panel["r3m_pct"] - panel["div_pct"]
    panel["r_times_d"] = panel["r3m_pct"] * panel["div_pct"]

    return panel, expiry

def run_ml(panel):
    # Feature set
    features = [
        "spx_close","es_settle","r3m_pct","div_pct","T_days","T_years",
        "model_F","basis_es_minus_model","basis_es_minus_spx","basis_model_minus_spx",
        "d5_spx_close","d10_spx_close","d20_spx_close",
        "d5_es_settle","d10_es_settle","d20_es_settle",
        "d5_model_F","d10_model_F","d20_model_F",
        "r_minus_d","r_times_d"
    ]
    # Drop initial rows with NaNs from momentum diffs
    df = panel.dropna(subset=features).copy()

    X = df[features].values
    y = df["realized_spx_at_expiry"].values
    dates = df["date"].values

    # Baselines
    df["baseline_ES"] = df["es_settle"].values
    df["baseline_Model"] = df["model_F"].values

    # TimeSeriesSplit (expanding)
    tscv = TimeSeriesSplit(n_splits=5)
    preds = {
        "Ridge": np.full_like(y, np.nan, dtype=float),
        "RF":    np.full_like(y, np.nan, dtype=float),
        "GB":    np.full_like(y, np.nan, dtype=float),
    }

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        # Ridge
        ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
        ridge.fit(X_tr, y_tr)
        preds["Ridge"][test_idx] = ridge.predict(X_te)

        # RandomForest
        rf = RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        preds["RF"][test_idx] = rf.predict(X_te)

        # GradientBoosting
        gb = GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42
        )
        gb.fit(X_tr, y_tr)
        preds["GB"][test_idx] = gb.predict(X_te)

    # Collect per-day table
    out = df[["date","realized_spx_at_expiry","es_settle","model_F"]].copy()
    out = out.rename(columns={"es_settle":"baseline_ES","model_F":"baseline_Model"})
    for k, arr in preds.items():
        out[f"pred_{k}"] = arr

    # Errors
    for col in ["baseline_ES","baseline_Model","pred_Ridge","pred_RF","pred_GB"]:
        out[f"abs_err_{col}"] = np.abs(out[col] - out["realized_spx_at_expiry"])

    out.to_csv("ml_per_day_predictions.csv", index=False)

    # Summary MAE
    summary = []
    for col in ["baseline_ES","baseline_Model","pred_Ridge","pred_RF","pred_GB"]:
        mae = out[f"abs_err_{col}"].mean()
        summary.append({"model": col, "MAE": mae})
    summary_df = pd.DataFrame(summary).sort_values("MAE")
    summary_df.to_csv("ml_summary_mae.csv", index=False)

    # Plot MAE bar
    plt.figure()
    plt.bar(summary_df["model"], summary_df["MAE"])
    plt.title("Mean Absolute Error vs. Realized SPX at Expiry")
    plt.xlabel("Method")
    plt.ylabel("MAE (index points)")
    plt.xticks(rotation=20)
    plt.savefig("ml_mae_bar.png", bbox_inches="tight")
    plt.close()

    print("Saved: ml_per_day_predictions.csv, ml_summary_mae.csv, ml_mae_bar.png")
    print(summary_df.to_string(index=False))

def main():
    panel, expiry = build_panel()
    run_ml(panel)

if __name__ == "__main__":
    main()
