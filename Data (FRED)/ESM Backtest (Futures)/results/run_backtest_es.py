#!/usr/bin/env python3
"""
Compare ES Futures vs. Carry Model at Predicting SPX at Expiry

Usage:
  1) Edit the four FILE PATHS below if needed (they're already set to your Arch paths).
  2) Run:
       python run_backtest_es.py
  3) Outputs:
       - es_fair_value_vs_futures_backtest.csv   (per-day records)
       - summary_stats.csv                       (MAE by horizon buckets & overall)
       - error_by_days_to_expiry.png             (line chart)
       - cumulative_mae_comparison.png           (line chart)

Notes:
  - Discrete comp model: F = S0 * (1 + r - d)^T, ACT/365.
  - r and d are read as PERCENT units in your CSVs (e.g., 4.50, 1.25).
  - Realized SPX is taken at the contract's expiry (3rd Friday of the quarter).
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt

# ---------- FILE PATHS (your Arch paths) ----------
SPX_CSV = "/home/snowden/Desktop/quant_portfolio_scaffold/data/spx.csv"               # date, close
ES_CONTRACT_CSV = "/home/snowden/Desktop/quant_portfolio_scaffold/data/es_target.csv" # date, settle
DGS3MO_CSV = "/home/snowden/Desktop/quant_portfolio_scaffold/data/fred_dgs3mo.csv"    # date, value (percent)
DIV_YLD_CSV = "/home/snowden/Desktop/quant_portfolio_scaffold/data/spx_div_yield.csv" # date, value (percent)

# ---------- SETTINGS ----------
DAY_COUNT = 365.0

# ---------- HELPERS ----------
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

def main():
    spx = load_series(SPX_CSV, "date")[["date", "close"]].rename(columns={"close":"spx_close"})
    es = load_series(ES_CONTRACT_CSV, "date")[["date", "settle"]].rename(columns={"settle":"es_settle"})
    r3m = load_series(DGS3MO_CSV, "date")[["date", "value"]].rename(columns={"value":"r3m_pct"})
    div = load_series(DIV_YLD_CSV, "date")[["date", "value"]].rename(columns={"value":"div_pct"})

    last_es_date = es["date"].max()
    expiry_month = nearest_quarterly_month(last_es_date)
    expiry = third_friday(last_es_date.year, expiry_month)

    panel = es.merge(spx, on="date", how="left").merge(r3m, on="date", how="left").merge(div, on="date", how="left")
    panel = panel[panel["date"] <= expiry].dropna(subset=["spx_close", "es_settle", "r3m_pct", "div_pct"]).copy()

    # Realized SPX at expiry (use last trading day on/before expiry)
    spx_expiry_row = spx[spx["date"] <= expiry].sort_values("date").iloc[-1]
    spx_at_expiry = float(spx_expiry_row["spx_close"])
    realized_date = spx_expiry_row["date"]

    rows = []
    for _, row in panel.iterrows():
        asof = row["date"]
        S0 = float(row["spx_close"])
        r = float(row["r3m_pct"])
        d = float(row["div_pct"])
        es_px = float(row["es_settle"])
        T_years = (expiry - asof).days / DAY_COUNT
        if T_years <= 0:
            continue
        model_F = compute_forward_discrete(S0, r, d, T_years)
        model_err = abs(model_F - spx_at_expiry)
        fut_err = abs(es_px - spx_at_expiry)
        rows.append({
            "as_of_date": asof,
            "days_to_expiry": (expiry - asof).days,
            "S0_spx": S0,
            "r3m_pct": r,
            "div_pct": d,
            "model_forward_discrete": model_F,
            "futures_price": es_px,
            "realized_spx_at_expiry": spx_at_expiry,
            "model_abs_error_vs_realized": model_err,
            "futures_abs_error_vs_realized": fut_err,
            "futures_better": fut_err < model_err
        })

    out = pd.DataFrame(rows).sort_values("as_of_date")
    if out.empty:
        print("No overlapping data before expiry. Check your CSVs and ensure es_target.csv is a single contract series.")
        return

    def bucketize(dte):
        if dte <= 21: return "≤1M (~21d)"
        if dte <= 42: return "1–2M"
        if dte <= 63: return "2–3M"
        return ">3M"

    out["horizon_bucket"] = out["days_to_expiry"].apply(bucketize)

    summary = out.groupby("horizon_bucket").agg(
        n=("as_of_date", "count"),
        mae_model=("model_abs_error_vs_realized", "mean"),
        mae_futures=("futures_abs_error_vs_realized", "mean"),
    ).reset_index()

    overall = pd.DataFrame([{
        "horizon_bucket": "Overall",
        "n": out.shape[0],
        "mae_model": out["model_abs_error_vs_realized"].mean(),
        "mae_futures": out["futures_abs_error_vs_realized"].mean(),
    }])

    summary_full = pd.concat([summary, overall], ignore_index=True)

    # Save outputs next to this script (current working directory)
    per_day_path = "es_fair_value_vs_futures_backtest.csv"
    summary_path = "summary_stats.csv"
    err_plot_path = "error_by_days_to_expiry.png"
    cum_plot_path = "cumulative_mae_comparison.png"

    out.to_csv(per_day_path, index=False)
    summary_full.to_csv(summary_path, index=False)

    # Plot 1: abs error vs days-to-expiry
    plt.figure()
    out_sorted = out.sort_values("days_to_expiry")
    plt.plot(out_sorted["days_to_expiry"], out_sorted["model_abs_error_vs_realized"], label="Model abs error")
    plt.plot(out_sorted["days_to_expiry"], out_sorted["futures_abs_error_vs_realized"], label="Futures abs error")
    plt.gca().invert_xaxis()
    plt.title("Absolute Error vs. Days to Expiry (predicting SPX at expiry)")
    plt.xlabel("Days to expiry")
    plt.ylabel("Absolute error (index points)")
    plt.legend()
    plt.savefig(err_plot_path, bbox_inches="tight")
    plt.close()

    # Plot 2: cumulative mean abs error vs date
    plt.figure()
    out_sorted = out.sort_values("as_of_date")
    cum_mae_model = out_sorted["model_abs_error_vs_realized"].expanding().mean()
    cum_mae_fut = out_sorted["futures_abs_error_vs_realized"].expanding().mean()
    plt.plot(out_sorted["as_of_date"], cum_mae_model, label="Cumulative mean abs error (Model)")
    plt.plot(out_sorted["as_of_date"], cum_mae_fut, label="Cumulative mean abs error (Futures)")
    plt.title("Cumulative Mean Absolute Error vs. Date")
    plt.xlabel("As-of date")
    plt.ylabel("Cumulative MAE (index points)")
    plt.legend()
    plt.savefig(cum_plot_path, bbox_inches="tight")
    plt.close()

    print("\n== Per-day results (head) ==")
    print(out.head().to_string(index=False))
    print("\n== Summary stats ==")
    print(summary_full.to_string(index=False))
    print("\nSaved:")
    print(per_day_path)
    print(summary_path)
    print(err_plot_path)
    print(cum_plot_path)

if __name__ == "__main__":
    main()
