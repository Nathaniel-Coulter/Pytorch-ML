
#!/usr/bin/env python3
"""
Plots for SPX, ES futures, and carry-model fair value:
  1) levels_three_line.png           — SPX vs ES vs Model (levels)
  2) indexed_total_return.png        — all three indexed to 100 at first date
  3) scatter_es_vs_realized.png      — ES (as-of) vs realized SPX at expiry, with y=x and OLS fit
  4) scatter_model_vs_realized.png   — Model (as-of) vs realized SPX at expiry, with y=x and OLS fit
  5) basis_es_minus_model.png        — ES − Model through time
  6) basis_es_minus_spx.png          — ES − SPX through time
Also writes: levels_and_index.csv (data used).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

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

def ols_fit(x, y):
    # Simple least squares for y = a + b x
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    X = np.vstack([np.ones_like(x_arr), x_arr]).T
    beta = np.linalg.lstsq(X, y_arr, rcond=None)[0]
    a, b = beta[0], beta[1]
    return a, b

def main():
    spx = load_series(SPX_CSV, "date")[["date", "close"]].rename(columns={"close":"spx_close"})
    es = load_series(ES_CONTRACT_CSV, "date")[["date", "settle"]].rename(columns={"settle":"es_settle"})
    r3m = load_series(DGS3MO_CSV, "date")[["date", "value"]].rename(columns={"value":"r3m_pct"})
    div = load_series(DIV_YLD_CSV, "date")[["date", "value"]].rename(columns={"value":"div_pct"})

    last_es_date = es["date"].max()
    expiry_month = nearest_quarterly_month(last_es_date)
    expiry = third_friday(last_es_date.year, expiry_month)

    # Merge daily
    panel = (
        es.merge(spx, on="date", how="left")
          .merge(r3m, on="date", how="left")
          .merge(div, on="date", how="left")
    )
    panel = panel[panel["date"] <= expiry].dropna(subset=["spx_close", "es_settle", "r3m_pct", "div_pct"]).copy()

    # Compute model fair value each day
    panel["T_years"] = (pd.to_datetime(expiry) - pd.to_datetime(panel["date"])).dt.days / DAY_COUNT
    panel = panel[panel["T_years"] >= 0].copy()
    panel["model_forward_discrete"] = panel.apply(
        lambda r: compute_forward_discrete(r["spx_close"], r["r3m_pct"], r["div_pct"], r["T_years"]), axis=1
    )

    # Realized SPX at expiry (on/prior last trading day)
    spx_expiry_row = spx[spx["date"] <= expiry].sort_values("date").iloc[-1]
    spx_at_expiry = float(spx_expiry_row["spx_close"])

    # Augment with realized, errors, basis
    panel = panel.sort_values("date")
    panel["realized_spx_at_expiry"] = spx_at_expiry
    panel["basis_es_minus_model"] = panel["es_settle"] - panel["model_forward_discrete"]
    panel["basis_es_minus_spx"] = panel["es_settle"] - panel["spx_close"]

    out_levels = panel[["date","spx_close","es_settle","model_forward_discrete","realized_spx_at_expiry",
                        "basis_es_minus_model","basis_es_minus_spx"]].copy()
    out_levels.to_csv("levels_and_index.csv", index=False)

    # Plot 1: Three-line levels
    plt.figure()
    plt.plot(out_levels["date"], out_levels["spx_close"], label="SPX (close)")
    plt.plot(out_levels["date"], out_levels["es_settle"], label="ES futures (settle)")
    plt.plot(out_levels["date"], out_levels["model_forward_discrete"], label="Model fair value")
    plt.title("SPX vs ES Futures vs Model (Levels)")
    plt.xlabel("Date")
    plt.ylabel("Index / Price level")
    plt.legend()
    plt.savefig("levels_three_line.png", bbox_inches="tight")
    plt.close()

    # Plot 2: Indexed (normalize to 100 at first date)
    base = out_levels.iloc[0]
    idx_df = out_levels.copy()
    idx_df["SPX_idx"] = out_levels["spx_close"] / base["spx_close"] * 100.0
    idx_df["ES_idx"] = out_levels["es_settle"] / base["es_settle"] * 100.0
    idx_df["Model_idx"] = out_levels["model_forward_discrete"] / base["model_forward_discrete"] * 100.0

    plt.figure()
    plt.plot(idx_df["date"], idx_df["SPX_idx"], label="SPX (indexed to 100)")
    plt.plot(idx_df["date"], idx_df["ES_idx"], label="ES (indexed to 100)")
    plt.plot(idx_df["date"], idx_df["Model_idx"], label="Model (indexed to 100)")
    plt.title("Indexed Paths (Total Return-style, No Dividends)")
    plt.xlabel("Date")
    plt.ylabel("Index (100 = first date)")
    plt.legend()
    plt.savefig("indexed_total_return.png", bbox_inches="tight")
    plt.close()

    # Plot 3: Scatter ES vs realized with y=x and OLS eqn
    x = out_levels["es_settle"]
    y = out_levels["realized_spx_at_expiry"]
    a, b = ols_fit(x, y)  # y = a + b x
    xline = np.linspace(x.min(), x.max(), 100)
    yline = a + b * xline

    plt.figure()
    plt.scatter(x, y, s=10, label="As-of observations")
    plt.plot(xline, xline, linestyle="-", label="y = x")
    plt.plot(xline, yline, linestyle="--", label=f"OLS: y = {a:.2f} + {b:.4f} x")
    plt.title("ES (as-of) vs Realized SPX at Expiry")
    plt.xlabel("ES futures price (as-of date)")
    plt.ylabel("Realized SPX at expiry")
    plt.legend()
    plt.savefig("scatter_es_vs_realized.png", bbox_inches="tight")
    plt.close()

    # Plot 4: Scatter Model vs realized with y=x and OLS eqn
    x2 = out_levels["model_forward_discrete"]
    y2 = out_levels["realized_spx_at_expiry"]
    a2, b2 = ols_fit(x2, y2)
    xline2 = np.linspace(x2.min(), x2.max(), 100)
    yline2 = a2 + b2 * xline2

    plt.figure()
    plt.scatter(x2, y2, s=10, label="As-of observations")
    plt.plot(xline2, xline2, linestyle="-", label="y = x")
    plt.plot(xline2, yline2, linestyle="--", label=f"OLS: y = {a2:.2f} + {b2:.4f} x")
    plt.title("Model (as-of) vs Realized SPX at Expiry")
    plt.xlabel("Model fair value (as-of date)")
    plt.ylabel("Realized SPX at expiry")
    plt.legend()
    plt.savefig("scatter_model_vs_realized.png", bbox_inches="tight")
    plt.close()

    # Plot 5: Basis ES - Model through time
    plt.figure()
    plt.plot(out_levels["date"], out_levels["basis_es_minus_model"], label="ES − Model")
    plt.axhline(0.0)
    plt.title("Basis: ES − Model Fair Value")
    plt.xlabel("Date")
    plt.ylabel("Index points")
    plt.legend()
    plt.savefig("basis_es_minus_model.png", bbox_inches="tight")
    plt.close()

    # Plot 6: Basis ES - SPX through time
    plt.figure()
    plt.plot(out_levels["date"], out_levels["basis_es_minus_spx"], label="ES − SPX spot")
    plt.axhline(0.0)
    plt.title("Basis: ES − SPX Spot")
    plt.xlabel("Date")
    plt.ylabel("Index points")
    plt.legend()
    plt.savefig("basis_es_minus_spx.png", bbox_inches="tight")
    plt.close()

    print("Wrote files:")
    print("  - levels_and_index.csv")
    print("  - levels_three_line.png")
    print("  - indexed_total_return.png")
    print("  - scatter_es_vs_realized.png")
    print("  - scatter_model_vs_realized.png")
    print("  - basis_es_minus_model.png")
    print("  - basis_es_minus_spx.png")

if __name__ == "__main__":
    main()
