# Write a new script that creates a 3-line levels plot (SPX, ES, Model)
# and a "total return-style" indexed plot (each series indexed to 100 at the first date).
from pathlib import Path
from textwrap import dedent

script = dedent("""
#!/usr/bin/env python3
\"\"\"
Three-line plot (levels) and indexed 'total return-style' plot for SPX, ES futures, and carry-model fair value.

Inputs (edit if needed):
  - SPX_CSV: date, close
  - ES_CONTRACT_CSV: date, settle   (single quarterly ES contract)
  - DGS3MO_CSV: date, value         (3M T-bill %, daily or forward-filled)
  - DIV_YLD_CSV: date, value        (dividend yield %, daily or forward-filled)

Outputs (current working directory):
  - levels_three_line.png
  - indexed_total_return.png
  - levels_and_index.csv
\"\"\"
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

    # Save levels to CSV
    out_levels = panel[["date","spx_close","es_settle","model_forward_discrete"]].copy()
    out_levels.sort_values("date", inplace=True)
    out_levels.to_csv("levels_and_index.csv", index=False)

    # Plot 1: Three-line levels plot
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

    # Plot 2: Indexed 'total return-style' (normalize to 100 at first date)
    base = out_levels.iloc[0]
    idx_df = out_levels.copy()
    idx_df["SPX_idx"] = out_levels["spx_close"] / base["spx_close"] * 100.0
    idx_df["ES_idx"] = out_levels["es_settle"] / base["es_settle"] * 100.0
    idx_df["Model_idx"] = out_levels["model_forward_discrete"] / base["model_forward_discrete"] * 100.0

    plt.figure()
    plt.plot(idx_df["date"], idx_df["SPX_idx"], label="SPX (indexed to 100)")
    plt.plot(idx_df["date"], idx_df["ES_idx"], label="ES futures (indexed to 100)")
    plt.plot(idx_df["date"], idx_df["Model_idx"], label="Model fair value (indexed to 100)")
    plt.title("Indexed Paths (Total Return-style, No Dividends)")
    plt.xlabel("Date")
    plt.ylabel("Index (100 = first date)")
    plt.legend()
    plt.savefig("indexed_total_return.png", bbox_inches="tight")
    plt.close()

    print("Wrote: levels_and_index.csv, levels_three_line.png, indexed_total_return.png")

if __name__ == "__main__":
    main()
""")

out_path = Path("/mnt/data/three_line_and_index_plots.py")
out_path.write_text(script, encoding="utf-8")
out_path
