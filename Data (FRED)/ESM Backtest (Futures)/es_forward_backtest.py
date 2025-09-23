#!/usr/bin/env python3
"""
ES Forward-Price Backtest (Discrete Compounding, Costs & Benefits)

What it does
------------
For a chosen ES quarterly contract (e.g., September 2025, "ESU25"), this script:
1) Picks an "as_of_date" (e.g., 2025-09-01) and finds the contract's expiry date (third Friday of Mar/Jun/Sep/Dec).
2) For lookback horizons of 1, 2, and 3 months before the as_of_date, it:
   - Reads S&P 500 spot level (S0) on the lookback date
   - Reads short risk-free rate r (e.g., FRED 3M T-bill) on that date (annualized, in percent)
   - Reads dividend yield d (annualized, in percent) on that date
   - Computes time-to-maturity T (years) from the lookback date to the contract expiry (ACT/365)
   - Computes model-implied forward price using the discrete-comp "with costs & benefits" formula:
       F = S0 * (1 + r - d)^T, where r and d are in decimals (not percent)
3) Compares each model-implied forward to the ACTUAL ES contract settlement/close on the as_of_date.

Inputs expected
---------------
Provide the following CSVs (you can change the file paths below):
- spx.csv            with columns: date, close
- es_target.csv      with columns: date, settle   # this should be the specific contract (e.g., ESU2025) settlements
- fred_dgs3mo.csv    with columns: date, value    # 3M T-bill yield in percent
- spx_div_yield.csv  with columns: date, value    # S&P 500 dividend yield in percent (approx; SPY-based acceptable)

All dates should be YYYY-MM-DD (ISO). Files can have additional columns; only the named ones are used.

Usage
-----
Edit the CONFIG section (paths, as_of_date, contract_code). Then run:

    python es_forward_backtest.py

Outputs
-------
- A table printed to stdout with model-implied vs. actual prices.
- A CSV "es_forward_backtest_output.csv" with the same table.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# ----------------------------
# CONFIG
# ----------------------------
SPX_CSV = "spx.csv"                 # spot index (S&P 500)
ES_CONTRACT_CSV = "es_target.csv"   # target ES contract settlements (e.g., ESU2025)
DGS3MO_CSV = "fred_dgs3mo.csv"      # 3M T-bill yield (percent)
DIV_YLD_CSV = "spx_div_yield.csv"   # S&P dividend yield (percent)

# Analysis choices
AS_OF_DATE = "2025-09-01"           # the date we compare model vs actual on the target contract
CONTRACT_CODE = "ESU2025"           # used only as a label in outputs
LOOKBACK_MONTHS = [1, 2, 3]         # compute model price 1,2,3 months before AS_OF_DATE

# Day-count convention for T (years)
DAY_COUNT = 365.0                   # ACT/365F

# If you prefer to override dividend yield with a fixed value, set OVERRIDE_D = None or a float in percent
OVERRIDE_D = None                   # e.g., 1.5 for 1.5%

# ----------------------------
# Helpers
# ----------------------------
def parse_date(s):
    return pd.to_datetime(s).date()

def third_friday(year, month):
    """Return the date of the third Friday for given year, month."""
    d = datetime(year, month, 1).date()
    # find first Friday
    while d.weekday() != 4:  # 0=Mon,...4=Fri
        d += timedelta(days=1)
    # third Friday = first Friday + 14 days
    d += timedelta(days=14)
    return d

def quarterly_expiry_for_date(year, month):
    """
    For a given month, return the expiry for the nearest standard ES quarterly contract in that month.
    Standard ES expiries are Mar (3), Jun (6), Sep (9), Dec (12) on the third Friday.
    """
    if month not in (3, 6, 9, 12):
        raise ValueError("This helper expects you already chose a quarterly month (Mar, Jun, Sep, Dec).")
    return third_friday(year, month)

def nearest_quarterly_month(date_obj):
    """Return the quarterly month that 'date_obj' belongs to if you intend the contract that expires in that quarter.
    Typically you'd pick the contract in the same quarter as AS_OF_DATE (e.g., Sep for dates in Jul-Sep)."""
    m = ((date_obj.month-1)//3 + 1)*3  # 3,6,9,12
    return m if m in (3,6,9,12) else 12

def load_series(path, date_col="date"):
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    return df

def align_on_or_prior(df, date_col, target_date):
    """Return the row with the last available observation on or prior to target_date."""
    sub = df[df[date_col] <= target_date]
    if sub.empty:
        raise ValueError(f"No data on or before {target_date} in {date_col}.")
    return sub.sort_values(date_col).iloc[-1]

def compute_forward_discrete(S0, r_pct, d_pct, T_years):
    """F = S0 * (1 + r - d)^T, with r,d as DECIMALS (r_pct/d_pct given in PERCENT)."""
    r = (r_pct or 0.0)/100.0
    d = (d_pct or 0.0)/100.0
    return S0 * ((1.0 + r - d) ** T_years)

# ----------------------------
# Main
# ----------------------------
def main():
    as_of = parse_date(AS_OF_DATE)

    # Determine the target contract expiry (assume same-quarter contract for AS_OF_DATE)
    q_month = nearest_quarterly_month(as_of)
    expiry = quarterly_expiry_for_date(as_of.year, q_month)

    # Load data
    spx = load_series(SPX_CSV, "date")[["date", "close"]].rename(columns={"close":"spx_close"})
    es = load_series(ES_CONTRACT_CSV, "date")[["date", "settle"]].rename(columns={"settle":"es_settle"})
    r3m = load_series(DGS3MO_CSV, "date")[["date", "value"]].rename(columns={"value":"r3m_pct"})
    div = load_series(DIV_YLD_CSV, "date")[["date", "value"]].rename(columns={"value":"div_pct"})

    # Actual ES settlement on as_of_date (or last available prior)
    es_row = align_on_or_prior(es, "date", as_of)
    actual_es = float(es_row["es_settle"])
    actual_date = es_row["date"]

    rows = []
    for m in LOOKBACK_MONTHS:
        lookback_date = (as_of - relativedelta(months=m))
        # fetch spot, rate, dividend on/prior to lookback_date
        s_row = align_on_or_prior(spx, "date", lookback_date)
        r_row = align_on_or_prior(r3m, "date", lookback_date)
        if OVERRIDE_D is None:
            d_row = align_on_or_prior(div, "date", lookback_date)
            d_pct = float(d_row["div_pct"])
            d_src = "DIV_SERIES"
        else:
            d_pct = float(OVERRIDE_D)
            d_src = "DIV_FIXED"

        S0 = float(s_row["spx_close"])
        r_pct = float(r_row["r3m_pct"])

        # T in years from lookback_date to expiry (ACT/365)
        T_years = (expiry - s_row["date"]).days / DAY_COUNT
        if T_years <= 0:
            # if lookback goes beyond expiry, skip
            continue

        model_F = compute_forward_discrete(S0, r_pct, d_pct, T_years)
        basis_pts = (actual_es - model_F)
        basis_pct = (basis_pts / actual_es) * 100.0

        rows.append({
            "contract": CONTRACT_CODE,
            "expiry": expiry.isoformat(),
            "as_of_date": as_of.isoformat(),
            "actual_es_date": actual_date.isoformat(),
            "actual_es_settle": round(actual_es, 2),
            "lookback_months": m,
            "lookback_obs_date": s_row["date"].isoformat(),
            "S0_spx": round(S0, 2),
            "r3m_pct": round(r_pct, 4),
            "div_pct": round(d_pct, 4),
            "T_years": round(T_years, 6),
            "model_forward_discrete": round(model_F, 2),
            "basis_pts_actual_minus_model": round(basis_pts, 2),
            "basis_pct_actual_minus_model": round(basis_pct, 4),
            "div_source": d_src,
        })

    out = pd.DataFrame(rows).sort_values("lookback_months")
    if out.empty:
        print("No rows produced. Check inputs and dates.")
        return

    print(out.to_string(index=False))

    out.to_csv("es_forward_backtest_output.csv", index=False)
    print("\nSaved: es_forward_backtest_output.csv")

if __name__ == "__main__":
    main()
