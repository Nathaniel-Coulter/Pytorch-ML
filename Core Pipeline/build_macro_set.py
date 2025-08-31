#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np

NUMERIC_GUESS = ("PX_LAST","Close","VALUE","Value","Index","Last","Last Price","level","Level")

def load_one(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # Find date col
    date_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("date","dt","timestamp","asof","as of","obs_date"):
            date_col = c; break
    if date_col is None: raise ValueError(f"{p}: no date column")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    # Pick a numeric column
    candidates = [c for c in df.columns if c!=date_col]
    pick = None
    for k in NUMERIC_GUESS:
        for c in candidates:
            if k.lower() in str(c).lower():
                pick = c; break
        if pick: break
    if pick is None:
        # last numeric
        for c in reversed(candidates):
            if pd.api.types.is_numeric_dtype(df[c]):
                pick = c; break
    if pick is None: raise ValueError(f"{p}: no numeric column found")
    s = df[[date_col, pick]].rename(columns={date_col:"Date", pick: "val"}).set_index("Date")["val"]
    # weekly Friday, ffill within week
    s = s.asfreq("B").ffill().resample("W-FRI").last()
    # name from filename
    name = re.sub(r"[^A-Za-z0-9]+","_", p.stem).lower()
    return s.rename(name).to_frame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="e.g. 'data/bbg/*.csv'")
    ap.add_argument("--yc", default="figures/yield_features_weekly.csv", help="existing yield-curve csv with Date,lvl,slope,curv")
    ap.add_argument("--out", default="figures/macro_features_weekly.csv")
    args = ap.parse_args()
    files = [Path(p) for p in sorted(Path().glob(args.glob))]
    if not files: raise SystemExit("No files matched.")
    wide = None
    for p in files:
        df = load_one(p)
        wide = df if wide is None else wide.join(df, how="outer")
    # attach yield curve if present (expects Date,lvl,slope,curv)
    yc = Path(args.yc)
    if yc.exists():
        dy = pd.read_csv(yc, parse_dates=["Date"]).set_index("Date")[["lvl","slope","curv"]]
        dy = dy.resample("W-FRI").last().ffill()
        wide = wide.join(dy, how="outer")
    wide = wide.sort_index().ffill(limit=8)  # ~2 months cap
    wide.to_csv(args.out, index=True)
    print(f"[ok] wrote {args.out} with {wide.shape[0]} rows × {wide.shape[1]} cols")
if __name__ == "__main__":
    main()
