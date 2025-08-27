#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent

def _read_datesmart_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # find a date-like column
    date_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("date", "dt", "timestamp", "observation_date"):
            date_col = c; break
    if date_col is None:
        # fallback: try first column
        cand = df.columns[0]
        try:
            pd.to_datetime(df[cand], errors="raise")
            date_col = cand
        except Exception:
            pass
    if date_col is None:
        raise ValueError(f"Could not detect a date column in {path}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    return df

def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    # Resample to weekly Friday, using last available in week
    return df.resample("W-FRI").last()

def build_features(yc1: Path, yc2: Path, out: Path) -> Path:
    # Load the two sources
    a = _read_datesmart_csv(yc1)
    b = _read_datesmart_csv(yc2)

    # Lowercase + strip columns for easier matching
    a.columns = [c.strip() for c in a.columns]
    b.columns = [c.strip() for c in b.columns]

    # Try to find tenor columns in each source by common substrings
    def find_col(cols, needles):
        clmap = {c: c.lower() for c in cols}
        for c, cl in clmap.items():
            if any(n in cl for n in needles):
                return c
        return None

    # Common tenors we care about
    c_3m_a = find_col(a.columns, ["3m","3 m","3-month","tbill","t-bill"])
    c_2y_a = find_col(a.columns, ["2y","2 y","2-year"])
    c_5y_a = find_col(a.columns, ["5y","5 y","5-year"])
    c_10y_a= find_col(a.columns,["10y","10 y","10-year","gs10","dgs10"])

    c_3m_b = find_col(b.columns, ["3m","3 m","3-month","tbill","t-bill"])
    c_2y_b = find_col(b.columns, ["2y","2 y","2-year"])
    c_5y_b = find_col(b.columns, ["5y","5 y","5-year"])
    c_10y_b= find_col(b.columns,["10y","10 y","10-year","gs10","dgs10"])

    def pick(a_col, b_col):
        # Prefer non-null coverage
        if a_col and a_col in a.columns and a[a_col].notna().sum() > 10:
            return ("a", a_col)
        if b_col and b_col in b.columns and b[b_col].notna().sum() > 10:
            return ("b", b_col)
        return (None, None)

    src_3m, col_3m = pick(c_3m_a, c_3m_b)
    src_2y, col_2y = pick(c_2y_a, c_2y_b)
    src_5y, col_5y = pick(c_5y_a, c_5y_b)
    src_10y,col_10y= pick(c_10y_a,c_10y_b)

    if not col_10y:
        raise ValueError("Could not find a usable 10Y series in either file.")
    if not (col_3m or col_2y):
        raise ValueError("Could not find a usable 3M or 2Y series in either file.")
    if not col_5y:
        raise ValueError("Could not find a usable 5Y series in either file.")

    def series(src, col):
        return (a if src=="a" else b)[col].astype(float)

    # Build a combined DataFrame on the union index
    df = pd.DataFrame(index=a.index.union(b.index))
    if col_10y: df["y10"] = series(src_10y, col_10y)
    if col_5y:  df["y5"]  = series(src_5y, col_5y)
    if col_2y:  df["y2"]  = series(src_2y, col_2y) if col_2y else np.nan
    if col_3m:  df["y3m"] = series(src_3m, col_3m) if col_3m else np.nan

    # Weekly Friday freq
    dfw = _to_weekly(df).sort_index()

    # Prefer slope 10Y-3M; fallback to 10Y-2Y if 3M missing
    dfw["lvl"] = dfw["y10"]
    dfw["slope"] = np.where(dfw["y3m"].notna(),
                            dfw["y10"] - dfw["y3m"],
                            dfw["y10"] - dfw["y2"])
    # Curvature: 2*5y - 2y - 10y; if 2y missing, fallback to 3m
    dfw["curv"] = 2*dfw["y5"] - np.where(dfw["y2"].notna(), dfw["y2"], dfw["y3m"]) - dfw["y10"]

    # Lag by 1 week to avoid look-ahead
    dfw[["lvl","slope","curv"]] = dfw[["lvl","slope","curv"]].shift(1)

    # Keep only the features and drop rows where all three are NaN
    out_df = dfw[["lvl","slope","curv"]].dropna(how="all")
    out_df = out_df.reset_index().rename(columns={"index":"Date"})
    out_df.to_csv(out, index=False)
    print(f"Wrote {out} with columns: {list(out_df.columns)} and {len(out_df)} rows")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yc1", type=str, default=str(ROOT/"data/yield-curve-rates-1990-2021.csv"))
    ap.add_argument("--yc2", type=str, default=str(ROOT/"data/1.5.15.20YR_Monthly_Mats_2006-2024.csv"))
    ap.add_argument("--out", type=str, default=str(ROOT/"figures/yield_features_weekly.csv"))
    args = ap.parse_args()
    build_features(Path(args.yc1), Path(args.yc2), Path(args.out))

if __name__ == "__main__":
    main()
