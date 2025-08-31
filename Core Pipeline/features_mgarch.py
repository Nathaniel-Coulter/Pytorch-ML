#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT  = ROOT / "outputs"
FIG  = ROOT / "figures"
DATA = ROOT / "data" / "Bloomberg"

# --- safe import arch; fallback to EWMA if missing ---
try:
    from arch.univariate import ConstantMean, GARCH, Normal
    HAVE_ARCH = True
except Exception:
    HAVE_ARCH = False

ASSET_TICKERS = ["SPY","GLD","TLT"]  # core set for MGARCH baseline
ROLL_W = 13  # ~quarterly (weekly freq), for rolling correlations

# Bloomberg series (weekly, we'll resample + log change)
BLOOMBERG_SERIES = {
    "VVIX":"VVIX INDEX.csv",
    "GVZ":"GVZ Index.csv",
    "OVX":"OVX (INDEX).csv",
    "SKEW":"SKEW Index.csv",
    "MOVE":"MOVE Index.csv",
    "CDX_IG":"CDX IG CDSI GEN 5Y.csv",
    "CDX_HY":"CDX HY CDSI GEN 5Y.csv",
    # Optional IVOL/OVDV files (if you want them in the panel too):
    "SPX_IVOL":"SPX IVM_mid(Worksheet).csv",
    "SPX_OVDV":"spx_ovdv.csv",
    "NDX_IVOL":"NDX IVOL_MID - OVDV(NDX IVOL_MID).csv",
    "NDX_OVDV":"ndx_ovdv.csv",
    "RVOL":"RVOL (HVG).csv",
}

def read_prices_weekly() -> pd.DataFrame:
    # prices.parquet from your loader (Daily multiindex -> weekly returns)
    px_all = pd.read_parquet(OUT / "prices.parquet")
    px = px_all.xs("Adj Close", level=1, axis=1)

    # --- HARDENING: force numeric, coerce junk to NaN ---
    px = px.apply(pd.to_numeric, errors="coerce")

    # Business daily -> W-FRI
    pxw = px.asfreq("B").ffill().resample("W-FRI").last()

    # Drop rows where ALL assets are NaN (e.g., initial dates)
    pxw = pxw.dropna(how="all")

    # Weekly log-returns
    rets = np.log(pxw / pxw.shift(1))
    return rets

def fit_garch_series(r: pd.Series) -> pd.Series:
    r = r.dropna()
    if len(r) < 100:
        return r.reindex_like(r) * np.nan
    if HAVE_ARCH:
        am = ConstantMean(r*100.0)  # scale to pct to help optimizer stability
        am.volatility = GARCH(1, 0, 1)
        am.distribution = Normal()
        res = am.fit(disp="off")
        cond_vol = res.conditional_volatility / 100.0
        # align back to original weekly index
        return cond_vol.reindex(r.index)
    else:
        # EWMA fallback (lambda=0.94)
        lam = 0.94
        x = r.values
        s2 = np.empty_like(x)
        s2[:] = np.nan
        var = np.nan
        for i, xi in enumerate(x):
            if not np.isfinite(xi): 
                continue
            var = (lam * (var if np.isfinite(var) else xi*xi) +
                   (1-lam) * xi*xi)
            s2[i] = var
        out = pd.Series(np.sqrt(s2), index=r.index)
        return out

def build_mgarch_features():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    rets = read_prices_weekly()
    rets = rets[ASSET_TICKERS]

    # Conditional vols via per-asset GARCH(1,1) (or EWMA fallback)
    sigma = {}
    for t in ASSET_TICKERS:
        sigma[t] = fit_garch_series(rets[t])
    sigma_df = pd.DataFrame({f"sigma_{t}": v for t, v in sigma.items()})

    # Rolling correlations as DCC proxy
    rho_cols = {}
    for i, a in enumerate(ASSET_TICKERS):
        for b in ASSET_TICKERS[i+1:]:
            col = f"rho_{a}_{b}"
            rho_cols[col] = rets[a].rolling(ROLL_W).corr(rets[b])

    rho_df = pd.DataFrame(rho_cols)

    # Also include Bloomberg vol/credit factors as weekly log changes
    bb = {}
    for name, fn in BLOOMBERG_SERIES.items():
        fp = DATA / fn
        if not fp.exists():
            continue
        df = pd.read_csv(fp)

        # Find/parse the date column
        date_col = None
        for c in df.columns:
            if str(c).lower() in ("date", "dt", "timestamp"):
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]  # fallback: assume first col is date
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

        # Try to coerce all non-date columns to numeric (some come as strings)
        data_cols = [c for c in df.columns if c != date_col]
        for c in data_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Pick the first column with enough non-nan data
        cand = None
        for c in data_cols:
            if df[c].notna().sum() > 5:  # arbitrary small threshold
                cand = c
                break
        if cand is None:
            # nothing usable in this file
            continue

        s = df[cand].rename(name)

        # --- HARDENING: remove duplicate timestamps & sort index ---
        s = s[~s.index.duplicated(keep="last")].sort_index()

        # Resample directly to weekly (W-FRI) and forward-fill
        sw = s.resample("W-FRI").last().ffill()

        # Weekly log-change (levels can be index-like; change is more stationary)
        bb[name] = np.log(sw / sw.shift(1))

    bb_df = pd.DataFrame(bb)

    # Combine all
    feats = pd.concat([sigma_df, rho_df, bb_df], axis=1)
    feats = feats.loc[rets.index]  # align to core weekly index
    feats.to_parquet(OUT / "mgarch_features.parquet")
    print("[ok] wrote outputs/mgarch_features.parquet with", feats.shape, "shape")

if __name__ == "__main__":
    build_mgarch_features()
