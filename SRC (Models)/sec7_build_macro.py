#!/usr/bin/env python3
"""
§7 macro panels (weekly W-FRI), RAW features only (no scaling):
- vol/credit panel: VVIX, GVZ, OVX, SKEW, MOVE, CDX_IG, SPX/NDX IVOL + OVDV, RVOL,
  SPX_VRP = IVOL - RVOL (if present), optional term slope if 1M/3M exist.
- mgarch panel: sigma_* and rho_* from §6 (outputs/mgarch_features.parquet)

Why RAW? We will lag/standardize by train fold inside the run script to avoid leakage.
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# §6 artifact
F = pd.read_parquet(OUT / "mgarch_features.parquet").sort_index()
F.index = pd.DatetimeIndex(F.index)

def keep(cols, obj):
    return [c for c in cols if c in obj.columns]

# ---------- vol/credit (RAW) ----------
want_vc = [
    "VVIX","GVZ","OVX","SKEW","MOVE","CDX_IG",
    "SPX_IVOL","NDX_IVOL","SPX_OVDV","NDX_OVDV","RVOL",
]
vc = F[keep(want_vc, F)].copy()

# SPX volatility risk premium if both present
if {"SPX_IVOL","RVOL"}.issubset(vc.columns):
    vc["SPX_VRP"] = vc["SPX_IVOL"] - vc["RVOL"]

# optional term slope if OVDV sheet carries multiple tenors (auto-detect)
tenor_cols = [c for c in vc.columns if "OVDV" in c]
if tenor_cols:
    one_m = [c for c in tenor_cols if "1" in c.upper() and "M" in c.upper()]
    three_m = [c for c in tenor_cols if "3" in c.upper() and "M" in c.upper()]
    if one_m and three_m:
        try:
            vc["SPX_term_slope"] = vc[one_m[0]] - vc[three_m[0]]
        except Exception:
            pass

vc = vc.asfreq("W-FRI").ffill().dropna(how="all")
vc.to_parquet(OUT/"sec7_macro_volcredit_raw.parquet")

# ---------- mgarch subset (RAW) ----------
mg = F[keep(["sigma_SPY","sigma_GLD","sigma_TLT",
             "rho_SPY_GLD","rho_SPY_TLT","rho_GLD_TLT"], F)].copy()
mg = mg.asfreq("W-FRI").ffill().dropna(how="all")
mg.to_parquet(OUT/"sec7_macro_mgarch_raw.parquet")

print("[ok] wrote:",
      OUT/"sec7_macro_volcredit_raw.parquet",
      OUT/"sec7_macro_mgarch_raw.parquet")
