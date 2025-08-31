#!/usr/bin/env python3
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT/"outputs", ROOT/"figures"
ASSETS = ["SPY","GLD","TLT"]
ROLL_WIN = 156  # ~3y weekly window for static MVO

def load_weekly_rets():
    # pull adjusted prices
    px = pd.read_parquet(OUT/"prices.parquet").xs("Adj Close", level=1, axis=1)

    # HARDEN: force numeric; anything weird -> NaN
    px = px.apply(pd.to_numeric, errors="coerce")

    # business daily -> weekly (Fri)
    pxw = px.asfreq("B").ffill().resample("W-FRI").last()

    # drop rows where *all* assets are NaN (start-up periods)
    pxw = pxw.dropna(how="all")

    # log returns
    r = np.log(pxw / pxw.shift(1))

    # keep the trio used in this baseline
    r = r[ASSETS].dropna(how="all")

    # (optional) if any column still has all-NaN in the window, drop it
    r = r.dropna(axis=1, how="all")

    return r

def ann_metrics(r):
    ann = r.mean()*52
    vol = r.std(ddof=0)*np.sqrt(52)
    shp = ann/vol if vol>0 else np.nan
    cw  = np.exp(r.cumsum())
    dd  = (cw/cw.cummax()-1).min()
    return float(ann), float(vol), float(shp), float(dd)

def tangency_w(mu, Sigma, long_only=True):
    # classic mean-variance with no risk-free; scale to sum=1
    try:
        inv = np.linalg.pinv(Sigma)
        w = inv @ mu
        w = np.maximum(w, 0) if long_only else w
        s = w.sum()
        return w/s if s>0 else np.ones_like(w)/len(w)
    except Exception:
        return np.ones_like(mu)/len(mu)

def run_static_mvo(R):
    rets, ws = [], []
    idx = R.index
    for t in range(ROLL_WIN, len(idx)):
        win = R.iloc[t-ROLL_WIN:t]
        mu  = win.mean().values
        Sig = win.cov().values
        w   = tangency_w(mu, Sig, long_only=True)
        ws.append(pd.Series(w, index=R.columns, name=idx[t]))
        rets.append((R.iloc[t] * w).sum())
    return pd.Series(rets, index=idx[ROLL_WIN:]), pd.DataFrame(ws)

def run_mgarch_mvo(R):
    F = pd.read_parquet(OUT/"mgarch_features.parquet")
    F = F.reindex(R.index).ffill()
    pairs = [("SPY","GLD"),("SPY","TLT"),("GLD","TLT")]
    pair2rho = {("SPY","GLD"): "rho_SPY_GLD", ("SPY","TLT"): "rho_SPY_TLT", ("GLD","TLT"): "rho_GLD_TLT"}

    rets, ws = [], []
    idx = R.index
    for t in range(ROLL_WIN, len(idx)):
        row = F.iloc[t]
        sig = np.array([row["sigma_SPY"], row["sigma_GLD"], row["sigma_TLT"]], dtype=float)
        if not np.all(np.isfinite(sig)):
            # fallback to static cov at this date
            win = R.iloc[t-ROLL_WIN:t]
            mu  = win.mean().values
            Sig = win.cov().values
        else:
            rho = np.eye(3)
            # fill symmetric correlations
            vals = {}
            for (a,b), key in pair2rho.items():
                vals[(a,b)] = row.get(key, np.nan)
            # if any rho missing, fallback to rolling corr
            if not np.all([np.isfinite(v) for v in vals.values()]):
                win = R.iloc[t-ROLL_WIN:t]
                Sig = win.cov().values
                mu  = win.mean().values
                w   = tangency_w(mu, Sig, long_only=True)
                ws.append(pd.Series(w, index=R.columns, name=idx[t]))
                rets.append((R.iloc[t] * w).sum())
                continue
            order = ["SPY","GLD","TLT"]
            lut = {( "SPY","GLD"): vals[("SPY","GLD")],
                   ( "SPY","TLT"): vals[("SPY","TLT")],
                   ( "GLD","TLT"): vals[("GLD","TLT")]}
            for i,a in enumerate(order):
                for j,b in enumerate(order):
                    if i<j:
                        r = lut.get((a,b), np.nan)
                        rho[i,j]=rho[j,i]=r
            Sig = np.outer(sig, sig) * rho
            win = R.iloc[t-ROLL_WIN:t]
            mu  = win.mean().values  # use rolling mean for expected return
        w   = tangency_w(mu, Sig, long_only=True)
        ws.append(pd.Series(w, index=R.columns, name=idx[t]))
        rets.append((R.iloc[t] * w).sum())
    return pd.Series(rets, index=idx[ROLL_WIN:]), pd.DataFrame(ws)

def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    R = load_weekly_rets()

    r_static, w_static = run_static_mvo(R)
    r_mgarch, w_mgarch = run_mgarch_mvo(R)

    rows=[]
    for name, r in [("Static MVO", r_static), ("MGARCH-MVO", r_mgarch)]:
        ann, vol, shp, mdd = ann_metrics(r)
        rows.append(dict(Model=name, AnnRet=ann, Vol=vol, Sharpe=shp, MaxDD=mdd))
    dfm = pd.DataFrame(rows).set_index("Model")
    dfm.to_csv(OUT/"sec6_mvo_metrics.csv")

    # equity curves
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,5))
    for name, r in [("Static MVO", r_static), ("MGARCH-MVO", r_mgarch)]:
        cw = np.exp(r.cumsum())
        plt.plot(cw.index, cw.values, label=name)
    plt.title("§6 MVO vs MGARCH-MVO — Cumulative wealth")
    plt.xlabel("Date"); plt.ylabel("Cumulative wealth"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG/"sec6_mvo_cumret.png", dpi=150)

    print("[ok] wrote:",
          OUT/"sec6_mvo_metrics.csv",
          FIG/"sec6_mvo_cumret.png")

if __name__ == "__main__":
    main()
