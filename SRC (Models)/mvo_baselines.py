#!/usr/bin/env python3
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"
FIG  = ROOT / "figures"

ASSET_PORT = ["SPY","QQQ","IWM","EFA","EEM","TLT","IEF","LQD","HYG","GLD","DBC","VNQ"]
CORE3 = ["SPY","GLD","TLT"]
ROLL_WIN = 156  # ~3y weekly

def load_weekly_returns():
    px = pd.read_parquet(OUT/"prices.parquet").xs("Adj Close", level=1, axis=1)
    px = px.apply(pd.to_numeric, errors="coerce")
    pxw = px.asfreq("B").ffill().resample("W-FRI").last()
    pxw = pxw.dropna(how="all")
    r = np.log(pxw / pxw.shift(1))
    return r

def tangency_weights(mu, Sigma, long_only=True, ridge=1e-6):
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(Sigma)):
        return np.ones_like(mu)/len(mu)
    k = Sigma.shape[0]
    Sigma = Sigma + ridge*np.eye(k)  # regularize
    try:
        inv = np.linalg.pinv(Sigma)
        w = inv @ mu
        if long_only:
            w = np.maximum(w, 0.0)
        s = w.sum()
        return (w/s) if s>0 else np.ones_like(mu)/len(mu)
    except Exception:
        return np.ones_like(mu)/len(mu)

def run_static_mvo(R):
    rets, ws = [], []
    idx, cols = R.index, R.columns
    for t in range(ROLL_WIN, len(idx)):
        win = R.iloc[t-ROLL_WIN:t].dropna(how="any")
        if len(win) < max(10, len(cols)+1):
            w = np.ones(len(cols))/len(cols)
        else:
            mu  = win.mean().values
            Sig = win.cov().values
            w   = tangency_weights(mu, Sig, long_only=True)
        ws.append(pd.Series(w, index=cols, name=idx[t]))
        rets.append((R.iloc[t] * w).sum())
    return pd.DataFrame(ws), pd.Series(rets, index=idx[ROLL_WIN:])

def run_mgarch_mvo(R):
    F = pd.read_parquet(OUT/"mgarch_features.parquet").reindex(R.index).ffill()
    pair2rho = {("SPY","GLD"): "rho_SPY_GLD",
                ("SPY","TLT"): "rho_SPY_TLT",
                ("GLD","TLT"): "rho_GLD_TLT"}
    order = ["SPY","GLD","TLT"]
    rets, ws = [], []
    idx, cols = R.index, R.columns
    for t in range(ROLL_WIN, len(idx)):
        row = F.iloc[t]
        sig = np.array([row.get(f"sigma_{a}", np.nan) for a in order], dtype=float)
        used_static = True
        if np.all(np.isfinite(sig)):
            rvals = {k: float(row.get(v, np.nan)) for k, v in pair2rho.items()}
            if np.all([np.isfinite(v) for v in rvals.values()]):
                rho = np.eye(3)
                rho[0,1] = rho[1,0] = rvals[("SPY","GLD")]
                rho[0,2] = rho[2,0] = rvals[("SPY","TLT")]
                rho[1,2] = rho[2,1] = rvals[("GLD","TLT")]
                Sig3 = np.outer(sig, sig) * rho
                win = R.iloc[t-ROLL_WIN:t].dropna(how="any")
                mu  = (win.mean().values if len(win) >= max(10, len(cols)+1)
                       else np.zeros(len(cols)))
                # start from static cov then inject 3x3 block
                Sig = (win.cov().values if len(win)>=max(10,len(cols)+1)
                       else np.cov(R[cols].dropna().T))
                idx3 = [cols.get_loc(x) for x in order]
                for i,ii in enumerate(idx3):
                    for j,jj in enumerate(idx3):
                        Sig[ii,jj] = Sig3[i,j]
                w = tangency_weights(mu, Sig, long_only=True)
                used_static = False
        if used_static:
            win = R.iloc[t-ROLL_WIN:t].dropna(how="any")
            if len(win) < max(10, len(cols)+1):
                w = np.ones(len(cols))/len(cols)
            else:
                mu  = win.mean().values
                Sig = win.cov().values
                w   = tangency_weights(mu, Sig, long_only=True)
        ws.append(pd.Series(w, index=cols, name=idx[t]))
        rets.append((R.iloc[t] * w).sum())
    return pd.DataFrame(ws), pd.Series(rets, index=idx[ROLL_WIN:])

def metrics(r):
    ann = r.mean()*52
    vol = r.std(ddof=0)*np.sqrt(52)
    shp = ann/vol if vol>0 else np.nan
    cw  = np.exp(r.cumsum())
    mdd = (cw/cw.cummax()-1).min()
    return dict(AnnRet=float(ann), Vol=float(vol), Sharpe=float(shp), MaxDD=float(mdd))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    R = load_weekly_returns().dropna(how="all")

    w_static, r_static = run_static_mvo(R[ASSET_PORT])
    w_mg,     r_mg     = run_mgarch_mvo(R[ASSET_PORT])

    w_static.to_csv(OUT/"sec6_weights_static_mvo.csv")
    w_mg.to_csv(OUT/"sec6_weights_mgarch_mvo.csv")
    pd.DataFrame({"static": r_static, "mgarch": r_mg}).to_csv(OUT/"sec6_mvo_port_rets.csv")

    m_static = metrics(r_static); m_mg = metrics(r_mg)
    pd.DataFrame([{"Model":"Static MVO", **m_static},
                  {"Model":"MGARCH-MVO", **m_mg}]).to_csv(OUT/"sec6_mvo_metrics.csv", index=False)

    plt.figure(figsize=(9,5))
    for name, r in [("Static MVO", r_static), ("MGARCH-MVO", r_mg)]:
        cw = np.exp(r.cumsum()); plt.plot(cw.index, cw.values, label=name)
    plt.title("§6 Static MVO vs MGARCH-MVO — Cumulative wealth")
    plt.xlabel("Date"); plt.ylabel("Cumulative wealth"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG/"sec6_mvo_cumret.png", dpi=150)

    print("[ok] wrote:",
          OUT/"sec6_weights_static_mvo.csv",
          OUT/"sec6_weights_mgarch_mvo.csv",
          OUT/"sec6_mvo_port_rets.csv",
          OUT/"sec6_mvo_metrics.csv",
          FIG/"sec6_mvo_cumret.png")

if __name__ == "__main__":
    main()
