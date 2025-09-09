#!/usr/bin/env python3
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"
FIG  = ROOT / "figures"

ASSETS = ["SPY","GLD","TLT"]
H = 52   # horizon (weeks)
N = 1000 # paths

def load_weekly_prices():
    px = pd.read_parquet(OUT/"prices.parquet").xs("Adj Close", level=1, axis=1)
    px = px.apply(pd.to_numeric, errors="coerce")
    pxw = px.asfreq("B").ffill().resample("W-FRI").last()
    pxw = pxw.dropna(how="all")
    if pxw is None or pxw.empty:
        raise RuntimeError("No weekly prices available after cleaning — check outputs/prices.parquet.")
    return pxw

def load_mgarch_feats():
    return pd.read_parquet(OUT / "mgarch_features.parquet")

def cov_from_sigmarho(sigmas, corr_mat):
    D = np.diag(sigmas)
    return D @ corr_mat @ D

def main():
    FIG.mkdir(parents=True, exist_ok=True)
    pxw = load_weekly_prices()
    feats = load_mgarch_feats()

    last = pxw.dropna().index[-1]
    p0 = pxw.loc[last, ASSETS].values
    r  = np.log(pxw / pxw.shift(1)).dropna(how="all")
    mu = r[ASSETS].mean().values  # weekly drift

    row = feats.loc[:last].iloc[-1]
    s = np.array([float(row["sigma_SPY"]), float(row["sigma_GLD"]), float(row["sigma_TLT"])])
    rho = np.array([[1.0, float(row["rho_SPY_GLD"]), float(row["rho_SPY_TLT"])],
                    [float(row["rho_SPY_GLD"]), 1.0, float(row["rho_GLD_TLT"])],
                    [float(row["rho_SPY_TLT"]), float(row["rho_GLD_TLT"]), 1.0]])
    Sigma = cov_from_sigmarho(s, rho)

    # Cholesky for correlated shocks
    L = np.linalg.cholesky(Sigma + 1e-12*np.eye(len(ASSETS)))

    paths = np.zeros((H+1, N, len(ASSETS)))
    paths[0,:,:] = p0
    for t in range(1, H+1):
        z = L @ np.random.randn(len(ASSETS), N)
        r_t = mu.reshape(-1,1) + z                      # weekly log-returns
        paths[t,:,:] = paths[t-1,:,:] * np.exp(r_t.T)   # GBM step

    # Equal-weight portfolio bands
    w = np.ones(len(ASSETS))/len(ASSETS)
    port = (paths * w.reshape(1,1,-1)).sum(axis=2)
    q = np.percentile(port, [5,25,50,75,95], axis=1)
    weeks = np.arange(H+1)

    plt.figure(figsize=(9,5))
    for a,b in [(0,4),(1,3)]:
        plt.fill_between(weeks, q[a], q[b], alpha=0.2)
    plt.plot(weeks, q[2], lw=2, label="Median")
    plt.title("§6 Monte Carlo GBM (1y) — equal-weight SPY/GLD/TLT")
    plt.xlabel("Weeks ahead"); plt.ylabel("Portfolio level")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG/"sec6_gbm_bands.png", dpi=150)

    print("[ok] wrote", FIG/"sec6_gbm_bands.png")

if __name__ == "__main__":
    main()
