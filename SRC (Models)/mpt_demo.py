#!/usr/bin/env python3
import numpy as np, pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs"

def efficient_frontier(mu, Sigma, n_points=50, short=False):
    """Compute a naive frontier by target-return quadratic programs (closed-form for no constraints ignored).
       Here we use a grid-search on weights with simplex projection for portability (no CVX dependency)."""
    n = len(mu)
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(200000):
        w = rng.random(n)
        w = w / w.sum()
        if not short and (w < -1e-8).any():
            continue
        ret = w @ mu
        vol = np.sqrt(w @ Sigma @ w)
        samples.append((vol, ret, w))
    samples = sorted(samples, key=lambda x: x[0])
    vols = [s[0] for s in samples]
    rets = [s[1] for s in samples]
    ws = [s[2] for s in samples]
    return np.array(vols), np.array(rets), np.array(ws)

def demo_static_frontier():
    prices_path = ROOT/"outputs"/"prices.parquet"
    if not prices_path.exists():
        # create a tiny synthetic price panel so the plot works immediately
        dates = pd.date_range("2015-01-01","2019-12-31", freq="B")
        rng = np.random.default_rng(0)
        cols = ["SPY","TLT","GLD","DBC"]
        rets = pd.DataFrame(rng.normal([0.0003,0.00015,0.0002,0.00025], [0.01,0.007,0.009,0.012], size=(len(dates),4)), index=dates, columns=cols)
        px = 100*np.exp(rets.cumsum())
    else:
        df = pd.read_parquet(prices_path)
        px = df.xs("Adj Close", level=1, axis=1).copy()
        # Coerce everything to numeric and drop garbage rows
        px = px.apply(pd.to_numeric, errors="coerce")
        px = px.replace([np.inf, -np.inf], np.nan)
        px = px.dropna(how="all")

        keep = [c for c in ["SPY","TLT","GLD","DBC"] if c in px.columns]
        if len(keep) < 4:
            keep = px.columns[:4].tolist()
        px = px[keep]

    rets = np.log(px/px.shift(1)).dropna()
    mu = rets.mean().values*252
    Sigma = rets.cov().values*252

    vols, rets_f, _ = efficient_frontier(mu, Sigma, n_points=80)

    plt.figure(figsize=(7,5))
    plt.scatter(vols, rets_f, s=6)
    plt.xlabel("Volatility (ann.)")
    plt.ylabel("Expected Return (ann.)")
    plt.title("Approximate Efficient Frontier (static window)")
    plt.tight_layout()
    out1 = OUT/"frontier_static.png"
    plt.savefig(out1, dpi=150)
    print(f"[ok] wrote {out1}")

def demo_shifting_frontier():
    # Create two regimes with different covariances to illustrate frontier shift
    rng = np.random.default_rng(1)
    cols = ["SPY","TLT","GLD","DBC"]
    T = 252*2
    mu1 = np.array([0.08, 0.03, 0.05, 0.06])/252
    mu2 = np.array([0.04, 0.04, 0.02, 0.01])/252
    # Covariances change: crisis-like correlation spike between SPY and DBC; TLT diversifies less
    Sigma1 = np.array([[0.20, -0.10, 0.05, 0.10],
                       [-0.10, 0.08, -0.05,-0.08],
                       [0.05, -0.05, 0.12, 0.02],
                       [0.10, -0.08, 0.02, 0.18]])/252
    Sigma2 = np.array([[0.28, -0.02, 0.10, 0.22],
                       [-0.02, 0.10, 0.00, 0.00],
                       [0.10, 0.00, 0.16, 0.12],
                       [0.22, 0.00, 0.12, 0.30]])/252

    def sample(mu, Sigma, n):
        return rng.multivariate_normal(mean=mu, cov=Sigma, size=n)

    R1 = sample(mu1, Sigma1, T)
    R2 = sample(mu2, Sigma2, T)
    df1 = pd.DataFrame(R1, columns=cols)
    df2 = pd.DataFrame(R2, columns=cols)

    def frontier_from(df):
        mu = df.mean().values*252
        Sigma = df.cov().values*252
        vols, rets, _ = efficient_frontier(mu, Sigma)
        return vols, rets

    v1, r1 = frontier_from(df1)
    v2, r2 = frontier_from(df2)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,5))
    plt.scatter(v1, r1, s=6, label="Regime 1")
    plt.scatter(v2, r2, s=6, label="Regime 2")
    plt.xlabel("Volatility (ann.)")
    plt.ylabel("Expected Return (ann.)")
    plt.title("Shifting Efficient Frontier Across Regimes")
    plt.legend()
    plt.tight_layout()
    out2 = OUT/"frontier_shift.png"
    plt.savefig(out2, dpi=150)
    print(f"[ok] wrote {out2}")

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True, parents=True)
    demo_static_frontier()
    demo_shifting_frontier()
