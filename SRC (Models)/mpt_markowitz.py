#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional
from math import sqrt
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs"

# ---------- helpers ----------
def annualize(mu_d, cov_d, periods=252):
    return mu_d * periods, cov_d * periods

def safe_scalar(x: np.ndarray) -> float:
    return float(np.asarray(x).reshape(1)[0])

@dataclass
class Stats:
    mu_ann: float
    vol_ann: float
    sharpe_excess: float

def portfolio_stats(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, rf: float) -> Stats:
    mu_p = safe_scalar(w @ mu)
    vol_p = sqrt(safe_scalar(w @ Sigma @ w))
    sharpe = (mu_p - rf) / vol_p if vol_p > 0 else np.nan
    return Stats(mu_p, vol_p, sharpe)

# ---------- unconstrained closed-form ----------
def gmv_weights_unconstrained(Sigma: np.ndarray) -> np.ndarray:
    n = Sigma.shape[0]
    invS = np.linalg.pinv(Sigma)
    one = np.ones((n, 1))
    w = invS @ one
    w = w / safe_scalar(one.T @ invS @ one)
    return w.ravel()

def tangency_weights_unconstrained(mu: np.ndarray, Sigma: np.ndarray, rf: float) -> np.ndarray:
    n = Sigma.shape[0]
    invS = np.linalg.pinv(Sigma)
    one = np.ones((n, 1))
    excess = (mu.reshape(-1, 1) - rf * one)
    w = invS @ excess
    w = w / safe_scalar(one.T @ w)
    return w.ravel()

def frontier_parametric_unconstrained(mu: np.ndarray, Sigma: np.ndarray, npts=100) -> Tuple[np.ndarray,np.ndarray]:
    invS = np.linalg.pinv(Sigma)
    one = np.ones((len(mu), 1))
    mu = mu.reshape(-1, 1)
    A = safe_scalar(one.T @ invS @ one)
    B = safe_scalar(one.T @ invS @ mu)
    C = safe_scalar(mu.T  @ invS @ mu)
    D = A * C - B * B
    m_min, m_max = safe_scalar(mu.min()), safe_scalar(mu.max())
    grid = np.linspace(m_min, m_max, npts)
    variances = (A * grid**2 - 2 * B * grid + C) / D
    sigmas = np.sqrt(np.maximum(0.0, variances))
    return sigmas, grid

# ---------- constrained (long-only / caps) via SLSQP ----------
def solve_gmv_constrained(Sigma: np.ndarray, lb: float = 0.0, ub: Optional[float]=None) -> np.ndarray:
    n = Sigma.shape[0]
    x0 = np.ones(n) / n
    bounds = [(lb, 1.0 if ub is None else ub)] * n
    cons = [{"type":"eq", "fun": lambda w: np.sum(w) - 1.0}]
    obj = lambda w: safe_scalar(w @ Sigma @ w)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter":1000})
    if not res.success: raise RuntimeError(f"GMV solve failed: {res.message}")
    return res.x

def solve_tangency_constrained(mu: np.ndarray, Sigma: np.ndarray, rf: float, lb: float = 0.0, ub: Optional[float]=None) -> np.ndarray:
    n = Sigma.shape[0]
    x0 = np.ones(n) / n
    bounds = [(lb, 1.0 if ub is None else ub)] * n
    cons = [{"type":"eq", "fun": lambda w: np.sum(w) - 1.0}]
    # maximize Sharpe => minimize negative Sharpe
    def obj(w):
        mu_p = safe_scalar(w @ mu)
        vol_p = sqrt(safe_scalar(w @ Sigma @ w))
        return - (mu_p - rf) / vol_p if vol_p > 0 else 1e6
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter":1000})
    if not res.success: raise RuntimeError(f"Tangency solve failed: {res.message}")
    return res.x

def constrained_frontier(mu: np.ndarray, Sigma: np.ndarray, rf: float, npts=60, lb: float=0.0, ub: Optional[float]=None) -> pd.DataFrame:
    """
    Sweep target returns between min/max asset means; for each target mean m, min variance s.t.
       sum w = 1, w in [lb, ub], w@mu = m
    """
    n = len(mu)
    m_min, m_max = mu.min(), mu.max()
    grid = np.linspace(m_min, m_max, npts)
    rows = []
    for m in grid:
        x0 = np.ones(n)/n
        bounds = [(lb, 1.0 if ub is None else ub)] * n
        cons = [
            {"type":"eq", "fun": lambda w, tgt=m: safe_scalar(w @ mu) - tgt},
            {"type":"eq", "fun": lambda w: np.sum(w) - 1.0},
        ]
        obj = lambda w: safe_scalar(w @ Sigma @ w)
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter":1000})
        if not res.success:
            continue
        w = res.x
        s = portfolio_stats(w, mu, Sigma, rf)
        rows.append({"mu_ann": s.mu_ann, "vol_ann": s.vol_ann})
    return pd.DataFrame(rows).sort_values("vol_ann").drop_duplicates(subset=["vol_ann"])

# ---------- main ----------
def main():
    OUT.mkdir(exist_ok=True, parents=True)

    df = pd.read_parquet(ROOT / "outputs" / "prices.parquet")
    tickers = list(df.columns.levels[0])
    px = df.xs("Adj Close", level=1, axis=1).astype(float).dropna()
    rets_d = np.log(px/px.shift(1)).dropna()

    mu_a, Sigma_a = annualize(rets_d.mean().values, rets_d.cov().values)

    rf = 0.03  # set annual risk-free here

    # ----- Unconstrained (can short / lever) -----
    w_gmv_uc  = gmv_weights_unconstrained(Sigma_a)
    w_tan_uc  = tangency_weights_unconstrained(mu_a, Sigma_a, rf=rf)
    gmv_uc_s  = portfolio_stats(w_gmv_uc, mu_a, Sigma_a, rf)
    tan_uc_s  = portfolio_stats(w_tan_uc, mu_a, Sigma_a, rf)
    ef_uc_sig, ef_uc_mu = frontier_parametric_unconstrained(mu_a, Sigma_a, npts=200)
    ef_uc = pd.DataFrame({"vol_ann": ef_uc_sig, "mu_ann": ef_uc_mu}).sort_values("vol_ann")

    # ----- Constrained long-only (and optional caps) -----
    cap = None           # e.g., set to 0.3 to cap each weight at 30%
    w_gmv_lo = solve_gmv_constrained(Sigma_a, lb=0.0, ub=cap)
    w_tan_lo = solve_tangency_constrained(mu_a, Sigma_a, rf=rf, lb=0.0, ub=cap)
    gmv_lo_s = portfolio_stats(w_gmv_lo, mu_a, Sigma_a, rf)
    tan_lo_s = portfolio_stats(w_tan_lo, mu_a, Sigma_a, rf)
    ef_lo = constrained_frontier(mu_a, Sigma_a, rf=rf, npts=120, lb=0.0, ub=cap)

    # ----- save artifacts -----
    pd.Series(mu_a, index=tickers, name="mu_ann").to_csv(OUT/"mu_ann.csv")
    pd.DataFrame(Sigma_a, index=tickers, columns=tickers).to_csv(OUT/"Sigma_ann.csv")
    ef_uc.to_csv(OUT/"efficient_frontier_unconstrained.csv", index=False)
    ef_lo.to_csv(OUT/"efficient_frontier_longonly.csv", index=False)
    (OUT/"gmv_unconstrained.json").write_text(json.dumps({
        "weights": dict(zip(tickers, map(float, w_gmv_uc))),
        "mu_ann": gmv_uc_s.mu_ann, "vol_ann": gmv_uc_s.vol_ann, "sharpe_excess": gmv_uc_s.sharpe_excess
    }, indent=2))
    (OUT/"tangency_unconstrained.json").write_text(json.dumps({
        "weights": dict(zip(tickers, map(float, w_tan_uc))),
        "mu_ann": tan_uc_s.mu_ann, "vol_ann": tan_uc_s.vol_ann, "sharpe_excess": tan_uc_s.sharpe_excess
    }, indent=2))
    (OUT/"gmv_longonly.json").write_text(json.dumps({
        "weights": dict(zip(tickers, map(float, w_gmv_lo))),
        "mu_ann": gmv_lo_s.mu_ann, "vol_ann": gmv_lo_s.vol_ann, "sharpe_excess": gmv_lo_s.sharpe_excess
    }, indent=2))
    (OUT/"tangency_longonly.json").write_text(json.dumps({
        "weights": dict(zip(tickers, map(float, w_tan_lo))),
        "mu_ann": tan_lo_s.mu_ann, "vol_ann": tan_lo_s.vol_ann, "sharpe_excess": tan_lo_s.sharpe_excess
    }, indent=2))

    # console summary
    def brief(tag, s: Stats):
        print(f"[{tag}] mu={s.mu_ann:.4f} vol={s.vol_ann:.4f} sharpe_excess={s.sharpe_excess:.3f}")

    brief("GMV_uc", gmv_uc_s)
    brief("TAN_uc", tan_uc_s)
    brief("GMV_lo", gmv_lo_s)
    brief("TAN_lo", tan_lo_s)

if __name__ == "__main__":
    main()
