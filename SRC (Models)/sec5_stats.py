#!/usr/bin/env python
"""
Statistical validation for Section 5.3.

Inputs (from Section 5 runs, in current folder by default):
- weekly_returns_transformer.csv
- weekly_returns_lstm.csv

Outputs (into --out_dir, default ./figures):
- sec5_stats_summary.txt   # human-readable summary
- sec5_stats_table.csv     # machine-friendly table of all stats

Tests included:
  (1) Diebold–Mariano on weekly net returns (loss = -r or quadratic utility).
  (2) Sharpe difference: circular block bootstrap CI + p-value.
  (3) HAC (Lo, 2002) Sharpe SE for each strategy + z-test for Sharpe difference.
  (4) Jobson–Korkie with Memmel adjustment (i.i.d. assumption).
"""

import os, math, argparse, numpy as np, pandas as pd
from typing import Tuple

# ---------- Helpers ----------
def ann_stats(weekly: pd.Series, rf_annual=0.03):
    rf_w = (1 + rf_annual) ** (1/52) - 1
    ex = weekly - rf_w
    ann_ret = weekly.mean() * 52
    ann_vol = weekly.std(ddof=0) * math.sqrt(52)
    sharpe = ex.mean() / (weekly.std(ddof=0) + 1e-12)
    return ann_ret, ann_vol, sharpe

def newey_west_lrv(x: np.ndarray, lag: int = None) -> float:
    """Long-run variance with Bartlett weights (Newey–West)."""
    T = len(x)
    if lag is None:
        lag = int(np.floor(T ** (1/3)))  # common automatic bandwidth
        lag = max(1, lag)
    x = x - x.mean()
    gamma0 = np.dot(x, x) / T
    lrv = gamma0
    for h in range(1, lag + 1):
        w = 1.0 - h / (lag + 1.0)
        cov = np.dot(x[h:], x[:-h]) / T
        lrv += 2.0 * w * cov
    return float(lrv)

def diebold_mariano(a: pd.Series, b: pd.Series, loss: str = "neg_return", h: int = 1, rf_annual=0.03) -> Tuple[float, float]:
    """
    DM test for equal expected loss between two strategies.
    a, b: weekly net returns (aligned Series)
    loss = 'neg_return' (ℓ = -r) or 'quad' (ℓ = -r + 0.5 * r^2 to mimic risk aversion)
    Returns: (DM_stat, two_sided_pvalue) using N(0,1) approximation.
    """
    idx = a.index.intersection(b.index)
    a = a.loc[idx].astype(float)
    b = b.loc[idx].astype(float)

    if loss == "neg_return":
        la = -a.values
        lb = -b.values
    elif loss == "quad":
        la = -a.values + 0.5 * (a.values ** 2)
        lb = -b.values + 0.5 * (b.values ** 2)
    else:
        raise ValueError("loss must be 'neg_return' or 'quad'")

    d = la - lb  # positive if A is worse than B
    T = len(d)
    dbar = d.mean()
    Sd = newey_west_lrv(d, lag=None)  # HAC long-run variance
    dm = dbar / math.sqrt(Sd / T + 1e-18)
    # two-sided p-value under N(0,1):
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(dm) / sqrt(2))))
    return dm, p

def block_bootstrap_sharpe_diff(a: pd.Series, b: pd.Series, B=5000, block_len=8, seed=1337, rf_annual=0.03):
    """Circular block bootstrap for ΔSharpe = Sharpe(a) - Sharpe(b). Returns (diffs, [lo,hi], p_one_sided)."""
    rng = np.random.default_rng(seed)
    idx = a.index.intersection(b.index)
    a = a.loc[idx].astype(float).values
    b = b.loc[idx].astype(float).values
    n = len(a)
    k = int(np.ceil(n / block_len))
    rf_w = (1 + rf_annual) ** (1/52) - 1
    diffs = np.empty(B)
    for i in range(B):
        starts = rng.integers(0, n, size=k)
        sel = np.concatenate([np.arange(s, s+block_len) % n for s in starts])[:n]
        aa = a[sel]; bb = b[sel]
        shA = ((aa - rf_w).mean() / (aa.std() + 1e-12))
        shB = ((bb - rf_w).mean() / (bb.std() + 1e-12))
        diffs[i] = shA - shB
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pval_one_sided = (diffs <= 0).mean()  # H0: A <= B
    return diffs, (lo, hi), pval_one_sided

def lo_hac_sharpe_se(x: pd.Series, rf_annual=0.03, q: int = None):
    """
    Lo (2002) HAC standard error for Sharpe ratio with serial correlation.
    Returns (Sharpe, SE). Uses automatic bandwidth q = floor((3T)^(1/3)) if None.
    """
    x = x.astype(float)
    T = len(x)
    rf_w = (1 + rf_annual) ** (1/52) - 1
    ex = x - rf_w
    mu = ex.mean()
    sigma = x.std(ddof=0)
    S = mu / (sigma + 1e-12)

    if q is None:
        q = int(np.floor((3*T) ** (1/3)))
        q = max(1, q)

    # Autocovariances of returns up to q
    gam = [np.cov(x[k:], x[:-k], ddof=0)[0,1] if k>0 else np.var(x, ddof=0) for k in range(q+1)]
    rho = [gam[k]/gam[0] if gam[0]!=0 else 0.0 for k in range(q+1)]
    phi = 1.0 + 2.0 * sum((1 - k/(q+1.0)) * rho[k] for k in range(1, q+1))  # variance inflation
    var_S = ( (1 + 0.5 * S**2) * phi ) / T
    se_S = math.sqrt(max(var_S, 0.0))
    return S, se_S

# i.i.d. Jobson–Korkie with Memmel correction (use with caution on weekly financial data)
def jobson_korkie_memmel(a: pd.Series, b: pd.Series, rf_annual=0.03):
    """
    Returns (Sa, Sb, z, two_sided_p). Assumes i.i.d. returns; not HAC-robust.
    Implemented for completeness; prefer bootstrap + Lo HAC SE for inference.
    """
    from math import erf, sqrt
    idx = a.index.intersection(b.index)
    a = a.loc[idx].astype(float); b = b.loc[idx].astype(float)
    T = len(a)
    rf_w = (1 + rf_annual)**(1/52) - 1
    ex_a = a - rf_w; ex_b = b - rf_w
    Sa = ex_a.mean() / (a.std(ddof=0) + 1e-12)
    Sb = ex_b.mean() / (b.std(ddof=0) + 1e-12)
    rho = np.corrcoef(a, b)[0,1]
    # Memmel (2003) finite-sample variance for Sharpe difference
    var_diff = (1/T) * ( (2*(1 - rho)) + 0.5*(Sa**2 + Sb**2) - rho*Sa*Sb )
    z = (Sa - Sb) / math.sqrt(max(var_diff, 1e-18))
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return (Sa, Sb, z, p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".")
    ap.add_argument("--out_dir", type=str, default="figures")
    ap.add_argument("--rf_annual", type=float, default=0.03)
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--block_len", type=int, default=8)
    ap.add_argument("--dm_loss", type=str, default="neg_return", choices=["neg_return","quad"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load weekly net returns (auto-detect column name)
    def pick_col(df):
        for c in ["weekly_r_net","Portfolio","weekly_net","net"]:
            if c in df.columns: return df[c]
        if df.shape[1]==1: return df.iloc[:,0]
        raise ValueError("Cannot find net return column.")
    rT = pd.read_csv(os.path.join(args.data_dir,"weekly_returns_transformer.csv"), index_col=0, parse_dates=True)
    rL = pd.read_csv(os.path.join(args.data_dir,"weekly_returns_lstm.csv"), index_col=0, parse_dates=True)
    a = pick_col(rT); b = pick_col(rL)
    idx = a.index.intersection(b.index)
    a = a.loc[idx].astype(float); b = b.loc[idx].astype(float)

    # Descriptive Sharpe/returns
    annT, volT, shT = ann_stats(a, rf_annual=args.rf_annual)
    annL, volL, shL = ann_stats(b, rf_annual=args.rf_annual)

    # (1) Diebold–Mariano
    dm_stat, dm_p = diebold_mariano(a, b, loss=args.dm_loss, rf_annual=args.rf_annual)

    # (2) Bootstrap ΔSharpe
    diffs, (ci_lo, ci_hi), p_one_sided = block_bootstrap_sharpe_diff(
        a, b, B=args.B, block_len=args.block_len, rf_annual=args.rf_annual
    )

    # (3) Lo (2002) HAC Sharpe SE + z-test for difference
    ShT, seT = lo_hac_sharpe_se(a, rf_annual=args.rf_annual)
    ShL, seL = lo_hac_sharpe_se(b, rf_annual=args.rf_annual)
    se_diff = math.sqrt(seT**2 + seL**2)
    z_diff = (ShT - ShL) / (se_diff + 1e-18)
    from math import erf, sqrt
    p_two_sided = 2 * (1 - 0.5 * (1 + erf(abs(z_diff) / sqrt(2))))

    # (4) JK/Memmel (i.i.d.)
    Sa, Sb, z_jk, p_jk = jobson_korkie_memmel(a, b, rf_annual=args.rf_annual)

    # Save outputs (TXT)
    lines = []
    lines.append("Section 5.3 Statistical Validation\n")
    lines.append(f"Transformer: AnnRet={annT:.4f}, Vol={volT:.4f}, Sharpe={shT:.4f}\n")
    lines.append(f"LSTM       : AnnRet={annL:.4f}, Vol={volL:.4f}, Sharpe={shL:.4f}\n\n")
    lines.append(f"Diebold–Mariano (loss='{args.dm_loss}'): DM={dm_stat:.3f}, p(two-sided)={dm_p:.4f}\n")
    lines.append(f"Bootstrap ΔSharpe: 95% CI=[{ci_lo:.3f}, {ci_hi:.3f}], p(one-sided A<=B)={p_one_sided:.4f}\n")
    lines.append(f"Lo (2002) HAC Sharpe SE: Sh_T={ShT:.3f} (SE={seT:.3f}), Sh_L={ShL:.3f} (SE={seL:.3f}); ΔSh={ShT-ShL:.3f}, z={z_diff:.3f}, p(two-sided)={p_two_sided:.4f}\n")
    lines.append(f"Jobson–Korkie (Memmel adj., i.i.d.): z={z_jk:.3f}, p(two-sided)={p_jk:.4f}\n")

    out_txt = os.path.join(args.out_dir, "sec5_stats_summary.txt")
    with open(out_txt, "w") as f:
        f.writelines(lines)

    # Save table (CSV)
    table = pd.DataFrame([{
        "AnnRet_T": annT, "Vol_T": volT, "Sharpe_T": shT,
        "AnnRet_L": annL, "Vol_L": volL, "Sharpe_L": shL,
        "DM_stat": dm_stat, "DM_p_two_sided": dm_p,
        "Boot_CI_lo": ci_lo, "Boot_CI_hi": ci_hi, "Boot_p_one_sided": p_one_sided,
        "Lo_Sh_T": ShT, "Lo_SE_T": seT, "Lo_Sh_L": ShL, "Lo_SE_L": seL,
        "Lo_z_diff": z_diff, "Lo_p_two_sided": p_two_sided,
        "JK_z": z_jk, "JK_p_two_sided": p_jk
    }])
    out_csv = os.path.join(args.out_dir, "sec5_stats_table.csv")
    table.to_csv(out_csv, index=False)

    print(f"Wrote: {os.path.relpath(out_txt)} and {os.path.relpath(out_csv)}")

if __name__ == "__main__":
    main()
