#!/usr/bin/env python
import os, math, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt

# ---------- Helpers ----------
def ann_stats(weekly, rf_annual=0.03):
    """Return dict of annualized metrics on a weekly return Series."""
    rf_w = (1 + rf_annual)**(1/52) - 1
    ex = weekly - rf_w
    ann_ret = weekly.mean() * 52
    ann_vol = weekly.std(ddof=0) * math.sqrt(52)
    sharpe = ex.mean() / (weekly.std(ddof=0) + 1e-12)
    # Sortino
    neg = weekly.copy(); neg[neg>0] = 0.0
    sortino = ex.mean() / ((neg.std(ddof=0) + 1e-12) * math.sqrt(52))
    # Drawdown
    nav = (1 + weekly).cumprod()
    peak = nav.cummax()
    dd = nav/peak - 1.0
    mdd = dd.min()
    calmar = ann_ret / (abs(mdd) + 1e-12)
    return dict(AnnRet=ann_ret, Vol=ann_vol, Sharpe=sharpe, Sortino=sortino, MaxDD=mdd, Calmar=calmar)

def block_bootstrap_sharpe_diff(rA, rB, B=5000, block_len=8, seed=1337, rf_annual=0.03):
    """Bootstrap CI for Sharpe(rA)-Sharpe(rB) using circular block bootstrap."""
    rng = np.random.default_rng(seed)
    rA = rA.dropna(); rB = rB.dropna()
    idx = rA.index.intersection(rB.index)
    rA = rA.loc[idx].values; rB = rB.loc[idx].values
    n = len(idx)
    k = int(np.ceil(n / block_len))
    diffs = np.empty(B)
    rf_w = (1 + rf_annual)**(1/52) - 1
    for b in range(B):
        starts = rng.integers(0, n, size=k)
        sel = np.concatenate([np.arange(s, s+block_len) % n for s in starts])[:n]
        a = rA[sel]; b_ = rB[sel]
        # Sharpe with weekly std; excess over rf
        shA = ((a - rf_w).mean() / (a.std() + 1e-12))
        shB = ((b_ - rf_w).mean() / (b_.std() + 1e-12))
        diffs[b] = shA - shB
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # one-sided p-value that A <= B (i.e., diff <= 0)
    pval = (diffs <= 0).mean()
    return diffs, (lo, hi), pval

def turnover_from_weights(W):
    """L1 turnover per period from weights DataFrame (rows sum to ~1)."""
    W_prev = W.shift(1).fillna(0.0)
    return (W - W_prev).abs().sum(axis=1)

# ---------- Main robustness script ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".", help="Folder with weights_*.csv and weekly_returns_*.csv")
    ap.add_argument("--fig_dir", type=str, default="figures", help="Output figures dir")
    ap.add_argument("--tc_bps_base", type=float, default=5.0, help="Base costs already applied in weekly_r_net (bps)")
    ap.add_argument("--tc_bps_stress", type=float, nargs="+", default=[10.0, 25.0, 50.0], help="Stress cost bps")
    ap.add_argument("--noise_sigma_mult", type=float, nargs="+", default=[0.25, 0.50], help="Multipliers of baseline sigma for noise stress")
    ap.add_argument("--rf_annual", type=float, default=0.03)
    ap.add_argument("--B", type=int, default=5000, help="Bootstrap resamples")
    ap.add_argument("--block_len", type=int, default=8, help="Bootstrap block length in weeks")
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    # Load series
    rT = pd.read_csv(os.path.join(args.data_dir, "weekly_returns_transformer.csv"), index_col=0, parse_dates=True)
    rL = pd.read_csv(os.path.join(args.data_dir, "weekly_returns_lstm.csv"), index_col=0, parse_dates=True)
    # Accept either columns naming scheme
    # If user saved "weekly_r_net" or "Portfolio" etc.
    def pick_col(df):
        for c in ["weekly_r_net", "Portfolio", "weekly_net", "net"]:
            if c in df.columns: return df[c]
        # fallback: if single column, use it
        if df.shape[1]==1: return df.iloc[:,0]
        raise ValueError("Can't find net return column in weekly returns CSV.")
    netT = pick_col(rT)
    netL = pick_col(rL)
    # Align
    idx = netT.index.intersection(netL.index)
    netT = netT.loc[idx].astype(float)
    netL = netL.loc[idx].astype(float)

    # Baseline metrics
    base_T = ann_stats(netT, rf_annual=args.rf_annual)
    base_L = ann_stats(netL, rf_annual=args.rf_annual)

    # -------- Drawdowns plot
    def equity_and_dd(x):
        nav = (1+x).cumprod()
        dd = nav / nav.cummax() - 1.0
        return nav, dd

    navT, ddT = equity_and_dd(netT)
    navL, ddL = equity_and_dd(netL)

    plt.figure(figsize=(11, 7))
    ax1 = plt.subplot(2,1,1)
    navT.plot(ax=ax1, label="Transformer", lw=2)
    navL.plot(ax=ax1, label="LSTM", lw=2, ls="--")
    ax1.set_title("Equity Curves (Weekly, Net of Base Costs)")
    ax1.legend(); ax1.grid(alpha=0.3, ls=":")
    ax2 = plt.subplot(2,1,2)
    ddT.plot(ax=ax2, label="Transformer", lw=2)
    ddL.plot(ax=ax2, label="LSTM", lw=2, ls="--")
    ax2.set_title("Drawdowns")
    ax2.legend(); ax2.grid(alpha=0.3, ls=":")
    plt.tight_layout()
    out1 = os.path.join(args.fig_dir, "sec5_drawdowns.pdf")
    plt.savefig(out1, bbox_inches="tight"); plt.close()

    # -------- Transaction-cost stress: recompute net using weights
    # Load weights to compute turnover and apply new cost rates
    WT = pd.read_csv(os.path.join(args.data_dir, "weights_transformer.csv"), index_col=0, parse_dates=True)
    WL = pd.read_csv(os.path.join(args.data_dir, "weights_lstm.csv"), index_col=0, parse_dates=True)
    # Align to returns index
    WT = WT.loc[idx].fillna(0.0); WL = WL.loc[idx].fillna(0.0)
    turnT = turnover_from_weights(WT)
    turnL = turnover_from_weights(WL)

    stress_rows = []
    # Baseline entry
    stress_rows.append(dict(Model="Transformer", Stress="Baseline", **base_T))
    stress_rows.append(dict(Model="LSTM",        Stress="Baseline", **base_L))

    # Costs stress
    for bps in args.tc_bps_stress:
        extra_bps = max(0.0, bps - args.tc_bps_base)
        extra_cost_T = turnT * (extra_bps/1e4)
        extra_cost_L = turnL * (extra_bps/1e4)
        netT_cost = (netT - extra_cost_T).dropna()
        netL_cost = (netL - extra_cost_L).dropna()
        stress_rows.append(dict(Model="Transformer", Stress=f"TC {bps:.0f}bps", **ann_stats(netT_cost, rf_annual=args.rf_annual)))
        stress_rows.append(dict(Model="LSTM",        Stress=f"TC {bps:.0f}bps", **ann_stats(netL_cost, rf_annual=args.rf_annual)))

    # Noise stress
    sigT = netT.std(); sigL = netL.std()
    rng = np.random.default_rng(1337)
    for mult in args.noise_sigma_mult:
        noiseT = rng.normal(0.0, sigT*mult, size=len(netT))
        noiseL = rng.normal(0.0, sigL*mult, size=len(netL))
        netT_noisy = (netT + noiseT).astype(float)
        netL_noisy = (netL + noiseL).astype(float)
        stress_rows.append(dict(Model="Transformer", Stress=f"Noise {mult:.2f}σ", **ann_stats(netT_noisy, rf_annual=args.rf_annual)))
        stress_rows.append(dict(Model="LSTM",        Stress=f"Noise {mult:.2f}σ", **ann_stats(netL_noisy, rf_annual=args.rf_annual)))

    summary = pd.DataFrame(stress_rows)
    out_csv = os.path.join(args.fig_dir, "sec5_robust_summary.csv")
    summary.to_csv(out_csv, index=False)

    # Plot stress Sharpe curves
    order = ["Baseline"] + [f"TC {int(x)}bps" for x in args.tc_bps_stress] + [f"Noise {m:.2f}σ" for m in args.noise_sigma_mult]
    def pick(df, model):
        d = df[df.Model==model].set_index("Stress").reindex(order)
        return d["Sharpe"]
    plt.figure(figsize=(10,5))
    pick(summary, "Transformer").plot(marker="o", lw=2, label="Transformer")
    pick(summary, "LSTM").plot(marker="s", lw=2, label="LSTM")
    plt.title("Sharpe Under Stress (Costs and Noise)")
    plt.ylabel("Sharpe"); plt.grid(alpha=0.3, ls=":")
    plt.legend()
    out2 = os.path.join(args.fig_dir, "sec5_stress_curves.pdf")
    plt.tight_layout(); plt.savefig(out2, bbox_inches="tight"); plt.close()

    # -------- Bootstrap Sharpe difference
    diffs, (lo,hi), pval = block_bootstrap_sharpe_diff(netT, netL, B=args.B, block_len=args.block_len, rf_annual=args.rf_annual)
    print(f"Bootstrap Sharpe diff (Transformer - LSTM): 95% CI = [{lo:.3f}, {hi:.3f}],  p(one-sided, <=0) = {pval:.4f}")
    # Save a tiny text file with the CI/pval
    with open(os.path.join(args.fig_dir, "sec5_bootstrap_summary.txt"), "w") as f:
        f.write(f"Sharpe diff 95% CI: [{lo:.3f}, {hi:.3f}]\n")
        f.write(f"p-value (H0: Transf <= LSTM): {pval:.4f}\n")

    print("Saved:",
          os.path.relpath(out1), os.path.relpath(out2), os.path.relpath(out_csv),
          "and figures/sec5_bootstrap_summary.txt")

if __name__ == "__main__":
    main()
