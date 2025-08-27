#!/usr/bin/env python
"""
Section 5.4 – Interpretability & Regime Behavior

Inputs (defaults assume current folder):
- weights_transformer.csv          # columns: tickers, index: dates
- weights_lstm.csv                 # columns: tickers, index: dates
- weekly_returns_transformer.csv   # contains portfolio weekly net returns (e.g., 'weekly_r_net')
- weekly_returns_lstm.csv          # same for LSTM
- (optional) asset_returns.csv     # weekly returns by ticker, incl. SPY column; index: dates

Outputs (to --out_dir, default ./figures):
- sec54_effN_turnover.csv
- sec54_regime_metrics.csv
- sec54_effN_topk_turnover.pdf/.png
- sec54_regime_sharpes.pdf/.png
- sec54_regime_sortino.pdf/.png
"""

import os, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------- helpers -----------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_indexed_csv(path):
    df = pd.read_csv(path, index_col=0)
    # try parse dates
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df.sort_index()

def pick_port_ret_col(df):
    for c in ["weekly_r_net","Portfolio","weekly_net","net","ret","return"]:
        if c in df.columns: return df[c].astype(float)
    if df.shape[1]==1: return df.iloc[:,0].astype(float)
    raise ValueError(f"Cannot find portfolio return column in {list(df.columns)}")

def eff_breadth(w: pd.DataFrame) -> pd.Series:
    # N_eff = 1 / sum_i w_i^2 (row-wise)
    return 1.0 / (w.pow(2).sum(axis=1) + 1e-18)

def topk_conc(w: pd.DataFrame, k=3) -> pd.Series:
    # sum of top-k weights (row-wise)
    return w.apply(lambda row: np.sort(row.values)[-k:].sum(), axis=1)

def turnover(w: pd.DataFrame) -> pd.Series:
    # L1 turnover between periods (row-wise change)
    dw = w.diff().abs().sum(axis=1)
    dw.iloc[0] = np.nan
    return dw

def series_to_weekly(x: pd.Series):
    # enforce weekly (Fri) frequency; forward-fill within week
    if x.index.inferred_type != "datetime64":
        try: x.index = pd.to_datetime(x.index)
        except: return x
    w = x.resample("W-FRI").last().ffill()
    return w

def sharpe(weekly, rf_annual=0.03):
    rf_w = (1 + rf_annual)**(1/52) - 1
    ex = weekly - rf_w
    return ex.mean() / (weekly.std(ddof=0) + 1e-12)

def sortino(weekly, rf_annual=0.03):
    rf_w = (1 + rf_annual)**(1/52) - 1
    ex = weekly - rf_w
    downside = ex[ex < 0]
    dd = downside.std(ddof=0)
    return ex.mean() / (dd + 1e-12)

def realized_vol(weekly, window=26):
    # rolling 26-week sample std (annualize if you want, but we only compare terciles)
    return weekly.rolling(window).std(ddof=0)

def make_market_proxy(asset_returns: pd.DataFrame):
    # equal-weight of available ETFs as a crude proxy if SPY is absent
    return asset_returns.mean(axis=1)

# ----------------------- main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_transformer", type=str, default="weights_transformer.csv")
    ap.add_argument("--weights_lstm", type=str, default="weights_lstm.csv")
    ap.add_argument("--weekly_returns_transformer", type=str, default="weekly_returns_transformer.csv")
    ap.add_argument("--weekly_returns_lstm", type=str, default="weekly_returns_lstm.csv")
    ap.add_argument("--asset_returns", type=str, default="", help="Optional weekly returns by ticker, incl. SPY column")
    ap.add_argument("--out_dir", type=str, default="figures")
    ap.add_argument("--rf_annual", type=float, default=0.03)
    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # ---- load weights ----
    W_T = read_indexed_csv(args.weights_transformer)
    W_L = read_indexed_csv(args.weights_lstm)
    # Normalize weights row-wise (safety)
    W_T = (W_T.clip(lower=0)).div((W_T.clip(lower=0)).sum(axis=1), axis=0).fillna(0)
    W_L = (W_L.clip(lower=0)).div((W_L.clip(lower=0)).sum(axis=1), axis=0).fillna(0)

    # Align to weekly Fridays
    def to_weekly_weights(W):
        if W.index.inferred_type != "datetime64":
            try: W.index = pd.to_datetime(W.index)
            except: pass
        return W.resample("W-FRI").last().ffill()
    W_T = to_weekly_weights(W_T)
    W_L = to_weekly_weights(W_L)

    # ---- interpretability time series ----
    effN_T = eff_breadth(W_T)
    effN_L = eff_breadth(W_L)
    top2_T = topk_conc(W_T, k=2); top2_L = topk_conc(W_L, k=2)
    top3_T = topk_conc(W_T, k=3); top3_L = topk_conc(W_L, k=3)
    turn_T = turnover(W_T);       turn_L = turnover(W_L)

    interp = pd.DataFrame({
        "effN_T": effN_T, "effN_L": effN_L,
        "top2_T": top2_T, "top2_L": top2_L,
        "top3_T": top3_T, "top3_L": top3_L,
        "turnover_T": turn_T, "turnover_L": turn_L
    })
    interp.to_csv(os.path.join(args.out_dir, "sec54_effN_turnover.csv"))

    # ---- plots: effective N, top-3, turnover ----
    plt.figure(figsize=(10,5))
    effN_T.plot()
    effN_L.plot()
    plt.title("Effective Breadth (1/∑ w²)")
    plt.legend(["Transformer","LSTM"])
    plt.xlabel("Date"); plt.ylabel("N_eff")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"sec54_effN.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(args.out_dir,"sec54_effN.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10,5))
    top3_T.plot()
    top3_L.plot()
    plt.title("Top-3 Concentration (∑ largest 3 weights)")
    plt.legend(["Transformer","LSTM"])
    plt.xlabel("Date"); plt.ylabel("Top-3 weight")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"sec54_top3.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(args.out_dir,"sec54_top3.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10,5))
    turn_T.plot()
    turn_L.plot()
    plt.title("Turnover per Rebalance (∑|Δw|)")
    plt.legend(["Transformer","LSTM"])
    plt.xlabel("Date"); plt.ylabel("Turnover")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"sec54_turnover.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(args.out_dir,"sec54_turnover.png"), bbox_inches="tight")
    plt.close()

    # ---- regime construction ----
    # Load portfolio weekly returns
    rT = read_indexed_csv(args.weekly_returns_transformer)
    rL = read_indexed_csv(args.weekly_returns_lstm)
    a = pick_port_ret_col(rT); b = pick_port_ret_col(rL)
    # Align
    idx = a.index.intersection(b.index)
    a = a.loc[idx].astype(float)
    b = b.loc[idx].astype(float)

    # Market proxy: SPY weekly returns if provided, else equal-weight of asset returns, else fallback to 'a'
    if args.asset_returns and os.path.exists(args.asset_returns):
        A = read_indexed_csv(args.asset_returns)
        A = A.apply(pd.to_numeric, errors="coerce")
        if "SPY" in A.columns:
            mkt = A["SPY"].astype(float)
        else:
            mkt = make_market_proxy(A).astype(float)
    else:
        # fallback: mean of Transformer and LSTM weekly returns (neutral proxy)
        mkt = pd.concat([a, b], axis=1).mean(axis=1)

    mkt = series_to_weekly(mkt).loc[idx]
    # Bull/Bear by sign
    bull = (mkt > 0)
    bear = (mkt <= 0)

    # Vol terciles by realized 26w vol
    rv = realized_vol(mkt, window=26).loc[idx]
    q1, q2 = rv.quantile([1/3, 2/3])
    lowv = rv <= q1
    medv = (rv > q1) & (rv <= q2)
    highv = rv > q2

    # compute metrics per mask
    def metrics(mask, name):
        x = a[mask]; y = b[mask]
        return {
            "regime": name,
            "n": int(mask.sum()),
            "sharpe_T": sharpe(x, args.rf_annual),
            "sharpe_L": sharpe(y, args.rf_annual),
            "sortino_T": sortino(x, args.rf_annual),
            "sortino_L": sortino(y, args.rf_annual),
        }

    rows = []
    rows += [metrics(bull,  "Bull")]
    rows += [metrics(bear,  "Bear")]
    rows += [metrics(lowv,  "LowVol (bottom tercile)")]
    rows += [metrics(medv,  "MedVol (middle tercile)")]
    rows += [metrics(highv, "HighVol (top tercile)")]

    reg = pd.DataFrame(rows)
    reg.to_csv(os.path.join(args.out_dir,"sec54_regime_metrics.csv"), index=False)

    # ---- bar plots: Sharpe & Sortino by regime ----
    # Order for plotting
    reg["order"] = reg["regime"].map({
        "Bull":0, "Bear":1,
        "LowVol (bottom tercile)":2, "MedVol (middle tercile)":3, "HighVol (top tercile)":4
    })
    reg = reg.sort_values("order")

    # Sharpe bars
    x = np.arange(len(reg))
    width = 0.38
    plt.figure(figsize=(11,5))
    plt.bar(x - width/2, reg["sharpe_T"].values, width, label="Transformer")
    plt.bar(x + width/2, reg["sharpe_L"].values, width, label="LSTM")
    plt.xticks(x, reg["regime"].values, rotation=20, ha="right")
    plt.ylabel("Sharpe (weekly, ex-RF)")
    plt.title("Regime-conditioned Sharpe")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"sec54_regime_sharpes.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(args.out_dir,"sec54_regime_sharpes.png"), bbox_inches="tight")
    plt.close()

    # Sortino bars
    plt.figure(figsize=(11,5))
    plt.bar(x - width/2, reg["sortino_T"].values, width, label="Transformer")
    plt.bar(x + width/2, reg["sortino_L"].values, width, label="LSTM")
    plt.xticks(x, reg["regime"].values, rotation=20, ha="right")
    plt.ylabel("Sortino (weekly, ex-RF)")
    plt.title("Regime-conditioned Sortino")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"sec54_regime_sortino.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(args.out_dir,"sec54_regime_sortino.png"), bbox_inches="tight")
    plt.close()

    print("Wrote:",
          "sec54_effN_turnover.csv, sec54_regime_metrics.csv,",
          "sec54_effN.pdf/png, sec54_top3.pdf/png, sec54_turnover.pdf/png,",
          "sec54_regime_sharpes.pdf/png, sec54_regime_sortino.pdf/png in", args.out_dir)

if __name__ == "__main__":
    main()
