#!/usr/bin/env python3
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT/"outputs", ROOT/"figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

# ===== Universe & risk controls =====
ASSETS = ["SPY","QQQ","IWM","EFA","EEM","TLT","IEF","LQD","HYG","GLD","DBC","VNQ"]

GROUPS = {
    "Equity": ["SPY","QQQ","IWM","EFA","EEM","VNQ"],
    "Bonds" : ["TLT","IEF","LQD","HYG"],
    "Cmdty" : ["GLD","DBC"],
}
W_MAX = 0.25         # per-name cap
GROUP_CAPS = {"Equity": 0.70, "Bonds": 0.80, "Cmdty": 0.35}
TC_BPS = 5.0         # transaction cost per 1 unit turnover in basis points
ROLL_WIN = 156       # ~3y weekly for rolling cov/means (MVO)
RIDGE = 1e-6         # numerical regularization

# ===== Helpers =====
def load_weekly_prices():
    px = pd.read_parquet(OUT/"prices.parquet").xs("Adj Close", level=1, axis=1)
    px = px.apply(pd.to_numeric, errors="coerce")
    pxw = px.asfreq("B").ffill().resample("W-FRI").last()
    return pxw[ASSETS].dropna(how="all")

def weekly_returns():
    pxw = load_weekly_prices()
    r = np.log(pxw/pxw.shift(1))
    return r.dropna(how="all")

def proj_caps_simplex(w, cols):
    """Project raw weights onto {w>=0, sum=1, w_i<=W_MAX, group caps}."""
    w = np.maximum(np.asarray(w, float), 0.0)
    # cap per-name
    w = np.minimum(w, W_MAX)
    # renormalize
    s = w.sum()
    w = w/s if s > 0 else np.ones_like(w)/len(w)

    # enforce group caps by proportional squeeze if violated
    col2idx = {c:i for i,c in enumerate(cols)}
    for g, members in GROUPS.items():
        cap = GROUP_CAPS.get(g, 1.0)
        idxs = [col2idx[c] for c in members if c in col2idx]
        if not idxs: continue
        gsum = float(w[idxs].sum())
        if gsum > cap + 1e-12:
            scale = cap / gsum
            w[idxs] *= scale
            # renormalize remainder to sum to 1 (do not break other groups)
            rest = [i for i in range(len(w)) if i not in idxs]
            s_g = w[idxs].sum(); s_rest = w[rest].sum()
            # scale rest to fill 1 - s_g if s_rest>0 else leave as is
            if s_rest > 0:
                w[rest] *= (1 - s_g) / s_rest

    # final clip & renorm
    w = np.clip(w, 0, W_MAX)
    s = w.sum()
    return w/s if s > 0 else np.ones_like(w)/len(w)

def turnover_cost(prev, cur):
    if prev is None: return 0.0, 0.0
    to = 0.5*np.abs(cur - prev).sum()
    cost = (TC_BPS/1e4) * to
    return to, cost

def tangency(mu, Sigma):
    mu = np.asarray(mu, float)
    Sig = np.asarray(Sigma, float) + RIDGE*np.eye(len(mu))
    try:
        w = np.linalg.pinv(Sig) @ mu
    except Exception:
        w = np.ones_like(mu)/len(mu)
    return w

def ann_metrics(r):
    ann = r.mean()*52
    vol = r.std(ddof=0)*np.sqrt(52)
    shp = ann/vol if vol>0 else np.nan
    cw  = np.exp(r.cumsum())
    mdd = (cw/cw.cummax()-1).min()
    return dict(AnnRet=float(ann), Vol=float(vol), Sharpe=float(shp), MaxDD=float(mdd))

def load_mgarch_feats():
    # from §6
    f = pd.read_parquet(OUT/"mgarch_features.parquet")
    return f.asfreq("W-FRI").ffill()

def build_sigma_from_mgarch(row):
    """Σ_t for SPY/GLD/TLT from MGARCH proxies; others from rolling cov (caller handles merge)."""
    try:
        s_spy = float(row["sigma_SPY"]); s_gld = float(row["sigma_GLD"]); s_tlt = float(row["sigma_TLT"])
        r_sg  = float(row["rho_SPY_GLD"]); r_st  = float(row["rho_SPY_TLT"]); r_gt  = float(row["rho_GLD_TLT"])
    except Exception:
        return None
    s = np.array([s_spy, s_gld, s_tlt])
    rho = np.array([[1.0, r_sg, r_st],
                    [r_sg, 1.0, r_gt],
                    [r_st, r_gt, 1.0]])
    return np.outer(s, s) * rho

def load_model_weights(patterns):
    """Return dict{name -> DataFrame(weights indexed by Date)} for experiments passed in patterns list."""
    out = {}
    for name, expdir in patterns:
        exp = ROOT/"experiments"/expdir
        f = exp/"weights_transformer.csv"
        if not f.exists():
            f = exp/"weights_lstm.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        # pick date column
        dcol = next((c for c in df.columns if str(c).lower() in ("date","dt","timestamp")), None)
        if dcol is None:
            # try to infer if first column is date-like
            dcol = df.columns[0]
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=[dcol]).set_index(dcol).sort_index()
        # keep only our universe cols
        keep = [c for c in ASSETS if c in df.columns]
        if keep:
            out[name] = df[keep]
    return out

# ===== Strategies =====
def static_mvo(R):
    ws, rets, prev = [], [], None
    cols, idx = R.columns, R.index
    for t in range(ROLL_WIN, len(idx)):
        win = R.iloc[t-ROLL_WIN:t].dropna(how="any")
        if len(win) < max(30, len(cols)+1):
            wraw = np.ones(len(cols))/len(cols)
        else:
            mu  = win.mean().values
            Sig = win.cov().values
            wraw = tangency(mu, Sig)
        w = proj_caps_simplex(wraw, cols)
        tw, cost = turnover_cost(prev, w)
        pr = (R.iloc[t].values @ w) - cost
        prev = w
        ws.append(pd.Series(w, index=cols, name=idx[t]))
        rets.append(pr)
    return pd.DataFrame(ws), pd.Series(rets, index=idx[ROLL_WIN:])

def mgarch_mvo(R, F):
    ws, rets, prev = [], [], None
    cols, idx = R.columns, R.index
    core = ["SPY","GLD","TLT"]
    for t in range(ROLL_WIN, len(idx)):
        win = R.iloc[t-ROLL_WIN:t].dropna(how="any")
        mu  = win.mean().values if len(win) >= max(30,len(cols)+1) else np.zeros(len(cols))
        Sig = win.cov().values if len(win) >= max(30,len(cols)+1) else np.diag(np.full(len(cols), 1e-4))
        # inject MGARCH Σ for core block if available at date
        row = F.loc[:idx[t]].tail(1)
        if not row.empty:
            Sigma3 = build_sigma_from_mgarch(row.iloc[0])
            if Sigma3 is not None:
                i = [cols.get_loc(x) for x in core if x in cols]
                if len(i)==3:
                    for a,ia in enumerate(i):
                        for b,ib in enumerate(i):
                            Sig[ia, ib] = Sigma3[a,b]
        wraw = tangency(mu, Sig)
        w = proj_caps_simplex(wraw, cols)
        tw, cost = turnover_cost(prev, w)
        pr = (R.iloc[t].values @ w) - cost
        prev = w
        ws.append(pd.Series(w, index=cols, name=idx[t]))
        rets.append(pr)
    return pd.DataFrame(ws), pd.Series(rets, index=idx[ROLL_WIN:])

def model_weights_port(R, W):
    """
    Apply caps/TC to model-provided weights W (Date×Assets).
    Align strictly on dates; if a date is missing in W, carry forward
    the last available raw weights (live-like).
    """
    # Clean + align
    W = W.copy()
    # ensure numeric
    for c in W.columns:
        W[c] = pd.to_numeric(W[c], errors="coerce")
    W = W.sort_index().ffill()                # allow carry-forward of model weights
    R = R.copy()

    # start after we have both returns and at least one weight row to execute
    start_dt = max(R.index[min(len(R.index)-1, ROLL_WIN)], W.index.min())
    idx = R.loc[start_dt:].index

    ws, rets, prev = [], [], None
    cols = R.columns

    for dt in idx:
        # get raw model weights for this date (or last known)
        try:
            wraw = W.loc[dt]
        except KeyError:
            # back-fill with most recent available weights before dt
            wraw = W.loc[:dt].tail(1).squeeze()
            if wraw is None or (isinstance(wraw, pd.Series) and wraw.empty):
                # if truly nothing yet, use equal-weight as a neutral placeholder
                wraw = pd.Series(np.ones(len(cols))/len(cols), index=cols)
        # ensure we have the same asset order
        wraw = pd.Series({c: float(wraw[c]) if c in wraw.index and pd.notna(wraw[c]) else 0.0 for c in cols})
        # project to feasible set
        w = proj_caps_simplex(wraw.values, cols)
        tw, cost = turnover_cost(prev, w)
        pr = float(R.loc[dt].values @ w) - cost
        prev = w
        ws.append(pd.Series(w, index=cols, name=dt))
        rets.append(pr)

    return pd.DataFrame(ws), pd.Series(rets, index=idx)

# ===== Main =====
def main():
    R = weekly_returns()
    F = load_mgarch_feats().reindex(R.index).ffill()

    # 1) Baselines
    w_static, r_static = static_mvo(R)
    w_mg,     r_mg     = mgarch_mvo(R, F)

    # 2) Model strategies (pull from experiments/)
    exps = []
    # returns-only if you ran Step 1:
    exps += [("LSTM (returns only)",      "sec10_returns_lstm"),
             ("Transformer (returns only)","sec10_returns_transformer")]
    # §7 runs:
    exps += [("LSTM (+MGARCH feats)"     , "mgarch_lstm"),
             ("Transformer (+MGARCH feats)", "mgarch_transformer"),
             ("LSTM (+vol-credit feats)" , "volcredit_lstm"),
             ("Transformer (+vol-credit feats)","volcredit_transformer")]

    Wdict = load_model_weights(exps)

    # 3) Build realized series with caps + costs
    series = {
        "Static MVO": (w_static, r_static),
        "MGARCH-MVO": (w_mg,     r_mg),
    }
    for name, _exp in exps:
        if name in Wdict:
            w, r = model_weights_port(R, Wdict[name])
            series[name] = (w, r)

    # 4) Save weights, returns, metrics
    metrics = []
    rets_table = {}
    for name, (w, r) in series.items():
        w.to_csv(OUT/f"sec10_weights_{name.replace(' ','_').replace('/','-')}.csv")
        rets_table[name] = r
        m = ann_metrics(r)
        m["Model"] = name
        metrics.append(m)

    pd.DataFrame(rets_table).to_csv(OUT/"sec10_port_rets.csv")
    pd.DataFrame(metrics).set_index("Model").to_csv(OUT/"sec10_metrics.csv")

    # 5) Plots
    # full-sample equity
    plt.figure(figsize=(10,6))
    for name, (_, r) in series.items():
        cw = np.exp(r.cumsum())
        plt.plot(cw.index, cw.values, label=name)
    plt.title("§10 Walk-forward: cumulative wealth (weekly, caps+costs)")
    plt.xlabel("Date"); plt.ylabel("Cumulative wealth"); plt.legend(ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig(FIG/"sec10_cumret.png", dpi=150)

    # Sharpe & MaxDD bars
    dfm = pd.DataFrame(metrics).set_index("Model")
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    dfm[["Sharpe"]].sort_values("Sharpe").plot(kind="barh", ax=ax[0]); ax[0].set_title("Sharpe (test)")
    dfm[["MaxDD"]].sort_values("MaxDD").plot(kind="barh", ax=ax[1]); ax[1].set_title("Max drawdown")
    for a in ax: a.set_xlabel("")
    plt.tight_layout(); plt.savefig(FIG/"sec10_sharpe_maxdd.png", dpi=150)

    # Stress slices (skip gracefully if out of range)
    stress = [
        ("GFC 2007–2009", "2007-07-01", "2009-06-30", "sec10_stress_2008.png"),
        ("COVID 2020",    "2020-02-01", "2020-06-30", "sec10_stress_2020.png"),
    ]
    for title, t0, t1, fn in stress:
        plt.figure(figsize=(9,5))
        have = False
        for name, (_, r) in series.items():
            s = r.loc[t0:t1]
            if s.empty: continue
            cw = np.exp(s.cumsum())
            plt.plot(cw.index, cw.values, label=name); have = True
        if have:
            plt.title(f"§10 Stress window: {title}")
            plt.xlabel("Date"); plt.ylabel("Cumulative wealth"); plt.legend(ncol=2, fontsize=8)
            plt.tight_layout(); plt.savefig(FIG/fn, dpi=150)
    print("[ok] wrote:",
          OUT/"sec10_metrics.csv",
          OUT/"sec10_port_rets.csv",
          FIG/"sec10_cumret.png",
          FIG/"sec10_sharpe_maxdd.png")
if __name__ == "__main__":
    main()
