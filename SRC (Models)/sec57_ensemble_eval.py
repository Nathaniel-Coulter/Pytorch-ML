#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"

def load_prices():
    prices = pd.read_parquet(OUT / "prices.parquet")
    px = prices.xs("Adj Close", level=1, axis=1)

    # Coerce numerics & clean before resampling
    px = px.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    px = px.asfreq("B").ffill().resample("W-FRI").last()
    rets = np.log(px / px.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return px, rets

def detect_date_column(df: pd.DataFrame) -> str | None:
    # common names first
    for c in df.columns:
        if c.lower() in ("date", "dt", "timestamp"):
            return c
    # common unlabeled index column written by pandas to_csv
    for c in df.columns:
        if c.lower().startswith("unnamed"):
            return c
    # fallback: try first column if it parses to many datetimes
    first = df.columns[0]
    try:
        dt = pd.to_datetime(df[first], errors="coerce")
        if dt.notna().mean() > 0.6:
            return first
    except Exception:
        pass
    return None

def load_weights(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = detect_date_column(df)
    if date_col is None:
        raise ValueError(f"Could not find a date-like column in {path}. "
                         f"Columns: {df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    # Coerce weight columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0.0)

    # Renormalize long-only weights per week
    w = df.abs()
    denom = w.sum(axis=1).replace(0, np.nan)
    w = w.div(denom, axis=0).fillna(0.0)
    return w

def align(w: pd.DataFrame, rets: pd.DataFrame):
    # weights are end-of-week; apply previous weights to current returns
    idx = rets.index.intersection(w.index)
    w = w.reindex(idx).ffill().fillna(0.0)
    pr = (w.shift(1) * rets.reindex(idx)).sum(axis=1).fillna(0.0)
    return pr, w.loc[idx]

def metrics(r: pd.Series):
    ann = r.mean() * 52
    vol = r.std(ddof=0) * np.sqrt(52)
    sharpe = ann / vol if vol > 0 else np.nan
    cw = np.exp(r.cumsum())
    peak = cw.cummax()
    dd = (cw / peak - 1.0).min()
    return dict(AnnRet=float(ann), Vol=float(vol), Sharpe=float(sharpe), MaxDD=float(dd))

def turnover(w: pd.DataFrame):
    tw = 0.5 * (w.diff().abs().sum(axis=1)).fillna(0.0)
    return float(tw.mean() * 52)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w_return", default=str(OUT / "weights_transformer_return.csv"))
    ap.add_argument("--w_risk",   default=str(OUT / "weights_transformer_risk.csv"))
    ap.add_argument("--w_dd",     default=str(OUT / "weights_transformer_dd.csv"))
    args = ap.parse_args()

    for p in [args.w_return, args.w_risk, args.w_dd]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing weights file: {p}")

    FIG.mkdir(exist_ok=True, parents=True)
    OUT.mkdir(exist_ok=True, parents=True)

    px, rets = load_prices()
    w_ret = load_weights(args.w_return)
    w_rsk = load_weights(args.w_risk)
    w_dd  = load_weights(args.w_dd)

    w_ens = (w_ret + w_rsk + w_dd) / 3.0
    w_ens = w_ens.div(w_ens.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    series = {}
    for k, w in {"return": w_ret, "risk": w_rsk, "dd": w_dd, "ensemble": w_ens}.items():
        r, w_al = align(w, rets)
        series[k] = dict(ret=r, w=w_al)

    rows = []
    for k in series:
        m = metrics(series[k]["ret"])
        m["Turnover"] = turnover(series[k]["w"])
        m["Model"] = k
        rows.append(m)
    dfm = pd.DataFrame(rows).set_index("Model")
    dfm.to_csv(OUT / "sec57_metrics.csv")

    # cumulative wealth
    plt.figure(figsize=(10, 6))
    for k in ["return", "risk", "dd", "ensemble"]:
        cw = np.exp(series[k]["ret"].cumsum())
        plt.plot(cw.index, cw.values, label=k)
    plt.title("§5.7 Specialists vs Ensemble — Cumulative wealth (test)")
    plt.xlabel("Date"); plt.ylabel("Cumulative wealth")
    plt.legend()
    plt.tight_layout(); plt.savefig(FIG / "sec57_cumret.png", dpi=150)

    # metrics bars
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    dfm[["AnnRet", "Sharpe", "Vol"]].plot(kind="bar", ax=axs[0])
    axs[0].set_title("Returns/Sharpe/Vol")
    dfm[["MaxDD"]].plot(kind="bar", ax=axs[1], legend=False)
    axs[1].set_title("Max drawdown")
    dfm[["Turnover"]].plot(kind="bar", ax=axs[2], legend=False)
    axs[2].set_title("Turnover (annualized)")
    for ax in axs: ax.set_xlabel("")
    plt.tight_layout(); plt.savefig(FIG / "sec57_metrics.png", dpi=150)

    print("[ok] wrote:",
          OUT / "sec57_metrics.csv",
          FIG / "sec57_cumret.png",
          FIG / "sec57_metrics.png")

if __name__ == "__main__":
    main()
