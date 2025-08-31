#!/usr/bin/env python3
# §5.6 RL + Tsallis entropy visuals:
# - Sharpe vs entropy (both models)
# - Cumulative wealth overlays (LSTM panel, Transformer panel)

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXP  = ROOT / "experiments"
OUT  = ROOT / "outputs"
FIG  = ROOT / "figures"

def parse_tag(p: Path):
    """
    Expect folders like: sec56_{model}_L{lookback}_ent{lambda}
    e.g., sec56_lstm_L126_ent0.003
    """
    m = re.match(r"sec56_(lstm|transformer)_L(\d+)_ent([\d\.]+)", p.name)
    if not m:
        return None
    model, L, ent = m.group(1), int(m.group(2)), float(m.group(3))
    return model, L, ent

def pick_ret_col(df: pd.DataFrame):
    # Prefer common names; otherwise pick the last numeric col
    priorities = ["ret", "return", "strategy_return", "pnl", "strategy"]
    cols = [c for c in df.columns if c.lower() != "date"]
    for pref in priorities:
        for c in cols:
            if pref == c.lower():
                return c
    # fallback: last numeric
    nums = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not nums:
        raise ValueError(f"No numeric return column found in {df.columns.tolist()}")
    return nums[-1]

def load_weekly_returns(file: Path) -> pd.Series:
    df = pd.read_csv(file)
    # handle Date
    date_col = None
    for c in df.columns:
        if c.lower() in ("date","dt","timestamp"):
            date_col = c; break
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    else:
        # make a fake weekly index if missing, still works for metrics
        df.index = pd.RangeIndex(len(df))
    # pick return column
    rcol = pick_ret_col(df)
    r = pd.to_numeric(df[rcol], errors="coerce").fillna(0.0)
    # Ensure weekly frequency if Date exists
    if isinstance(r.index, pd.DatetimeIndex):
        r = r.asfreq("W-FRI", method="pad").fillna(0.0)
    return r

def ann_metrics(r: pd.Series):
    # Assume weekly frequency
    ann = r.mean()*52
    vol = r.std(ddof=0)*np.sqrt(52)
    sharpe = ann/vol if vol>0 else np.nan
    cw = np.exp(r.cumsum())
    peak = cw.cummax()
    maxdd = (cw/peak - 1.0).min()
    return dict(AnnRet=float(ann), Vol=float(vol), Sharpe=float(sharpe), MaxDD=float(maxdd))

def gather_runs():
    rows = []
    curves = {"lstm":{}, "transformer":{}}
    for d in sorted(EXP.glob("sec56_*")):
        if not d.is_dir():
            continue
        parsed = parse_tag(d)
        if not parsed:
            continue
        model, L, ent = parsed
        # pick correct file name pattern
        rr = list(d.glob(f"weekly_returns_{model}.csv"))
        if not rr:
            # some users copy outputs/ first; also check outputs folder directly
            rr = list((ROOT/"outputs").glob(f"weekly_returns_{model}.csv"))
        if not rr:
            print(f"[warn] no weekly returns for {d.name}")
            continue
        r = load_weekly_returns(rr[0])
        m = ann_metrics(r)
        rows.append(dict(Model=model, Lookback=L, Entropy=ent, **m))
        curves[model][ent] = r
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No sec56_* runs found with weekly returns.")
    return df.sort_values(["Model","Entropy"]), curves

def plot_sharpe(df: pd.DataFrame, path: Path):
    plt.figure(figsize=(8,5))
    for model, g in df.groupby("Model"):
        g = g.sort_values("Entropy")
        plt.plot(g["Entropy"], g["Sharpe"], marker="o", label=model)
    plt.title("§5.6 Sharpe vs. entropy strength (λ)")
    plt.xlabel("Entropy λ")
    plt.ylabel("Test Sharpe")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)

def plot_cum_panels(curves, path: Path):
    fig, axs = plt.subplots(1,2, figsize=(12,5), sharey=True)
    for ax, model in zip(axs, ["lstm","transformer"]):
        if not curves[model]:
            ax.set_visible(False); continue
        for ent, r in sorted(curves[model].items(), key=lambda kv: kv[0]):
            cw = np.exp(r.cumsum())
            ax.plot(cw.index, cw.values, label=f"λ={ent:g}")
        ax.set_title(f"{model.upper()}: cumulative wealth (test)")
        ax.set_xlabel("Date"); ax.set_ylabel("Cumulative wealth")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("§5.6 Entropy sweep — cumulative wealth overlays")
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(path, dpi=150)

def main():
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df, curves = gather_runs()
    # Save tidy metrics table
    df.to_csv(OUT/"sec56_metrics_by_entropy.csv", index=False)
    # Plots
    plot_sharpe(df, FIG/"sec56_sharpe_vs_entropy.png")
    plot_cum_panels(curves, FIG/"sec56_cumret_panels.png")
    print("[ok] wrote:",
          OUT/"sec56_metrics_by_entropy.csv",
          FIG/"sec56_sharpe_vs_entropy.png",
          FIG/"sec56_cumret_panels.png")

if __name__ == "__main__":
    main()
