#!/usr/bin/env python3
import pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXP  = ROOT / "experiments"
OUT  = ROOT / "outputs"
FIG  = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("volcredit","lstm"),
    ("volcredit","transformer"),
    ("mgarch","lstm"),
    ("mgarch","transformer"),
]

def read_weekly_returns_csv(path: Path) -> pd.Series:
    """
    Robust loader for weekly_returns_<model>.csv that tolerates:
    - index saved as first column (no 'Date' header)
    - 'Unnamed: 0' or 'index' headers
    - single unnamed return column
    Returns a pandas Series with DatetimeIndex.
    """
    df = pd.read_csv(path)
    # figure out date column
    date_col = None
    for cand in ["Date","date","Index","index","Unnamed: 0"]:
        if cand in df.columns:
            date_col = cand; break
    if date_col is None:
        # assume first column holds dates
        date_col = df.columns[0]
    # coerce to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    # figure out returns column (prefer 'ret' if present)
    ret_col = None
    for cand in ["ret","returns","weekly_return","value"]:
        if cand in df.columns:
            ret_col = cand; break
    if ret_col is None:
        # else, pick the first non-empty numeric column
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not nums:
            raise ValueError(f"No numeric return column found in {path}")
        ret_col = nums[0]
    s = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    s = s.asfreq("W-FRI")  # align weekly
    return s.rename("ret")

def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    ann = r.mean()*52
    vol = r.std(ddof=0)*np.sqrt(52)
    shp = ann/vol if vol>0 else np.nan
    cw  = np.exp(r.cumsum())
    mdd = (cw/cw.cummax() - 1.0).min()
    return dict(AnnRet=float(ann), Vol=float(vol), Sharpe=float(shp), MaxDD=float(mdd))

def load_curve(expdir: Path) -> pd.Series:
    # try canonical file; else scan for the first matching pattern
    candidates = [
        expdir / "weekly_returns_transformer.csv",
        expdir / "weekly_returns_lstm.csv",
    ]
    f = None
    for c in candidates:
        if c.exists(): f = c; break
    if f is None:
        # fallback: first csv starting with 'weekly_returns'
        picks = sorted(expdir.glob("weekly_returns*.csv"))
        if not picks:
            raise FileNotFoundError(f"No weekly_returns*.csv in {expdir}")
        f = picks[0]
    return read_weekly_returns_csv(f)

def main():
    rows = []
    curves = {}

    for tag, model in PAIRS:
        expdir = EXP / f"{tag}_{model}"
        if not expdir.exists():
            print(f"[warn] missing {expdir}")
            continue
        r = load_curve(expdir)
        # restrict to test range if your §5 split is implicit; here we keep all
        rows.append(dict(Model=model.capitalize(), FeatureSet=tag, **metrics(r)))
        curves[f"{tag}_{model}"] = r

    # save metrics table
    dfm = pd.DataFrame(rows)
    dfm.to_csv(OUT/"sec7_metrics.csv", index=False)

    # plot cumulative wealth panels
    plt.figure(figsize=(10,6))
    for k, s in curves.items():
        cw = np.exp(s.cumsum())
        plt.plot(cw.index, cw.values, label=k.replace("_"," / "))
    plt.title("§7 Cumulative wealth — models × feature sets")
    plt.xlabel("Date"); plt.ylabel("Cumulative wealth"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG/"sec7_cumret_panels.png", dpi=150)

    # bar: Sharpe by feature set & model
    if not dfm.empty:
        piv = dfm.pivot_table(index="Model", columns="FeatureSet", values="Sharpe")
        plt.figure(figsize=(7,5))
        piv.plot(kind="bar")
        plt.title("§7 Test Sharpe by feature set")
        plt.ylabel("Sharpe"); plt.tight_layout()
        plt.savefig(FIG/"sec7_sharpe_by_feature.png", dpi=150)

    print("[ok] wrote:",
          OUT/"sec7_metrics.csv",
          FIG/"sec7_cumret_panels.png",
          FIG/"sec7_sharpe_by_feature.png")

if __name__ == "__main__":
    main()
