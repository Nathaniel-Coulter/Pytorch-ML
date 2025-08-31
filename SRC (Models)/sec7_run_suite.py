#!/usr/bin/env python3
import os, re, ast, shutil
from pathlib import Path
import numpy as np, pandas as pd
import subprocess
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT/"outputs", ROOT/"figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

ASSETS   = ["SPY","QQQ","IWM","EFA","EEM","TLT","IEF","LQD","HYG","GLD","DBC","VNQ"]
LOOKBACK = 126
EPOCHS   = 40
LAMBDA   = 0.003

# ---------- helpers ----------
def parse_metrics_from_stdout(text: str) -> dict:
    """
    Extract the Python dict after '<model>] Metrics:' from sec5_run.py stdout.
    Strip numpy scalar wrappers and literal_eval to a dict of plain floats.
    """
    m = re.search(r"\]\s*Metrics:\s*(\{.*\})", text, flags=re.S)
    if not m:
        raise ValueError("Could not locate metrics dict in stdout.")
    blob = m.group(1)
    # remove np.floatXX(...) wrappers
    blob = re.sub(r"np\.float\d+\(", "", blob)
    blob = blob.replace(")", "")
    d = ast.literal_eval(blob)
    # coerce numpy/ints/strings to floats where possible
    out = {}
    for k, v in d.items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = v
    return out

def load_weekly_returns():
    px = pd.read_parquet(OUT/"prices.parquet").xs("Adj Close", level=1, axis=1)
    px = px.apply(pd.to_numeric, errors="coerce")
    pxw = px.asfreq("B").ffill().resample("W-FRI").last().dropna(how="all")
    r = np.log(pxw/pxw.shift(1))
    # keep only columns we actually have
    cols = [c for c in ASSETS if c in r.columns]
    return r[cols].dropna(how="all")

def split_idx(idx, train_n=364, val_n=202):
    train = idx[:train_n]
    val   = idx[train_n:train_n+val_n]
    test  = idx[train_n+val_n:]
    return train, val, test

def standardize_by_train(df):
    """Lag by 1, then z-score by training stats only; hard clip ±5σ."""
    df = df.copy().shift(1)  # T-1 only
    trn, _, _ = split_idx(df.index)
    mu  = df.loc[trn].mean()
    sig = df.loc[trn].std(ddof=0).replace(0, np.nan)
    z = (df - mu) / (sig + 1e-8)
    return z.clip(-5, 5)

def ensure_panel_csv(z, name):
    fn = OUT/f"sec7_features_{name}.csv"
    df = z.reset_index().rename(columns={"index":"Date"})
    df.to_csv(fn, index=False)
    return fn

def run_model(tag, model, features_csv):
    """
    Call your training runner with env injection for a side panel.
    Assumes src/sec5_run.py reads SEC7_FEATURES_CSV if present.
    """
    env = os.environ.copy()
    env["SEC7_FEATURES_CSV"]  = str(features_csv)
    env["SEC7_FEATURES_NAME"] = tag
    env.setdefault("TFM_DMODEL", "96")
    env.setdefault("TFM_HEADS",  "4")
    env.setdefault("TFM_LAYERS", "3")
    env.setdefault("TFM_DROPOUT","0.1")

    cmd = [
        "python","src/sec5_run.py",
        "--model", model,
        "--lookback", str(LOOKBACK),
        "--epochs",   str(EPOCHS),
        "--tc_bps",   "0",
        "--entropy_reg", str(LAMBDA)
    ]
    cp = subprocess.run(cmd, env=env, capture_output=True, text=True)

    expdir = ROOT/"experiments"/f"{tag}_{model}"
    expdir.mkdir(parents=True, exist_ok=True)
    (expdir/"stdout.txt").write_text(cp.stdout)
    (expdir/"stderr.txt").write_text(cp.stderr)

    # scrape metrics robustly
    try:
        metrics = parse_metrics_from_stdout(cp.stdout)
    except Exception as e:
        raise RuntimeError(f"Failed to parse metrics for {tag}/{model}: {e}. See {expdir/'stdout.txt'}")

    metrics["Model"] = model
    metrics["FeatureSet"] = tag

    # copy model artifacts if present
    for base in ["weekly_returns", "weights"]:
        src = OUT/f"{base}_{model}.csv"
        if src.exists():
            shutil.copyfile(src, expdir/src.name)

    return metrics, expdir

# ---------- main ----------
def main():
    # load returns for index alignment
    R = load_weekly_returns()
    idx_all = R.index

    # raw panels produced by sec7_build_macro.py
    vc_raw = pd.read_parquet(OUT/"sec7_macro_volcredit_raw.parquet").reindex(idx_all).ffill()
    mg_raw = pd.read_parquet(OUT/"sec7_macro_mgarch_raw.parquet").reindex(idx_all).ffill()

    # lag + train-only standardize
    vc = standardize_by_train(vc_raw)
    mg = standardize_by_train(mg_raw)

    fn_vc = ensure_panel_csv(vc, "volcredit")
    fn_mg = ensure_panel_csv(mg, "mgarch")

    results = []
    for tag, fn in [("volcredit", fn_vc), ("mgarch", fn_mg)]:
        for model in ["lstm","transformer"]:
            print(f"[run] {tag} :: {model}")
            m, _ = run_model(tag, model, fn)
            results.append(m)

    df = pd.DataFrame(results)
    df.to_csv(OUT/"sec7_results_raw.csv", index=False)

    # quick Sharpe bar
    if not df.empty and {"Model","FeatureSet","Sharpe"}.issubset(df.columns):
        piv = df.pivot_table(index="Model", columns="FeatureSet", values="Sharpe")
        plt.figure(figsize=(6,5))
        piv.plot(kind="bar")
        plt.title("§7 Test Sharpe by feature set")
        plt.ylabel("Sharpe")
        plt.tight_layout()
        plt.savefig(FIG/"sec7_sharpe_by_feature.png", dpi=150)

    print("[ok] wrote", OUT/"sec7_results_raw.csv", "and plots under", FIG)

if __name__ == "__main__":
    main()
