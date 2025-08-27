#!/usr/bin/env python3
"""
Generate Section 5.5 visuals: metrics bars, cumulative returns, rolling Sharpe, weights bars.
Usage:
  python src/sec5_make_figs.py \
    --base experiments/sec5_base_nomacro \
    --macro experiments/sec5_plus_macro \
    --outdir figures
"""

import argparse
from pathlib import Path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def find_col(df: pd.DataFrame, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def load_metrics(run_dir: Path) -> pd.DataFrame:
    """Load sec5_results.csv (one or two rows: LSTM/Transformer)."""
    f = run_dir / "sec5_results.csv"
    df = read_csv_safe(f)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Try to find model column
    model_col = None
    for c in df.columns:
        if c.lower() in ("model","arch","architecture","net","algo"):
            model_col = c; break
    if model_col is None:
        # If no model column, assume index differentiates or there are exactly two rows in fixed order
        df.insert(0, "model", ["lstm","transformer"][:len(df)])
        model_col = "model"
    # Normalize to lowercase labels
    df[model_col] = df[model_col].astype(str).str.lower().str.replace("rnn","lstm")
    return df.rename(columns={model_col: "model"})

def load_weekly_returns(run_dir: Path, model: str) -> pd.DataFrame:
    """Load weekly returns CSV for a model and return DataFrame with Date + ret."""
    fname = f"weekly_returns_{model}.csv"
    f = run_dir / fname
    df = read_csv_safe(f)
    # Detect date column
    date_col = find_col(df, ["Date","date","timestamp"])
    if date_col is None:
        # fallback: first column
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # Detect returns column
    ret_col = None
    for name in ["ret","return","weekly_return","portfolio_ret","strategy_ret","r","weekly"]:
        c = find_col(df, [name])
        if c is not None:
            ret_col = c; break
    if ret_col is None:
        # Heuristic: first numeric column excluding date
        numeric = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            raise ValueError(f"Could not find a return column in {f} (columns={df.columns.tolist()})")
        ret_col = numeric[0]

    out = df[[date_col, ret_col]].rename(columns={date_col:"Date", ret_col:"ret"})
    out["ret"] = pd.to_numeric(out["ret"], errors="coerce")
    out = out.dropna(subset=["ret"])
    return out

def load_weights(run_dir: Path, model: str) -> pd.DataFrame:
    """Load weights_{model}.csv and return numeric columns only (assets)."""
    f = run_dir / f"weights_{model}.csv"
    df = read_csv_safe(f)
    # Find a date/time column if present
    date_col = find_col(df, ["Date","date","timestamp"])
    if date_col is not None:
        df = df.drop(columns=[date_col])
    # Keep numeric asset columns
    keep = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not keep:
        # If nothing numeric, maybe the rest needs coercion
        df = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
        keep = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    out = df[keep].copy()
    # Drop all-NaN cols
    out = out.loc[:, out.notna().any(axis=0)]
    return out

def cum_wealth(ret: pd.Series) -> pd.Series:
    return (1.0 + ret.fillna(0)).cumprod()

def rolling_sharpe(ret: pd.Series, window=26, periods_per_year=52) -> pd.Series:
    r = ret.rolling(window)
    mu = r.mean()
    sd = r.std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.sqrt(periods_per_year) * (mu / sd.replace(0,np.nan))
    return s

def bar_compare_metric(compare_df: pd.DataFrame, metric: str, outpath: Path):
    """
    compare_df index: ['lstm','transformer'], columns: ['baseline','macro'] for the metric value
    """
    fig = plt.figure(figsize=(7,5))
    X = np.arange(len(compare_df.index))
    width = 0.35
    # two bars per model: baseline, macro
    plt.bar(X - width/2, compare_df["baseline"].values, width, label="baseline")
    plt.bar(X + width/2, compare_df["macro"].values, width, label="+macro")
    plt.xticks(X, [m.upper() for m in compare_df.index])
    plt.ylabel(metric)
    plt.title(f"{metric} (test)")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path to baseline run directory (e.g., experiments/sec5_base_nomacro)")
    ap.add_argument("--macro", required=True, help="Path to +macro run directory (e.g., experiments/sec5_plus_macro)")
    ap.add_argument("--outdir", default="figures", help="Where to write figures")
    args = ap.parse_args()

    base_dir = Path(args.base)
    macro_dir = Path(args.macro)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)

    # --- Load metrics ---
    base_metrics = load_metrics(base_dir)
    macro_metrics = load_metrics(macro_dir)

    # Normalize metric column names
    def get_metric(df, name_candidates):
        for name in name_candidates:
            if name in df.columns:
                return df[name]
            # case-insensitive match
            for c in df.columns:
                if c.lower() == name.lower():
                    return df[c]
        # try partial
        for c in df.columns:
            if any(name.lower() in c.lower() for name in name_candidates):
                return df[c]
        return None

    # Build compare DF for Sharpe, AnnRet, Vol
    models = ["lstm","transformer"]
    compare = {}
    for metric_key, candidates in {
        "Sharpe": ["Sharpe"],
        "AnnRet": ["AnnRet","AnnualizedReturn","AnnReturn","AnnRet_%","AnnRetPct"],
        "Vol": ["Vol","AnnualizedVolatility","AnnVol"]
    }.items():
        rows = []
        for m in models:
            # filter by model row (best-effort)
            b_row = base_metrics[base_metrics["model"].str.contains(m)].copy()
            p_row = macro_metrics[macro_metrics["model"].str.contains(m)].copy()
            b_val = None
            p_val = None
            if not b_row.empty:
                s = get_metric(b_row, candidates)
                if s is not None: b_val = float(s.iloc[0])
            if not p_row.empty:
                s = get_metric(p_row, candidates)
                if s is not None: p_val = float(s.iloc[0])
            rows.append((m, b_val, p_val))
        dfm = pd.DataFrame(rows, columns=["model","baseline","macro"]).set_index("model")
        compare[metric_key] = dfm

    # Write combined comparison CSV
    combo = pd.concat({k:v for k,v in compare.items()}, axis=1)
    combo.to_csv("outputs/sec55_compare.csv")

    # --- Metrics bar plots ---
    bar_compare_metric(compare["Sharpe"], "Sharpe", outdir/"sec55_metrics_sharpe.png")
    bar_compare_metric(compare["AnnRet"], "Annualized return", outdir/"sec55_metrics_annret.png")
    bar_compare_metric(compare["Vol"], "Annualized volatility", outdir/"sec55_metrics_vol.png")

    # --- Time series (cumret + rolling sharpe) ---
    # Load all four return series
    r_base_lstm  = load_weekly_returns(base_dir, "lstm")
    r_base_trf   = load_weekly_returns(base_dir, "transformer")
    r_macro_lstm = load_weekly_returns(macro_dir, "lstm")
    r_macro_trf  = load_weekly_returns(macro_dir, "transformer")

    # Align on common dates across all
    df_ret = (
        r_base_lstm.rename(columns={"ret":"ret_base_lstm"})
        .merge(r_base_trf.rename(columns={"ret":"ret_base_trf"}), on="Date", how="inner")
        .merge(r_macro_lstm.rename(columns={"ret":"ret_macro_lstm"}), on="Date", how="inner")
        .merge(r_macro_trf.rename(columns={"ret":"ret_macro_trf"}), on="Date", how="inner")
        .sort_values("Date")
    )

    w_base_lstm  = cum_wealth(df_ret["ret_base_lstm"])
    w_base_trf   = cum_wealth(df_ret["ret_base_trf"])
    w_macro_lstm = cum_wealth(df_ret["ret_macro_lstm"])
    w_macro_trf  = cum_wealth(df_ret["ret_macro_trf"])

    fig = plt.figure(figsize=(8,5))
    plt.plot(df_ret["Date"], w_base_lstm,  label="LSTM baseline")
    plt.plot(df_ret["Date"], w_macro_lstm, label="LSTM +macro")
    plt.plot(df_ret["Date"], w_base_trf,   label="Transformer baseline")
    plt.plot(df_ret["Date"], w_macro_trf,  label="Transformer +macro")
    plt.ylabel("Cumulative wealth (test)")
    plt.title("Section 5.5 — Cumulative returns (test)")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir/"sec55_cumret.png", dpi=200)
    plt.close(fig)

    rs_base_lstm  = rolling_sharpe(df_ret["ret_base_lstm"])
    rs_macro_lstm = rolling_sharpe(df_ret["ret_macro_lstm"])
    rs_base_trf   = rolling_sharpe(df_ret["ret_base_trf"])
    rs_macro_trf  = rolling_sharpe(df_ret["ret_macro_trf"])

    fig = plt.figure(figsize=(8,5))
    plt.plot(df_ret["Date"], rs_base_lstm,  label="LSTM baseline")
    plt.plot(df_ret["Date"], rs_macro_lstm, label="LSTM +macro")
    plt.plot(df_ret["Date"], rs_base_trf,   label="Transformer baseline")
    plt.plot(df_ret["Date"], rs_macro_trf,  label="Transformer +macro")
    plt.ylabel("Rolling Sharpe (26w)")
    plt.title("Section 5.5 — Rolling Sharpe (test)")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir/"sec55_rollsharpe.png", dpi=200)
    plt.close(fig)

    # --- Weights bars: mean absolute weights by asset ---
    def mean_abs_weights(run_dir, model):
        W = load_weights(run_dir, model)
        # Average absolute weights across time
        return W.abs().mean(axis=0)

    maw_base_lstm  = mean_abs_weights(base_dir,  "lstm")
    maw_macro_lstm = mean_abs_weights(macro_dir, "lstm")
    assets_lstm = sorted(set(maw_base_lstm.index) | set(maw_macro_lstm.index))
    base_vals = [maw_base_lstm.get(a, np.nan) for a in assets_lstm]
    macro_vals= [maw_macro_lstm.get(a, np.nan) for a in assets_lstm]

    fig = plt.figure(figsize=(10,5))
    X = np.arange(len(assets_lstm))
    width = 0.4
    plt.bar(X - width/2, base_vals,  width, label="baseline")
    plt.bar(X + width/2, macro_vals, width, label="+macro")
    plt.xticks(X, assets_lstm, rotation=45, ha="right")
    plt.ylabel("Mean |weight|")
    plt.title("Section 5.5 — LSTM mean |weights| by asset (test)")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir/"sec55_weights_lstm.png", dpi=200)
    plt.close(fig)

    maw_base_trf  = mean_abs_weights(base_dir,  "transformer")
    maw_macro_trf = mean_abs_weights(macro_dir, "transformer")
    assets_trf = sorted(set(maw_base_trf.index) | set(maw_macro_trf.index))
    base_vals = [maw_base_trf.get(a, np.nan) for a in assets_trf]
    macro_vals= [maw_macro_trf.get(a, np.nan) for a in assets_trf]

    fig = plt.figure(figsize=(10,5))
    X = np.arange(len(assets_trf))
    width = 0.4
    plt.bar(X - width/2, base_vals,  width, label="baseline")
    plt.bar(X + width/2, macro_vals, width, label="+macro")
    plt.xticks(X, assets_trf, rotation=45, ha="right")
    plt.ylabel("Mean |weight|")
    plt.title("Section 5.5 — Transformer mean |weights| by asset (test)")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir/"sec55_weights_transformer.png", dpi=200)
    plt.close(fig)

    print("[ok] Wrote figures:",
          (outdir/"sec55_metrics_sharpe.png"),
          (outdir/"sec55_metrics_annret.png"),
          (outdir/"sec55_metrics_vol.png"),
          (outdir/"sec55_cumret.png"),
          (outdir/"sec55_rollsharpe.png"),
          (outdir/"sec55_weights_lstm.png"),
          (outdir/"sec55_weights_transformer.png"))
    print("[ok] Wrote table: outputs/sec55_compare.csv")

if __name__ == "__main__":
    main()
