#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

def _read_csv_robust(fp: Path) -> pd.DataFrame:
    """
    Robust CSV reader:
      - If the first row starts with 'Ticker', skip it (yfinance sometimes writes a label row)
      - Otherwise read normally.
    """
    with fp.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    if first.startswith("Ticker"):
        df = pd.read_csv(fp, skiprows=1)
    else:
        df = pd.read_csv(fp)
    return df

def load_from_csv(ticker: str) -> pd.DataFrame:
    """
    Load a single-ticker CSV from data/ with columns: Date, Adj Close, Volume (others ignored).
    Returns a DataFrame indexed by Date with columns [Adj Close, Volume].
    """
    fp = DATA / f"{ticker}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing CSV for {ticker} at {fp}. Provide CSVs or run --fetch.")

    df = _read_csv_robust(fp)

    # Normalize date/index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    else:
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
        df = df.dropna(subset=[first_col]).sort_values(first_col).set_index(first_col)
        df.index.name = "Date"

    # Drop stray columns
    for junk in ["Ticker", "Symbols"]:
        if junk in df.columns:
            df = df.drop(columns=[junk])

    # Keep only needed columns
    cols = [c for c in ["Adj Close", "Volume"] if c in df.columns]
    if "Adj Close" not in cols:
        raise ValueError(f"{fp} must include an 'Adj Close' column; got {df.columns.tolist()}")
    return df[cols]

def fetch_with_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Optional helper that requires internet + yfinance."""
    import yfinance as yf  # type: ignore
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    df = df.rename(columns={"Adj Close": "Adj Close", "Volume": "Volume"})
    df = df[["Adj Close", "Volume"]]
    df.index.name = "Date"
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="Fetch data via yfinance into data/")
    ap.add_argument("--config", type=str, default=str(ROOT / "config.json"))
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    tickers = cfg["tickers"]
    start, end = cfg["start_date"], cfg["end_date"]

    DATA.mkdir(exist_ok=True, parents=True)
    OUT.mkdir(exist_ok=True, parents=True)

    panel = {}
    for t in tickers:
        if args.fetch:
            try:
                df = fetch_with_yfinance(t, start, end)
                df.reset_index().to_csv(DATA / f"{t}.csv", index=False)
                print(f"[fetch] saved {t} -> {DATA/f'{t}.csv'}")
            except Exception as e:
                print(f"[warn] fetch failed for {t}: {e}")
        try:
            df = load_from_csv(t)
            panel[t] = df
        except Exception as e:
            print(f"[warn] skipping {t}: {e}")

    if not panel:
        print("[error] No tickers loaded. Provide CSVs in data/ or use --fetch.")
        sys.exit(1)

    # Align by date and write parquet (MultiIndex columns: (ticker, field))
    all_df = pd.concat({k: v for k, v in panel.items()}, axis=1)
    all_df.sort_index(inplace=True)
    out_path = OUT / "prices.parquet"
    all_df.to_parquet(out_path)
    print(f"[ok] wrote {out_path} with shape {all_df.shape}")

if __name__ == "__main__":
    main()
