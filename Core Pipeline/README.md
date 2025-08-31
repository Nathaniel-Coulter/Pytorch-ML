# Quant Portfolio Scaffold

This scaffold sets up a single dataset + pipeline for:
- **Section "Version 2"**: classical MPT / shifting efficient frontier visualization
- **Later sections**: attention-based transformer allocator, RL (Tsallis), multi-agent heads, NEAT search

## Universe (default)
SPY, QQQ, IWM, EFA, EEM, TLT, IEF, LQD, HYG, GLD, DBC, VNQ (daily → weekly rebal)

## Quick Start
1) (Optional) Place cached CSVs in `data/` (one file per ticker, columns must include `Date, Adj Close, Volume`).
2) OR run the fetcher (requires internet + yfinance):
   ```bash
   pip install yfinance
   python src/loader.py --fetch
   ```
3) Build features:
   ```bash
   python src/features.py --build
   ```
4) Run MPT demo (creates efficient frontier plot and a shifting-frontier toy example):
   ```bash
   python src/mpt_demo.py
   ```

Artifacts are written to `outputs/`.

## Notes
- Internet access is **not** required if you provide CSVs. The code will happily load from `data/`.
- Rebalancing is weekly by default; adjust in `config.json`.
- Transformer/RL code can attach to the same `panel.parquet` built by `features.py`.
