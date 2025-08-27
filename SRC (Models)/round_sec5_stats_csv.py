import pandas as pd
from pathlib import Path

src = Path("/home/snowden/Desktop/quant_portfolio_scaffold/figures/sec5_stats_table.csv")
dst = Path("/home/snowden/Desktop/quant_portfolio_scaffold/figures/sec5_stats_table_rounded.csv")

df = pd.read_csv(src)

# Choose which columns to round (all numeric by default)
for col in df.columns:
    # try to coerce to numeric; if numeric, round to 3 decimals
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().any():
        df[col] = s.round(3).astype(str)  # keep as string to avoid scientific notation surprises

df.to_csv(dst, index=False)
print(f"Wrote {dst}")
