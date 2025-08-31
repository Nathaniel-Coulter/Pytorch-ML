#!/usr/bin/env python3
from pathlib import Path
import re, ast, pandas as pd, numpy as np, matplotlib.pyplot as plt

EXP = Path("experiments")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
FIG = Path("figures"); FIG.mkdir(exist_ok=True)

rows = []
log_re = re.compile(r"sec58_tfm_d(?P<d>\d+)_h(?P<h>\d+)_L(?P<L>\d+)_dr(?P<dr>[0-9.]+)\.log$")
metrics_re = re.compile(r"\[transformer\] Metrics:\s*(\{.*\})")

for log in sorted(EXP.glob("sec58_tfm_*.log")):
    mtag = log_re.search(log.name)
    if not mtag:
        continue
    d_model = int(mtag.group("d"))
    heads = int(mtag.group("h"))
    depth = int(mtag.group("L"))
    dropout = float(mtag.group("dr"))

    txt = log.read_text()
    mm = metrics_re.findall(txt)
    if not mm:
        continue

    s = mm[-1]
    # turn "np.float32(0.1234)" → "0.1234" for literal_eval
    s = re.sub(r"np\.float32\(([^)]+)\)", r"\1", s)
    mdict = ast.literal_eval(s)

    rows.append({
        "tag": log.stem,
        "d_model": d_model, "heads": heads, "depth": depth, "dropout": dropout,
        "AnnRet": float(mdict.get("AnnRet", np.nan)),
        "Vol": float(mdict.get("Vol", np.nan)),
        "Sharpe": float(mdict.get("Sharpe", np.nan)),
        "MaxDD": float(mdict.get("MaxDD", np.nan)),
        "Turnover": float(mdict.get("Turnover", np.nan)),
    })

df = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
df.to_csv(OUT / "sec58_search_results.csv", index=False)
print("[ok] wrote", OUT / "sec58_search_results.csv")

# --- quick plots ---
# 1) top-10 bar by Sharpe
top = df.head(10).copy()
plt.figure(figsize=(10,5))
plt.bar(top["tag"], top["Sharpe"])
plt.xticks(rotation=45, ha="right")
plt.title("§5.8 Random search — top-10 Sharpe (test)")
plt.tight_layout()
plt.savefig(FIG / "sec58_top10_sharpe.png", dpi=150)

# 2) Sharpe vs (d_model, depth) as scatter
plt.figure(figsize=(7,5))
sizes = 50 + 50*df["heads"]  # encode heads as marker size
plt.scatter(df["d_model"], df["Sharpe"], s=sizes, alpha=0.8)
for L in sorted(df["depth"].unique()):
    sel = df["depth"]==L
    plt.scatter(df.loc[sel,"d_model"], df.loc[sel,"Sharpe"], s=10, alpha=0.0, label=f"depth={L}")
plt.xlabel("d_model"); plt.ylabel("Sharpe (test)")
plt.title("§5.8 Sharpe vs d_model (marker size ∝ heads)")
plt.tight_layout()
plt.savefig(FIG / "sec58_scatter_dmodel_sharpe.png", dpi=150)

print("[ok] wrote",
      FIG / "sec58_top10_sharpe.png",
      "and",
      FIG / "sec58_scatter_dmodel_sharpe.png")
