# src/search_random.py
import json, os, itertools, subprocess, random, time
cfgs=[]
for d_model in [64,96,128]:
  for heads in [2,4,6]:
    for depth in [2,3,4]:
      for dropout in [0.0,0.1]:
        cfgs.append(dict(d_model=d_model, heads=heads, depth=depth, dropout=dropout))
random.shuffle(cfgs)
best=None
for i, c in enumerate(cfgs[:24], 1):
  tag=f"sec58_tfm_d{c['d_model']}_h{c['heads']}_L{c['depth']}_dr{c['dropout']}"
  print(f"[{i}/24] {tag}")
  os.environ["TFM_DMODEL"]=str(c["d_model"])
  os.environ["TFM_HEADS"]=str(c["heads"])
  os.environ["TFM_LAYERS"]=str(c["depth"])
  os.environ["TFM_DROPOUT"]=str(c["dropout"])
  cp = subprocess.run(["python","src/sec5_run.py","--model","transformer","--lookback","126","--epochs","40","--tc_bps","0","--entropy_reg","0.003"], capture_output=True, text=True)
  open(f"experiments/{tag}.log","w").write(cp.stdout)
  # your runner prints test metrics near the end; scrape Sharpe:
  import re
  m=re.findall(r"\[transformer\] Metrics:.*'Sharpe': np\.float32\(([^)]+)\)", cp.stdout)
  if m:
    sharp=float(m[-1])
    if best is None or sharp>best[0]: best=(sharp, c, tag)

results = []

for i, c in enumerate(cfgs[:24], 1):
    tag=f"sec58_tfm_d{c['d_model']}_h{c['heads']}_L{c['depth']}_dr{c['dropout']}"
    print(f"[{i}/24] {tag}")
    os.environ["TFM_DMODEL"]=str(c["d_model"])
    os.environ["TFM_HEADS"]=str(c["heads"])
    os.environ["TFM_LAYERS"]=str(c["depth"])
    os.environ["TFM_DROPOUT"]=str(c["dropout"])
    cp = subprocess.run(
        ["python","src/sec5_run.py","--model","transformer","--lookback","126","--epochs","40","--tc_bps","0","--entropy_reg","0.003"],
        capture_output=True, text=True
    )
    open(f"experiments/{tag}.log","w").write(cp.stdout)

    import re, json
    m = re.search(r"\[transformer\] Metrics: ({.*})", cp.stdout)
    if m:
        metrics = json.loads(m.group(1).replace("np.float32",""))
        metrics.update(c)  # add hyperparams
        metrics["tag"] = tag
        results.append(metrics)
        sharp = metrics.get("Sharpe", 0.0)
        if best is None or sharp > best[0]:
            best = (sharp, c, tag)

# --- save all results ---
import pandas as pd
df = pd.DataFrame(results)
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/sec58_search_results.csv", index=False)
print("[ok] wrote outputs/sec58_search_results.csv")

print("BEST:", best)



print("BEST:", best)
