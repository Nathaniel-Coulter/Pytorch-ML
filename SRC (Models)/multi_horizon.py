#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Repro
np.random.seed(42)
torch.manual_seed(42)

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------- Data -----------------
def load_ew_returns():
    # Read the panel and slice out Adj Close across tickers
    prices = pd.read_parquet(OUT / "prices.parquet")

    # Slice to Adj Close (MultiIndex level=1) and coerce to numeric
    px = prices.xs("Adj Close", level=1, axis=1)
    px = px.apply(pd.to_numeric, errors="coerce")

    # Clean up index/order and remove all-NaN rows if any
    px = px.sort_index().dropna(how="all")

    # Robust log-returns (ignore NaNs, drop inf)
    rets = np.log(px / px.shift(1))
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    # Equal-weight daily return across available ETFs each day
    ew = rets.mean(axis=1, skipna=True)
    ew.name = "ew_ret"
    return ew

def make_supervised(ew, L=126, horizons=(1,5,21,63)):
    X, Y, dates = [], [], []
    arr = ew.values
    for t in range(L, len(arr) - max(horizons)):
        seq = arr[t-L:t]                              # last L daily returns
        ys  = [arr[t:t+h].sum() for h in horizons]    # forward cum log-returns
        X.append(seq); Y.append(ys); dates.append(ew.index[t])
    X = np.array(X, dtype=np.float32)                 # [N,L]
    Y = np.array(Y, dtype=np.float32)                 # [N,4]
    dates = pd.Index(dates)
    return X, Y, dates

def time_splits(dates):
    # 2006–2014 / 2015–2018 / 2019–2025
    tr = dates < pd.Timestamp("2015-01-01")
    va = (dates >= pd.Timestamp("2015-01-01")) & (dates < pd.Timestamp("2019-01-01"))
    te = dates >= pd.Timestamp("2019-01-01")
    return tr, va, te

# ----------------- Models -----------------
class LSTM_MultiHead(nn.Module):
    def __init__(self, input_dim=1, hidden=64, num_layers=1, horizons=4, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden, horizons)  # 4 heads in one linear

    def forward(self, x):
        # x: [B, L] -> [B, L, 1]
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        hT = out[:, -1, :]
        return self.head(hT)

def train_lstm(Xtr, Ytr, Xva, Yva, epochs=25, lr=1e-3, batch=128, grad_clip=1.0, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = LSTM_MultiHead()
    model.to(device)
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float()),
        batch_size=batch, shuffle=True
    )
    vaX = torch.from_numpy(Xva).float().to(device)
    vaY = torch.from_numpy(Yva).float().to(device)

    best = {"val": float("inf"), "state": None}
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(vaX), vaY).item()
        if val_loss < best["val"]:
            best = {"val": val_loss, "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
        print(f"[LSTM] epoch {ep+1:02d}  val_mse={val_loss:.6f}")
    model.load_state_dict(best["state"])
    return model

def ridge_per_horizon(Xtr, Ytr, Xva, Yva):
    alphas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    best = []
    for h in range(Ytr.shape[1]):
        ytr, yva = Ytr[:, h], Yva[:, h]
        sc_best, a_best = -1e9, None
        for a in alphas:
            m = Ridge(alpha=a, fit_intercept=True)
            m.fit(Xtr, ytr)
            pred = m.predict(Xva)
            sc = r2_score(yva, pred)
            if sc > sc_best:
                sc_best, a_best = sc, a
        best.append(a_best)
    # fit final on train+val
    Xtv = np.vstack([Xtr, Xva])
    Ytv = np.vstack([Ytr, Yva])
    models = []
    for h, a in enumerate(best):
        m = Ridge(alpha=a, fit_intercept=True)
        m.fit(Xtv, Ytv[:, h])
        models.append(m)
    return models

def eval_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    ic = spearmanr(y_true, y_pred).statistic
    return r2, ic

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=126)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    ew = load_ew_returns()
    X, Y, dates = make_supervised(ew, L=args.L)
    tr, va, te = time_splits(dates)

    # Standardize using train only
    mu = X[tr].mean(axis=0, keepdims=True)
    sd = X[tr].std(axis=0, keepdims=True) + 1e-8
    X = (X - mu) / sd

    Xtr, Ytr = X[tr], Y[tr]
    Xva, Yva = X[va], Y[va]
    Xte, Yte = X[te], Y[te]

    # Baseline: Ridge per horizon
    ridge_models = ridge_per_horizon(Xtr, Ytr, Xva, Yva)
    ridge_preds = np.column_stack([ridge_models[h].predict(Xte) for h in range(Y.shape[1])])

    # LSTM multi-head
    lstm = train_lstm(Xtr, Ytr, Xva, Yva,
                      epochs=args.epochs, lr=args.lr, batch=args.batch,
                      grad_clip=1.0, seed=42)
    lstm.eval()
    with torch.no_grad():
        lstm_preds = lstm(torch.from_numpy(Xte).float()).cpu().numpy()

    horizons = [1, 5, 21, 63]
    rows = []
    for j, h in enumerate(horizons):
        r2_r, ic_r = eval_metrics(Yte[:, j], ridge_preds[:, j])
        r2_l, ic_l = eval_metrics(Yte[:, j], lstm_preds[:, j])
        rows.append({"horizon": h, "model": "Ridge", "R2": r2_r, "IC": ic_r})
        rows.append({"horizon": h, "model": "LSTM",  "R2": r2_l, "IC": ic_l})
        print(f"h={h:>3}  Ridge: R2={r2_r:.4f} IC={ic_r:.4f} | LSTM: R2={r2_l:.4f} IC={ic_l:.4f}")

    df = pd.DataFrame(rows)
    csv_path = OUT / "mh_results.csv"
    df.to_csv(csv_path, index=False)

    # --- R2 bar ---
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    hs = np.array(horizons)
    rid = [df[(df.horizon==h) & (df.model=="Ridge")]["R2"].values[0] for h in hs]
    lst = [df[(df.horizon==h) & (df.model=="LSTM")]["R2"].values[0]  for h in hs]
    ax.bar(hs - width/2, rid, width, label="Ridge")
    ax.bar(hs + width/2, lst, width, label="LSTM")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Out-of-sample $R^2$")
    ax.set_title("Multi-horizon OOS $R^2$ (EW portfolio returns)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mh_r2_bar.png", dpi=160)

    # --- IC bar ---
    fig, ax = plt.subplots(figsize=(7, 4))
    rid = [df[(df.horizon==h) & (df.model=="Ridge")]["IC"].values[0] for h in hs]
    lst = [df[(df.horizon==h) & (df.model=="LSTM")]["IC"].values[0]  for h in hs]
    ax.bar(hs - width/2, rid, width, label="Ridge")
    ax.bar(hs + width/2, lst, width, label="LSTM")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Spearman IC")
    ax.set_title("Multi-horizon OOS IC (EW portfolio returns)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mh_ic_bar.png", dpi=160)

    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote {OUT/'mh_r2_bar.png'}")
    print(f"[ok] wrote {OUT/'mh_ic_bar.png'}")

if __name__ == "__main__":
    main()
