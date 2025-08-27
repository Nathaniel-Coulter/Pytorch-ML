#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------
# 1) Multi-timescale simulator
# ------------------------------
def simulate_multiscale(n=12000, T=200, A=1.0,
                        p_start=0.02, mean_len=10, phi=0.7, sigma_b=0.8,
                        sigma_eps=0.05, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    slow = A * np.sin(2 * np.pi * t / T)

    burst = np.zeros(n, dtype=np.float32)
    i = 0
    while i < n:
        if rng.random() < p_start:
            L = max(3, int(rng.normal(mean_len, 2)))
            u = 0.0
            for k in range(L):
                if i + k >= n: break
                u = phi * u + rng.normal(0.0, sigma_b)
                burst[i + k] = u
            i += L
        else:
            i += 1

    noise = rng.normal(0.0, sigma_eps, size=n).astype(np.float32)
    y = slow.astype(np.float32) + burst + noise
    return y, slow.astype(np.float32), burst

def make_supervised(y, slow, burst, L=128):
    X, Y, Y_s, Y_b = [], [], [], []
    for t in range(L, len(y)-1):
        X.append(y[t-L:t])
        Y.append(y[t])
        Y_s.append(slow[t])
        Y_b.append(burst[t])
    return (np.array(X, np.float32),
            np.array(Y, np.float32),
            np.array(Y_s, np.float32),
            np.array(Y_b, np.float32))

def splits(n, train=0.6, val=0.2):
    i_tr = int(n * train)
    i_va = int(n * (train + val))
    tr = np.arange(0, i_tr)
    va = np.arange(i_tr, i_va)
    te = np.arange(i_va, n)
    return tr, va, te

# ------------------------------
# 2) Models
# ------------------------------
class RNNReg(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.rnn = nn.RNN(1, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, 1)
    def forward(self, x):
        x = x.unsqueeze(-1)  # [B,L,1]
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

class LSTMReg(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

def train_best(model, Xtr, Ytr, Xva, Yva, epochs=25, batch=128, lr=1e-3, clip=1.0, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                           batch_size=batch, shuffle=True)

    best = {"val": 1e9, "state": None}
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            if clip: nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            vaX = torch.from_numpy(Xva).to(device)
            vaY = torch.from_numpy(Yva).to(device)
            val = crit(model(vaX), vaY).item()
        if val < best["val"]:
            best = {"val": val, "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
        print(f"[{model.__class__.__name__}] epoch {ep+1:02d}  val_mse={val:.6f}")
    model.load_state_dict(best["state"])
    return model

# ------------------------------
# 3) Run experiment
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    y, s, b = simulate_multiscale()
    X, Y, Yslow, Yburst = make_supervised(y, s, b, L=args.L)

    tr, va, te = splits(len(X))
    mu, sd = X[tr].mean(axis=0, keepdims=True), X[tr].std(axis=0, keepdims=True) + 1e-8
    X = (X - mu) / sd

    Xtr, Ytr = X[tr], Y[tr]; Xva, Yva = X[va], Y[va]; Xte, Yte = X[te], Y[te]
    S_te, B_te = Yslow[te], Yburst[te]

    # Train models
    rnn  = train_best(RNNReg(64),  Xtr, Ytr, Xva, Yva, epochs=args.epochs, batch=args.batch, lr=args.lr)
    lstm = train_best(LSTMReg(64), Xtr, Ytr, Xva, Yva, epochs=args.epochs, batch=args.batch, lr=args.lr)

    with torch.no_grad():
        teX = torch.from_numpy(Xte)
        yhat_rnn  = rnn(teX).numpy()
        yhat_lstm = lstm(teX).numpy()

    # R^2: total, slow, burst
    rows = []
    def add_row(name, yhat):
        rows.append({"model": name, "metric": "R2_total", "value": r2_score(Yte, yhat)})
        rows.append({"model": name, "metric": "R2_slow",  "value": r2_score(S_te, yhat)})
        rows.append({"model": name, "metric": "R2_burst", "value": r2_score(B_te, yhat)})
    add_row("RNN",  yhat_rnn)
    add_row("LSTM", yhat_lstm)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "multiscale_results.csv", index=False)

    # Bar chart
    pivot = df.pivot(index="metric", columns="model", values="value").loc[["R2_total","R2_slow","R2_burst"]]
    ax = pivot.plot(kind="bar", figsize=(7,4))
    ax.set_ylabel("Out-of-sample $R^2$")
    ax.set_xlabel("")
    ax.set_title("Multi-timescale OOS $R^2$ (total vs. components)")
    plt.tight_layout()
    plt.savefig(OUT / "multiscale_r2.png", dpi=160)
    plt.close()

    # Trajectory figure
    k0, k1 = 100, 500  # slice of test region
    t = np.arange(k0, k1)
    plt.figure(figsize=(9,4))
    plt.plot(t, Yte[k0:k1], label="True")
    plt.plot(t, yhat_rnn[k0:k1], label="RNN")
    plt.plot(t, yhat_lstm[k0:k1], label="LSTM")
    plt.xlabel("Test time index")
    plt.ylabel("Value")
    plt.title("Held-out trajectory: true vs. predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "multiscale_traj.png", dpi=160)
    plt.close()

    print(f"[ok] wrote {OUT/'multiscale_results.csv'}")
    print(f"[ok] wrote {OUT/'multiscale_r2.png'}")
    print(f"[ok] wrote {OUT/'multiscale_traj.png'}")

if __name__ == "__main__":
    main()
