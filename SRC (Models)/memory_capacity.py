#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# -------------------------
# Data
# -------------------------
def gen_batch(n, m=3, pad=50, vocab=8, seed=None):
    """
    Copy/Repeat task:
      input  = [prompt length m][junk length pad], tokens in {1..vocab}
      target = prompt (length m)
    Exact-sequence accuracy is success only if all m tokens match.
    """
    rng = np.random.default_rng(seed)
    prompt = rng.integers(1, vocab + 1, size=(n, m))
    junk   = rng.integers(1, vocab + 1, size=(n, pad))
    X = np.concatenate([prompt, junk], axis=1)
    Y = prompt.copy()
    return X, Y

def make_loaders(pad, m=3, vocab=8, n_train=20000, n_val=4000, batch=128):
    Xtr, Ytr = gen_batch(n_train, m=m, pad=pad, vocab=vocab, seed=123)
    Xva, Yva = gen_batch(n_val,   m=m, pad=pad, vocab=vocab, seed=456)
    Xtr = torch.tensor(Xtr, dtype=torch.long)
    Ytr = torch.tensor(Ytr, dtype=torch.long)
    Xva = torch.tensor(Xva, dtype=torch.long)
    Yva = torch.tensor(Yva, dtype=torch.long)
    train = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch, shuffle=True,  drop_last=False)
    val   = DataLoader(TensorDataset(Xva, Yva), batch_size=batch, shuffle=False, drop_last=False)
    return train, val

# -------------------------
# Models
# -------------------------
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, emb=16, hidden=64, out_len=3):
        super().__init__()
        self.emb   = nn.Embedding(vocab_size + 1, emb)  # tokens 0..vocab (we never use 0)
        self.rnn   = nn.RNN(emb, hidden, batch_first=True)
        self.head  = nn.Linear(hidden, out_len * (vocab_size + 1))
        self.vocab = vocab_size + 1
        self.out_len = out_len

    def forward(self, x):
        e = self.emb(x.long())         # [B, T, E]
        out, _ = self.rnn(e)           # [B, T, H]
        hT = out[:, -1, :]             # [B, H]
        logits = self.head(hT).view(-1, self.out_len, self.vocab)  # [B, m, V]
        return logits

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, emb=16, hidden=64, out_len=3):
        super().__init__()
        self.emb   = nn.Embedding(vocab_size + 1, emb)
        self.lstm  = nn.LSTM(emb, hidden, batch_first=True)
        self.head  = nn.Linear(hidden, out_len * (vocab_size + 1))
        self.vocab = vocab_size + 1
        self.out_len = out_len

    def forward(self, x):
        e = self.emb(x.long())
        out, _ = self.lstm(e)
        hT = out[:, -1, :]
        logits = self.head(hT).view(-1, self.out_len, self.vocab)
        return logits

# -------------------------
# Train / Eval
# -------------------------
def exact_seq_acc(logits, y):
    """Exact match across all m positions."""
    pred = logits.argmax(-1)                 # [B, m]
    return (pred == y).all(dim=1).float().mean().item()

def train_eval(model, train_loader, val_loader, epochs=20, lr=1e-3, grad_clip=1.0, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    hist = {"val_seq_acc": []}
    best = {"val": -1.0, "state": None}   # maximize val_seq_acc

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)  # [B, m, V]
            # average CE across m positions
            loss = sum(crit(logits[:, t, :], yb[:, t]) for t in range(yb.size(1))) / yb.size(1)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        # validation
        model.eval()
        accs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                accs.append(exact_seq_acc(logits, yb))
        val_acc = float(np.mean(accs))
        hist["val_seq_acc"].append(val_acc)
        print(f"  epoch {ep+1:02d}  val_seq_acc={val_acc:.3f}")

        if val_acc > best["val"]:
            best = {"val": val_acc, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    # load best-by-accuracy
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return hist

# -------------------------
# Run + Plot
# -------------------------
def run(pads=(10, 50, 100, 200, 400), m=3, vocab=8, epochs=20, batch=128, hidden=64, lr=1e-3):
    device = torch.device("cpu")
    results = []
    chance = 1.0 / (vocab ** m)

    for pad in pads:
        print(f"\n=== padding={pad} ===")
        train_loader, val_loader = make_loaders(pad=pad, m=m, vocab=vocab, n_train=20000, n_val=4000, batch=batch)

        # RNN
        print("[RNN]")
        rnn = SimpleRNN(vocab_size=vocab, hidden=hidden, out_len=m).to(device)
        rnn_hist = train_eval(rnn, train_loader, val_loader, epochs=epochs, lr=lr, device=device)
        rnn_seq_acc = rnn_hist["val_seq_acc"][-1]
        print(f"padding={pad:>4}  RNN seq-acc={rnn_seq_acc:.3f}")
        results.append({"pad": int(pad), "model": "RNN", "seq_acc": float(rnn_seq_acc)})

        # LSTM
        print("[LSTM]")
        lstm = SimpleLSTM(vocab_size=vocab, hidden=hidden, out_len=m).to(device)
        lstm_hist = train_eval(lstm, train_loader, val_loader, epochs=epochs, lr=lr, device=device)
        lstm_seq_acc = lstm_hist["val_seq_acc"][-1]
        print(f"padding={pad:>4}  LSTM seq-acc={lstm_seq_acc:.3f}")
        results.append({"pad": int(pad), "model": "LSTM", "seq_acc": float(lstm_seq_acc)})

    # results → CSV
    df = pd.DataFrame(results)
    df["pad"] = pd.to_numeric(df["pad"], errors="coerce").astype("Int64")
    df["model"] = df["model"].astype(str)

    csv_path = OUT / "copy_repeat_results.csv"
    df.to_csv(csv_path, index=False)

    # robust bar plot
    pads_sorted = sorted(int(p) for p in pads)
    idx = pd.MultiIndex.from_product([pads_sorted, ["RNN", "LSTM"]], names=["pad", "model"])
    pivot = (
        df.set_index(["pad", "model"])
          .reindex(idx)
          .assign(seq_acc=lambda x: x["seq_acc"].fillna(chance))
          .reset_index()
          .pivot(index="pad", columns="model", values="seq_acc")
    )

    ax = pivot.plot(kind="bar", figsize=(7, 4))
    ax.set_ylabel("Exact sequence accuracy")
    ax.set_xlabel("Padding length")
    ax.set_title(f"Copy/Repeat Memory Capacity (m={m}, vocab={vocab})")
    ax.axhline(chance, linestyle="--", linewidth=1)
    ax.legend(title="Model")
    plt.tight_layout()
    bar_path = OUT / "copy_repeat_bar.png"
    plt.savefig(bar_path, dpi=160)
    plt.close()

    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote {bar_path}")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pads", nargs="+", type=int, default=[10, 50, 100, 200, 400])
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--vocab", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    run(
        pads=tuple(args.pads),
        m=args.m,
        vocab=args.vocab,
        epochs=args.epochs,
        batch=args.batch,
        hidden=args.hidden,
        lr=args.lr,
    )
