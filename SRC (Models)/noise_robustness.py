#!/usr/bin/env python3
# src/noise_robustness.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# I/O
# -------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# -------------------------
# Data
# -------------------------
def generate_noisy_data(n_samples: int, seq_len: int, sigma: float, device: torch.device):
    """
    Latent binary sequence z_t ~ Bernoulli(0.5).
    Observed x_t = z_t + epsilon_t, epsilon_t ~ N(0, sigma^2).
    Label y = z_0 (first latent token).
    """
    z = torch.randint(0, 2, (n_samples, seq_len), device=device, dtype=torch.float32)
    if sigma > 0:
        noise = torch.randn(n_samples, seq_len, device=device) * sigma
        x = z + noise
    else:
        x = z.clone()
    y = z[:, 0].long()  # target is the first latent token
    return x.cpu(), y.cpu()

def make_loaders(n_train: int, n_test: int, seq_len: int, sigma: float, batch: int, num_workers: int = 2):
    device = torch.device("cpu")
    Xtr, ytr = generate_noisy_data(n_train, seq_len, sigma, device)
    Xte, yte = generate_noisy_data(n_test,  seq_len, sigma, device)

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=False  # set True if you ever move tensors to GPU
    )
    test_loader = DataLoader(
        TensorDataset(Xte, yte),
        batch_size=batch,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=False
    )
    return train_loader, test_loader
# -------------------------
# Models
# -------------------------
class SimpleRNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_classes=2):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = x.unsqueeze(-1)  # [B,T,1]
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_classes=2, forget_bias=1.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # set forget bias
        for name, param in self.lstm.named_parameters():
            if "bias_ih" in name:
                # PyTorch LSTM bias layout: [b_ii|b_if|b_ig|b_io]
                H = hidden_dim
                with torch.no_grad():
                    param[H:2*H].add_(forget_bias)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TinyTransformer(nn.Module):
    """
    Small Transformer encoder (optional) for robustness contrast.
    Uses 1-2 layers, 4 heads, d_model=64. Causal mask so it can't peek ahead.
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_classes=2):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = None  # learnable positional bias not critical here
        self.fc  = nn.Linear(d_model, num_classes)

    def _causal_mask(self, T, device):
        # (T,T) mask with True where we should mask (upper triangle)
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x):
        # x: [B,T]
        B, T = x.shape
        h = self.proj(x.unsqueeze(-1))  # [B,T,1] -> [B,T,d]
        mask = self._causal_mask(T, x.device)
        h = self.enc(h, mask=mask)
        # take last token
        return self.fc(h[:, -1, :])

# -------------------------
# Train / Eval
# -------------------------
def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)

def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total   += yb.size(0)
    return correct / total

def train_model(model, train_loader, test_loader, epochs=20, lr=1e-3, grad_clip=1.0, device=torch.device("cpu")):
    model.to(device)
    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.parameters(), lr=lr)
    hist = {"test_acc": []}
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        te_acc = accuracy(model, test_loader, device)
        hist["test_acc"].append(te_acc)
    return hist

# -------------------------
# Robustness summaries
# -------------------------
def summarize_robustness(df: pd.DataFrame, failure_threshold=0.6):
    """
    df columns: L, sigma, model, seed, test_acc, final_acc (optional)
    Returns per-model-per-L: slope (acc vs sigma), sigma_star (first sigma < threshold), auc_sigma.
    """
    rows = []
    for (model, L), g in df.groupby(["model", "L"]):
        g = g.sort_values("sigma")
        # mean across seeds at each sigma
        agg = g.groupby("sigma")["final_acc"].mean().reset_index()
        xs = agg["sigma"].values
        ys = agg["final_acc"].values
        # slope via simple linear fit
        if len(xs) >= 2:
            A = np.vstack([xs, np.ones_like(xs)]).T
            b, a = np.linalg.lstsq(A, ys, rcond=None)[0]  # ys ≈ b*x + a
            slope = float(b)
        else:
            slope = np.nan
        # sigma* (first sigma where acc < threshold); if none, put NaN
        sigma_star = np.nan
        for s_val, acc_val in zip(xs, ys):
            if acc_val < failure_threshold:
                sigma_star = float(s_val); break
        # AUC over sigma (trapezoid)
        auc = float(trapezoid(ys, xs)) if len(xs) >= 2 else np.nan
        rows.append({"model": model, "L": int(L), "slope": slope, "sigma_star": sigma_star, "auc_sigma": auc})
    return pd.DataFrame(rows)

# -------------------------
# Plotting
# -------------------------
def plot_curves_by_length(df, sigmas, L, out_png):
    """
    For a fixed L, plot accuracy vs sigma for each model (mean across seeds).
    """
    plt.figure(figsize=(7,4))
    sub = df[df["L"] == L]
    for model, g in sub.groupby("model"):
        agg = g.groupby("sigma")["final_acc"].mean().reindex(sigmas).reset_index()
        plt.plot(agg["sigma"], agg["final_acc"], marker="o", label=model)
    plt.xlabel("Noise σ")
    plt.ylabel("Test accuracy")
    plt.title(f"Noise robustness @ sequence length L={L}")
    plt.ylim(0.45, 1.01)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_heatmap(df, sigmas, seq_lens, model, out_png):
    """
    Heatmap of accuracy for one model: rows=L, cols=sigma.
    """
    pivot = (
        df[df["model"] == model]
          .groupby(["L","sigma"])["final_acc"].mean()
          .unstack("sigma")
          .reindex(index=seq_lens, columns=sigmas)
    )
    plt.figure(figsize=(1.1*len(sigmas)+2, 0.7*len(seq_lens)+2))
    plt.imshow(pivot.values, aspect="auto", origin="lower", vmin=0.5, vmax=1.0)
    plt.colorbar(label="Accuracy")
    plt.xticks(ticks=range(len(sigmas)), labels=[f"{s:.2f}" for s in sigmas], rotation=0)
    plt.yticks(ticks=range(len(seq_lens)), labels=[str(L) for L in seq_lens])
    plt.xlabel("Noise σ"); plt.ylabel("Sequence length L")
    plt.title(f"Accuracy heatmap — {model}")
    # annotate
    for i in range(len(seq_lens)):
        for j in range(len(sigmas)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_robustness_bars(summary_df, metric, out_png):
    """
    metric in {"slope","sigma_star","auc_sigma"}
    """
    models = summary_df["model"].unique().tolist()
    Ls = sorted(summary_df["L"].unique().tolist())
    width = 0.8 / len(models)
    plt.figure(figsize=(8,4))
    for i, model in enumerate(models):
        sub = summary_df[summary_df["model"] == model].set_index("L").reindex(Ls)
        xs = np.arange(len(Ls)) + i*width
        plt.bar(xs, sub[metric].values, width=width, label=model)
    plt.xticks(np.arange(len(Ls)) + width*(len(models)-1)/2, [str(L) for L in Ls])
    plt.xlabel("Sequence length L")
    label = {
        "slope": "d(accuracy)/dσ (more negative = worse)",
        "sigma_star": "σ* at failure threshold (higher = more robust)",
        "auc_sigma": "AUC over σ (higher = more robust)"
    }[metric]
    plt.ylabel(label)
    plt.title(f"Noise robustness summary — {metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# -------------------------
# Sweep
# -------------------------
def run_sweep(
    seq_lens, sigmas, seeds=3, epochs=20, n_train=20000, n_test=4000, batch=128, lr=1e-3,
    grad_clip=1.0, forget_bias=1.0, with_transformer=False
):
    device = torch.device("cpu")
    rows = []

    for L in seq_lens:
        print(f"\n=== Sequence length L={L} ===")
        for sigma in sigmas:
            print(f"  -- sigma={sigma:.2f}")
            # fresh data loaders per (L, sigma)
            train_loader, test_loader = make_loaders(n_train, n_test, L, sigma, batch)

            for s in range(seeds):
                set_seed(1000 + s)

                # RNN
                rnn = SimpleRNN(hidden_dim=64)
                hist_rnn = train_model(rnn, train_loader, test_loader, epochs=epochs, lr=lr, grad_clip=grad_clip, device=device)
                rows.append({
                    "model": "RNN", "L": int(L), "sigma": float(sigma), "seed": int(s),
                    "final_acc": float(hist_rnn["test_acc"][-1])
                })

                # LSTM
                set_seed(2000 + s)
                lstm = SimpleLSTM(hidden_dim=64, forget_bias=forget_bias)
                hist_lstm = train_model(lstm, train_loader, test_loader, epochs=epochs, lr=lr, grad_clip=grad_clip, device=device)
                rows.append({
                    "model": "LSTM", "L": int(L), "sigma": float(sigma), "seed": int(s),
                    "final_acc": float(hist_lstm["test_acc"][-1])
                })

                # Optional Transformer
                if with_transformer:
                    set_seed(3000 + s)
                    tr = TinyTransformer(d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_classes=2)
                    hist_tr = train_model(tr, train_loader, test_loader, epochs=epochs, lr=lr, grad_clip=grad_clip, device=device)
                    rows.append({
                        "model": "Transformer", "L": int(L), "sigma": float(sigma), "seed": int(s),
                        "final_acc": float(hist_tr["test_acc"][-1])
                    })

    df = pd.DataFrame(rows)
    return df

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="+", type=int, default=[50, 100, 200, 500], help="sequence lengths")
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0], help="noise std dev grid")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--samples", type=int, default=20000, help="train samples per (L, sigma)")
    ap.add_argument("--test_samples", type=int, default=4000, help="test samples per (L, sigma)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--forget_bias", type=float, default=1.0)
    ap.add_argument("--with_transformer", action="store_true", help="also train a tiny Transformer")
    args = ap.parse_args()

    df = run_sweep(
        seq_lens=args.seq,
        sigmas=args.sigmas,
        seeds=args.seeds,
        epochs=args.epochs,
        n_train=args.samples,
        n_test=args.test_samples,
        batch=args.batch,
        lr=args.lr,
        grad_clip=args.grad_clip,
        forget_bias=args.forget_bias,
        with_transformer=args.with_transformer
    )

    # Save raw results
    csv_path = OUT / "noise_robustness_results.csv"
    df.to_csv(csv_path, index=False)

    # Per-length curves
    for L in args.seq:
        plot_curves_by_length(df, sigmas=args.sigmas, L=L, out_png=OUT / f"noise_acc_vs_sigma_L{L}.png")

    # Heatmaps
    for model in df["model"].unique():
        plot_heatmap(df, sigmas=args.sigmas, seq_lens=args.seq, model=model, out_png=OUT / f"noise_heatmap_{model}.png")

    # Summaries
    summary = summarize_robustness(df, failure_threshold=0.6)
    sum_csv = OUT / "noise_robustness_summary.csv"
    summary.to_csv(sum_csv, index=False)

    # Bar charts for robustness metrics
    for metric in ["slope", "sigma_star", "auc_sigma"]:
        plot_robustness_bars(summary, metric, out_png=OUT / f"noise_robustness_bars_{metric}.png")

    # JSON handy blob
    with open(OUT / "noise_robustness_raw.json", "w") as f:
        json.dump({
            "args": vars(args),
            "head": df.head(10).to_dict(orient="records")
        }, f, indent=2)

    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote {sum_csv}")
    print(f"[ok] wrote plots to {OUT}")

if __name__ == "__main__":
    main()
