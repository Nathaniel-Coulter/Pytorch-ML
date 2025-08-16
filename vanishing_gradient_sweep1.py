#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).parent.name == "experiments") else Path.cwd()
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# -------------------------
# Synthetic dataset
# -------------------------
def generate_data(n_samples: int, seq_len: int, device: torch.device):
    """
    Binary sequences of length seq_len, entries in {0,1}.
    Label = first token of the sequence (position 0).
    """
    X = torch.randint(low=0, high=2, size=(n_samples, seq_len), dtype=torch.float32, device=device)
    y = X[:, 0].long()
    return X.cpu(), y.cpu()  # keep CPU dataloaders (portable, small)

def make_loaders(n_train: int, n_test: int, seq_len: int, batch: int):
    device = torch.device("cpu")
    Xtr, ytr = generate_data(n_train, seq_len, device)
    Xte, yte = generate_data(n_test,  seq_len, device)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True, drop_last=False)
    test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=batch, shuffle=False, drop_last=False)
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
        x = x.unsqueeze(-1)  # [B, T, 1]
        out, _ = self.rnn(x)
        logits = self.fc(out[:, -1, :])
        return logits

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        return logits

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

def train_model(model, train_loader, test_loader, epochs=10, lr=1e-3, device=torch.device("cpu")):
    model.to(device)
    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.parameters(), lr=lr)
    hist = {"train_acc": [], "test_acc": []}
    for ep in range(epochs):
        model.train()
        correct, total = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total   += yb.size(0)
        tr_acc = correct / total
        te_acc = accuracy(model, test_loader, device)
        hist["train_acc"].append(tr_acc)
        hist["test_acc"].append(te_acc)
        print(f"  epoch {ep+1:02d}: train={tr_acc:.3f}  test={te_acc:.3f}")
    return hist

# -------------------------
# Experiment sweep
# -------------------------
def run_sweep(seq_lens, seeds=3, epochs=10, n_train=20000, n_test=4000, batch=128, lr=1e-3):
    device = torch.device("cpu")  # keep CPU-friendly
    results_rows = []
    curves = {}  # (model, L) -> list of mean test_acc per epoch

    for L in seq_lens:
        print(f"\n=== Sequence length L={L} ===")
        # accumulate per seed
        rnn_test_by_seed, lstm_test_by_seed = [], []
        rnn_curve_acc = np.zeros(epochs, dtype=float)
        lstm_curve_acc = np.zeros(epochs, dtype=float)

        # fixed loaders per length (new data each length)
        train_loader, test_loader = make_loaders(n_train, n_test, L, batch)

        for s in range(seeds):
            print(f" seed {s+1}/{seeds}")
            set_seed(1000 + s)
            # RNN
            rnn = SimpleRNN(hidden_dim=64)
            hist_rnn = train_model(rnn, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
            rnn_test_by_seed.append(hist_rnn["test_acc"][-1])
            rnn_curve_acc += np.array(hist_rnn["test_acc"])

            # LSTM
            set_seed(2000 + s)
            lstm = SimpleLSTM(hidden_dim=64)
            hist_lstm = train_model(lstm, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
            lstm_test_by_seed.append(hist_lstm["test_acc"][-1])
            lstm_curve_acc += np.array(hist_lstm["test_acc"])

        # averages
        rnn_mean = float(np.mean(rnn_test_by_seed))
        rnn_std  = float(np.std(rnn_test_by_seed))
        lstm_mean = float(np.mean(lstm_test_by_seed))
        lstm_std  = float(np.std(lstm_test_by_seed))

        curves[("RNN", L)]  = (rnn_curve_acc / seeds).tolist()
        curves[("LSTM", L)] = (lstm_curve_acc / seeds).tolist()

        results_rows.append({"seq_len": L, "model": "RNN",  "test_acc_mean": rnn_mean,  "test_acc_std": rnn_std})
        results_rows.append({"seq_len": L, "model": "LSTM", "test_acc_mean": lstm_mean, "test_acc_std": lstm_std})

    results = pd.DataFrame(results_rows).sort_values(["seq_len","model"])
    return results, curves

# -------------------------
# Plotting
# -------------------------
def plot_curves(curves, epochs, outpath_png):
    plt.figure(figsize=(8,5))
    for (model, L), accs in curves.items():
        plt.plot(range(1, epochs+1), accs, label=f"{model} (L={L})")
    plt.xlabel("Epoch"); plt.ylabel("Test Accuracy")
    plt.title("Vanishing Gradients: accuracy vs. epoch (by sequence length)")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(outpath_png, dpi=160)
    plt.close()

def plot_by_length(results_df, outpath_png):
    pivot = results_df.pivot(index="seq_len", columns="model", values="test_acc_mean")
    plt.figure(figsize=(6,4))
    for model in pivot.columns:
        plt.plot(pivot.index, pivot[model], marker="o", label=model)
    plt.xlabel("Sequence length (timesteps)")
    plt.ylabel("Final Test Accuracy")
    plt.title("Final accuracy vs. sequence length")
    plt.ylim(0.45, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath_png, dpi=160)
    plt.close()

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="+", type=int, default=[50, 100, 200, 500], help="sequence lengths to sweep")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--samples", type=int, default=20000, help="train samples per length")
    ap.add_argument("--test_samples", type=int, default=4000, help="test samples per length")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    results, curves = run_sweep(
        seq_lens=args.seq,
        seeds=args.seeds,
        epochs=args.epochs,
        n_train=args.samples,
        n_test=args.test_samples,
        batch=args.batch,
        lr=args.lr,
    )

    # save artifacts
    csv_path = OUT / "vanishing_gradients_results.csv"
    png_curves = OUT / "vanishing_gradients_curves.png"
    png_bylen  = OUT / "vanishing_gradients_by_len.png"
    json_path  = OUT / "vanishing_gradients_raw.json"

    results.to_csv(csv_path, index=False)
    plot_curves(curves, epochs=args.epochs, outpath_png=png_curves)
    plot_by_length(results, outpath_png=png_bylen)
    with open(json_path, "w") as f:
        json.dump(
            {f"{k[0]}_L{k[1]}": v for k, v in curves.items()},
            f, indent=2
        )

    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote {png_curves}")
    print(f"[ok] wrote {png_bylen}")
    print(f"[ok] wrote {json_path}")

if __name__ == "__main__":
    main()
[I] (.venv) snowden@arch ~/D/quant_portfolio_scaffold> 
