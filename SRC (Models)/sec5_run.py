# sec5_run.py
import math, os, time, argparse, random, warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

# =========================
# Config
# =========================
@dataclass
class Config:
    tickers: tuple = ("SPY","QQQ","IWM","EFA","EEM","TLT","IEF","LQD","HYG","GLD","DBC","VNQ")
    start: str = "2006-01-03"
    end:   str = "2025-08-14"
    lookback: int = 252           # L (e.g., 126 or 252)
    rebalance: str = "W-FRI"      # weekly (Friday close)
    train_end: str = "2014-12-31"
    val_end:   str = "2018-12-31"
    risk_free_annual: float = 0.03
    tc_bps: float = 5.0           # one-way transaction costs in bps
    batch_size: int = 64
    epochs: int = 50
    lr: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 7             # early stopping on val Sharpe
    entropy_reg: float = 1e-3     # encourage diversification; 0 to disable
    seed: int = 1337
    results_csv: str = "sec5_results.csv"

def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# =========================
# Data & Features
# =========================
def load_prices(cfg: Config) -> pd.DataFrame:
    """
    Robust yfinance loader that returns adjusted close prices for cfg.tickers
    between cfg.start and cfg.end, regardless of yfinance's column layout.

    With auto_adjust=True, yfinance returns adjusted 'Close' and omits 'Adj Close'.
    For multiple tickers, columns are a MultiIndex (level 0 = field, level 1 = ticker).
    """
    df = yf.download(
        list(cfg.tickers),
        start=cfg.start,
        end=cfg.end,
        auto_adjust=True,     # already adjusted for splits/divs
        progress=False,
        group_by="column"     # ensures first level is OHLCV fields if MultiIndex
    )

    # If it's a Series (single ticker) -> make DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame(name="Close")

    # MultiIndex (field, ticker) -> select 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        # Common structure: first level = ['Open','High','Low','Close','Volume']
        if "Close" in df.columns.get_level_values(0):
            px = df["Close"].copy()
        elif "Adj Close" in df.columns.get_level_values(0):
            px = df["Adj Close"].copy()
        else:
            raise KeyError("Neither 'Close' nor 'Adj Close' present in yfinance DataFrame.")
        # Ensure columns are tickers in desired order
        px = px.reindex(columns=list(cfg.tickers))
    else:
        # Single-level columns: try Close then Adj Close
        if "Close" in df.columns:
            px = df["Close"].to_frame() if not isinstance(df["Close"], pd.Series) else df[["Close"]]
            # If multiple tickers were requested, yfinance sometimes returns columns per ticker directly
            # In that case, use df as-is.
            if set(cfg.tickers).issubset(set(df.columns)):
                px = df[list(cfg.tickers)]
        elif "Adj Close" in df.columns:
            px = df[list(cfg.tickers)] if set(cfg.tickers).issubset(set(df.columns)) else df[["Adj Close"]]
        else:
            # If df already looks like a price table with tickers as columns
            if set(cfg.tickers).issubset(set(df.columns)):
                px = df[list(cfg.tickers)]
            else:
                raise KeyError("Could not locate price columns; expected 'Close' or 'Adj Close'.")

    px = px.dropna(how="all")  # drop all-NaN rows
    # yfinance sometimes brings partial NaNs early in the series for newer ETFs
    px = px.dropna(axis=0, how="any")  # keep only rows where all 12 ETFs exist (coherent panel)
    return px

def to_logrets(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()

def weekly_rebalance_index(returns: pd.DataFrame, rule="W-FRI") -> pd.DatetimeIndex:
    # Rebalance at Friday close; targets are next week (Mon..Fri)
    return returns.resample(rule).last().index.intersection(returns.index)

def standardize_returns(train_ret: pd.DataFrame, full_ret: pd.DataFrame):
    mu = train_ret.mean()
    sigma = train_ret.std().replace(0.0, 1.0)
    z_full = (full_ret - mu) / sigma
    return z_full, mu, sigma

def build_samples(cfg: Config, zrets: pd.DataFrame, raw_rets: pd.DataFrame):
    idx_reb = weekly_rebalance_index(zrets, cfg.rebalance)
    # Keep only rebalance dates that have full lookback and at least one forward week
    # We'll build samples for all consecutive pairs (reb_d, next_reb_d)
    pairs = []
    for i in range(len(idx_reb)-1):
        d = idx_reb[i]
        d_next = idx_reb[i+1]
        # require full lookback window ending at d
        if zrets.index.get_indexer([d])[0] < cfg.lookback:
            continue
        pairs.append((d, d_next))
    # Split by dates
    train_pairs = [(d0,d1) for (d0,d1) in pairs if d0 <= pd.to_datetime(cfg.train_end)]
    val_pairs   = [(d0,d1) for (d0,d1) in pairs if (d0 > pd.to_datetime(cfg.train_end)) and (d0 <= pd.to_datetime(cfg.val_end))]
    test_pairs  = [(d0,d1) for (d0,d1) in pairs if d0 > pd.to_datetime(cfg.val_end)]

    def make_XY(pairs_list):
        X_list, Y_list, D_list = [], [], []
        for (d0, d1) in pairs_list:
            j = zrets.index.get_loc(d0)
            # features: last L z-scored returns up to d0 inclusive
            win = zrets.iloc[j-cfg.lookback+1:j+1]      # [L rows] x [N assets]
            # target: next-week simple returns for each asset: exp(sum logrets)) - 1
            # sum raw log returns from (j+1 .. j_next), then exp-1
            j_next = zrets.index.get_loc(d1)
            weekly_simple = np.exp(raw_rets.iloc[j+1:j_next+1].sum(axis=0)) - 1.0  # Series by asset
            X = win.values.T.astype(np.float32)  # [N, L]
            Y = weekly_simple.loc[zrets.columns].values.astype(np.float32)  # [N]
            X_list.append(X); Y_list.append(Y); D_list.append(d1)           # weight applies for next week
        if len(X_list)==0:
            return None
        X_arr = np.stack(X_list)      # [B, N, L]
        Y_arr = np.stack(Y_list)      # [B, N]
        D_arr = np.array(D_list)
        return X_arr, Y_arr, D_arr

    train = make_XY(train_pairs)
    val   = make_XY(val_pairs)
    test  = make_XY(test_pairs)
    return train, val, test, idx_reb

class SeqAllocDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)  # [B, N, L]
        self.Y = torch.from_numpy(Y)  # [B, N]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

# =========================
# Models (LSTM & Transformer)
# =========================
class LSTMAllocator(nn.Module):
    def __init__(self, n_assets: int, seq_len: int, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n = n_assets
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.asset_bias = nn.Parameter(torch.zeros(n_assets))
    def forward(self, x):                       # x: [B, N, L]
        B, N, L = x.shape
        x = x.reshape(B*N, L, 1)
        h, _ = self.lstm(x)                     # [B*N, L, H]
        h_last = h[:, -1, :]                    # [B*N, H]
        s = self.head(h_last).reshape(B, N) + self.asset_bias   # scores
        w = torch.softmax(s, dim=1)             # simplex, long-only
        return w                                 # [B, N]

class PosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, L, D]
    def forward(self, x):  # x: [B, L, D]
        return x + self.pe[:, :x.shape[1], :]

class TransformerAllocator(nn.Module):
    def __init__(self, n_assets: int, seq_len: int, d_model: int = 64, nhead: int = 4, depth: int = 2,
                 d_ff: int = 128, dropout: float = 0.1, cross_asset_attn: bool = True):
        super().__init__()
        self.n = n_assets
        self.inp = nn.Linear(1, d_model)
        self.pos = PosEnc(d_model, max_len=seq_len)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.cross_asset_attn = cross_asset_attn
        if cross_asset_attn:
            self.cross = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
            self.cross_ln = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.asset_bias = nn.Parameter(torch.zeros(n_assets))

    def forward(self, x):                        # x: [B, N, L]
        B, N, L = x.shape
        x = x.reshape(B*N, L, 1)
        h = self.inp(x)                          # [B*N, L, D]
        h = self.pos(h)
        # causal mask for time
        mask = torch.triu(torch.ones(L, L, device=h.device)==1, diagonal=1)
        h = self.enc(h, mask)                    # [B*N, L, D]
        z = h[:, -1, :].reshape(B, N, -1)        # pooled per asset: [B, N, D]
        if self.cross_asset_attn and N > 1:
            z2, _ = self.cross(z, z, z)          # cross-asset attention
            z = self.cross_ln(z + z2)
        s = self.head(z).squeeze(-1) + self.asset_bias   # [B, N]
        w = torch.softmax(s, dim=1)
        return w

# =========================
# Loss, Metrics, Eval
# =========================
def sharpe_proxy_loss(w, Y, eps=1e-6, entropy_reg=0.0):
    # w: [B,N], Y: [B,N] weekly simple returns for next period
    r_p = (w * Y).sum(dim=1)                    # [B]
    mu = r_p.mean()
    sd = r_p.std(unbiased=False)
    sharpe = mu / (sd + eps)
    loss = -sharpe
    if entropy_reg > 0:
        # encourage diversification: maximize entropy => subtract (-Σ w log w) with minus sign
        ent = -(w * (w.clamp_min(1e-8)).log()).sum(dim=1).mean()
        loss += -entropy_reg * ent
    return loss, sharpe.detach(), mu.detach(), sd.detach()

def compute_metrics(weekly_r, rf_annual=0.03, tc_series=None):
    # weekly_r: pd.Series of realized weekly portfolio returns (after costs if applied)
    rf_w = (1.0 + rf_annual)**(1/52) - 1.0
    ex_r = weekly_r - rf_w
    ann_ret = weekly_r.mean() * 52.0
    ann_vol = weekly_r.std(ddof=0) * math.sqrt(52.0)
    sharpe = (ex_r.mean() / (weekly_r.std(ddof=0) + 1e-12))
    # Sortino
    downside = weekly_r.copy()
    downside[downside>0] = 0.0
    sortino = (ex_r.mean() / ((downside.std(ddof=0) + 1e-12) * math.sqrt(52.0)))
    # Drawdown
    nav = (1.0 + weekly_r).cumprod()
    peak = nav.cummax()
    dd = (nav/peak - 1.0)
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd if max_dd != 0 else 1e-12)
    # Costs summary (bps average per week)
    if tc_series is not None and len(tc_series)==len(weekly_r):
        avg_cost_bps = (tc_series.mean() * 1e4)
    else:
        avg_cost_bps = np.nan
    return dict(AnnRet=ann_ret, Vol=ann_vol, Sharpe=sharpe, Sortino=sortino, MaxDD=max_dd, Calmar=calmar, AvgCost_bps=avg_cost_bps)

def turnover_and_costs(weights_df: pd.DataFrame, tc_bps: float):
    # weights_df indexed by rebalance dates, columns tickers; rows sum to 1
    w_prev = weights_df.shift(1).fillna(0.0)
    turnover = (weights_df - w_prev).abs().sum(axis=1)  # L1 turnover per period
    costs = turnover * (tc_bps / 1e4)                   # one-way bps
    effN = 1.0 / (weights_df.pow(2).sum(axis=1) + 1e-12)
    return turnover, costs, effN

# =========================
# Train / Validate / Test
# =========================
def train_model(model, cfg: Config, train_ds, val_ds):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = {"val_sharpe": -1e9, "state": None, "epoch": -1}
    patience = cfg.patience; bad = 0

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    for epoch in range(1, cfg.epochs+1):
        model.train()
        tr_loss = []; tr_sh = []
        for X,Y in train_loader:
            X = X.to(DEVICE); Y = Y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            w = model(X)
            loss, sh, mu, sd = sharpe_proxy_loss(w, Y, entropy_reg=cfg.entropy_reg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss.append(loss.item()); tr_sh.append(sh.item())
        # Validate (Sharpe)
        model.eval(); vals = []
        with torch.no_grad():
            for X,Y in val_loader:
                X = X.to(DEVICE); Y = Y.to(DEVICE)
                w = model(X)
                loss, sh, mu, sd = sharpe_proxy_loss(w, Y, entropy_reg=0.0)
                vals.append(sh.item())
        val_sh = float(np.mean(vals)) if vals else -1e9
        tqdm.write(f"Epoch {epoch:03d}  train_loss {np.mean(tr_loss):.4f}  val_sharpe {val_sh:.4f}")
        if val_sh > best["val_sharpe"] + 1e-6:
            best = {"val_sharpe": val_sh, "state": {k:v.cpu() for k,v in model.state_dict().items()}, "epoch": epoch}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                tqdm.write(f"Early stop at epoch {epoch}. Best val Sharpe={best['val_sharpe']:.4f} @ {best['epoch']}")
                break
    # Load best
    model.load_state_dict(best["state"])
    return model, best

def evaluate_model(model, cfg: Config, test_X, test_Y, test_dates, tickers):
    model.eval()
    with torch.no_grad():
        W = []
        for i in range(0, len(test_X), 512):
            Xb = torch.from_numpy(test_X[i:i+512]).to(DEVICE)
            Wb = model(Xb).cpu().numpy()
            W.append(Wb)
        W = np.vstack(W)   # [T, N]
    weights = pd.DataFrame(W, index=pd.to_datetime(test_dates), columns=tickers)
    # Realized weekly returns by period from test_Y
    weekly_r = pd.Series((weights.values * test_Y).sum(axis=1), index=weights.index)
    # Turnover & costs
    turnover, costs, effN = turnover_and_costs(weights, cfg.tc_bps)
    weekly_r_net = weekly_r - costs
    mets = compute_metrics(weekly_r_net, rf_annual=cfg.risk_free_annual, tc_series=costs)
    mets.update(dict(Turnover=turnover.mean(), EffN=effN.mean()))
    return weights, weekly_r, weekly_r_net, costs, mets

# =========================
# Orchestration
# =========================
def run_once(model_name: str, cfg: Config):
    set_seed(cfg.seed)
    prices = load_prices(cfg).dropna(how="any")
    missing = [t for t in cfg.tickers if t not in prices.columns]
    if missing:
        raise RuntimeError(f"Missing tickers from downloaded data: {missing}")
    rets = to_logrets(prices)

    # Standardize inputs on TRAIN only
    train_mask = (rets.index <= pd.to_datetime(cfg.train_end))
    zrets, mu, sigma = standardize_returns(rets[train_mask], rets)

    # Build samples (weekly)
    train, val, test, _ = build_samples(cfg, zrets, rets)
    if train is None or val is None or test is None:
        raise RuntimeError("Not enough samples. Try smaller lookback or different date bounds.")

    (Xtr, Ytr, Dtr), (Xv, Yv, Dv), (Xte, Yte, Dte) = train, val, test
    n_assets = Xtr.shape[1]; L = Xtr.shape[2]
    print(f"Samples: train={len(Xtr)}  val={len(Xv)}  test={len(Xte)} | assets={n_assets}  L={L}")

    # Datasets
    train_ds = SeqAllocDataset(Xtr, Ytr)
    val_ds   = SeqAllocDataset(Xv,  Yv)

    # Model
    if model_name.lower()=="lstm":
        model = LSTMAllocator(n_assets, L, hidden=128, layers=2, dropout=0.1)
    elif model_name.lower()=="transformer":
        model = TransformerAllocator(n_assets, L, d_model=64, nhead=4, depth=2, d_ff=128, dropout=0.1, cross_asset_attn=True)
    else:
        raise ValueError("model must be 'lstm' or 'transformer'")

    # Train with early stopping on val Sharpe
    model, best = train_model(model, cfg, train_ds, val_ds)
    print(f"Best val Sharpe: {best['val_sharpe']:.4f} @ epoch {best['epoch']}")

    # Evaluate OOS (2019–2025)
    weights, wk_r, wk_r_net, costs, mets = evaluate_model(model, cfg, Xte, Yte, Dte, cfg.tickers)

    # Persist artifacts
    tag = model_name.lower()
    weights.to_csv(f"weights_{tag}.csv", float_format="%.6f")
    pd.DataFrame({"weekly_r_gross": wk_r, "weekly_r_net": wk_r_net, "costs": costs}).to_csv(f"weekly_returns_{tag}.csv", float_format="%.8f")
    row = {"Model": tag, **{k: float(v) for k,v in mets.items()}}
    if not os.path.exists(cfg.results_csv):
        pd.DataFrame([row]).to_csv(cfg.results_csv, index=False)
    else:
        pd.concat([pd.read_csv(cfg.results_csv), pd.DataFrame([row])], ignore_index=True).to_csv(cfg.results_csv, index=False)

    print(f"[{tag}] Metrics:", {k: round(v,4) for k,v in mets.items()})
    print(f"Saved: weights_{tag}.csv, weekly_returns_{tag}.csv, appended -> {cfg.results_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="both", choices=["lstm","transformer","both"])
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--tc_bps", type=float, default=5.0)
    ap.add_argument("--entropy_reg", type=float, default=1e-3)
    args = ap.parse_args()

    cfg = Config(lookback=args.lookback, epochs=args.epochs, tc_bps=args.tc_bps, entropy_reg=args.entropy_reg)
    if args.model in ("lstm","transformer"):
        run_once(args.model, cfg)
    else:
        run_once("lstm", cfg)
        run_once("transformer", cfg)

if __name__ == "__main__":
    main()
