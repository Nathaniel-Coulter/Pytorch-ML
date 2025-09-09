#!/usr/bin/env python3
# src/sec9_evt_plot.py
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from scipy.stats import genpareto  # SciPy is standard in your stack

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT/"outputs", ROOT/"figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

ASSET = "SPY"         # change to another ticker if you like
TAIL_Q = 0.05         # use 5% lowest returns as POT threshold (95th quantile of losses)
ALPHA  = 0.99         # report 99% tail quantile based on GPD

def load_weekly_rets():
    px = pd.read_parquet(OUT/"prices.parquet").xs("Adj Close", level=1, axis=1)
    px = px.apply(pd.to_numeric, errors="coerce")
    pxw = px.asfreq("B").ffill().resample("W-FRI").last().dropna(how="all")
    r = np.log(pxw/pxw.shift(1)).dropna(how="all")
    return r

def fit_pot_left_tail(r: pd.Series):
    x = r.dropna().values
    # Left tail (losses): work with negative returns, so large values = large losses
    z = -x
    u = np.quantile(z, 1 - TAIL_Q)    # POT threshold on losses
    excess = z[z > u] - u             # exceedances above u
    # SciPy genpareto parameterization: shape=xi, loc=0 (excesses), scale=beta
    xi, loc, beta = genpareto.fit(excess, floc=0.0)
    return dict(xi=xi, beta=beta, u=u, n=len(excess), N=len(z), excess=excess, z=z)

def tail_quantile(u, xi, beta, p):
    # POT quantile on the original loss scale (GPD on excesses)
    # q_p = u + beta/xi * [ ( (1-p)/(1-p_u) )^(-xi) - 1 ], with p_u = tail prob beyond u
    return u + beta/xi * ( (p)**(-xi) - 1 ) if xi != 0 else u - beta*np.log(p)

def main():
    R = load_weekly_rets()
    r_spy = R[ASSET].dropna()
    pot = fit_pot_left_tail(r_spy)

    # Empirical survival for exceedances
    ex = np.sort(pot["excess"])
    n = len(ex)
    if n < 30:
        print("[warn] Not enough exceedances for a stable fit.")
    emp_sf = 1 - (np.arange(1, n+1) / (n+1))

    # Theoretical survival under fitted GPD
    th_sf = 1 - genpareto.cdf(ex, c=pot["xi"], loc=0.0, scale=pot["beta"])

    # QQ data
    probs = (np.arange(1, n+1) - 0.5) / n
    gpd_q = genpareto.ppf(probs, c=pot["xi"], loc=0.0, scale=pot["beta"])

    # Report 99% tail VaR on losses, then convert back to return quantile
    # p_exceed = proportion beyond threshold
    p_exceed = TAIL_Q
    # Within tail, 99% of *tail* => overall p = 1 - (1 - p_exceed)*(0.99) ≈ 1 - 0.99*(1 - p_exceed)
    p_tail = 0.99
    p_overall = 1 - (1 - p_exceed)*p_tail
    # Convert using GPD quantile on excess scale:
    q_excess_99 = genpareto.ppf(p_tail, c=pot["xi"], loc=0.0, scale=pot["beta"])
    loss_q = pot["u"] + q_excess_99
    ret_q = -loss_q  # back to return space (negative number)

    # Save metrics
    pd.DataFrame([{
        "Ticker": ASSET,
        "Tail_q": TAIL_Q,
        "Threshold_u(losses)": pot["u"],
        "xi": pot["xi"],
        "beta": pot["beta"],
        "Exceedances": pot["n"],
        "Sample": pot["N"],
        "VaR_99_weekly": ret_q
    }]).to_csv(OUT/"sec9_evt_metrics.csv", index=False)

    # Plot: (1) survival of exceedances (empirical vs GPD), (2) QQ plot
    plt.figure(figsize=(12,5))

    ax1 = plt.subplot(1,2,1)
    ax1.plot(ex, emp_sf, marker="o", linestyle="", alpha=0.6, label="Empirical SF")
    ax1.plot(ex, th_sf, lw=2, label="GPD fit SF")
    ax1.set_title(f"EVT POT (left tail) — {ASSET}\nthreshold u={pot['u']:.4f}, xi={pot['xi']:.3f}, beta={pot['beta']:.4f}")
    ax1.set_xlabel("Excess over u (loss scale)")
    ax1.set_ylabel("Survival probability")
    ax1.set_yscale("log")
    ax1.legend()

    ax2 = plt.subplot(1,2,2)
    ax2.plot(gpd_q, ex, marker="o", linestyle="", alpha=0.6)
    lo = min(gpd_q.min(), ex.min()); hi = max(gpd_q.max(), ex.max())
    ax2.plot([lo,hi],[lo,hi], lw=2, alpha=0.7)
    ax2.set_title("GPD QQ plot (exceedances)")
    ax2.set_xlabel("Theoretical quantiles (GPD)")
    ax2.set_ylabel("Empirical quantiles (excess)")
    plt.tight_layout()
    plt.savefig(FIG/"sec9_evt_spx.png", dpi=150)

    print("[ok] wrote",
          OUT/"sec9_evt_metrics.csv",
          FIG/"sec9_evt_spx.png")

if __name__ == "__main__":
    main()
