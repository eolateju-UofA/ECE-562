import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = "runs/"
OUT_DIR  = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
STYLE = dict(linewidth=2.0)
FILL  = dict(alpha=0.20)

def load(fname):
    return pd.read_csv(os.path.join(DATA_DIR, fname))

def smooth(series, w=3):
    return series.rolling(w, min_periods=1, center=True).mean()

def setup_ax(ax, xlabel="Environment Steps", ylabel="Episodic Return"):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"),           bbox_inches="tight")
    print(f"  saved {name}.png  +  {name}.pdf")
    plt.close(fig)

# ── Plot 1: Q6(a) — Benchmark, mean ± std over 3 seeds ────────────────────────
def plot_benchmark():
    fig, ax = plt.subplots(figsize=(6.5, 4))

    dfs = [load(f"InvertedPendulum-v5__lam0.95__clip0.2__seed{s}.csv")
           for s in (1, 2, 3)]

    steps  = dfs[0]["global_step"].values
    matrix = np.stack([df["eval_return_mean"].values for df in dfs])
    mu     = matrix.mean(axis=0)
    sigma  = matrix.std(axis=0)

    ax.plot(steps, smooth(pd.Series(mu)),
            color="#2563EB", label="Mean eval return", **STYLE)
    ax.fill_between(steps,
                    smooth(pd.Series(mu - sigma)),
                    smooth(pd.Series(mu + sigma)),
                    color="#2563EB", label="±1 std", **FILL)
    ax.axhline(1000, color="gray", linestyle=":", linewidth=1.2,
               label="Max return (1000)")

    setup_ax(ax)
    ax.set_title(
        "Q6(a) — InvertedPendulum-v5 Benchmark\n"
        r"($\epsilon=0.2,\;\lambda=0.95,\;\gamma=0.99$, seeds 1–3)",
        fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "Q6a_benchmark")

# ── Plot 2: Q6(b) — GAE λ ablation ────────────────────────────────────────────
def plot_gae_ablation():
    fig, ax = plt.subplots(figsize=(6.5, 4))

    configs = [
        ("lam0.0__clip0.2__seed1",  r"$\lambda=0.0$ (1-step TD)",   "#DC2626"),
        ("lam0.95__clip0.2__seed1", r"$\lambda=0.95$ (GAE default)", "#2563EB"),
        ("lam1.0__clip0.2__seed1",  r"$\lambda=1.0$ (Monte Carlo)",  "#16A34A"),
    ]

    for tag, label, color in configs:
        df = load(f"InvertedPendulum-v5__{tag}.csv")
        ax.plot(df["global_step"], smooth(df["eval_return_mean"]),
                color=color, label=label, **STYLE)

    ax.axhline(1000, color="gray", linestyle=":", linewidth=1.2,
               label="Max return (1000)")

    setup_ax(ax)
    ax.set_title(r"Q6(b) — GAE $\lambda$ Trade-off Ablation", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "Q6b_gae_ablation")

# ── Plot 3: Q6(c) — Clipping ablation, learning curves ────────────────────────
def plot_clip_returns():
    fig, ax = plt.subplots(figsize=(6.5, 4))

    clipped   = load("InvertedPendulum-v5__lam0.95__clip0.2__seed1.csv")
    unclipped = load("InvertedPendulum-v5__lam0.95__clip1000.0__seed1.csv")

    ax.plot(clipped["global_step"],
            smooth(clipped["eval_return_mean"]),
            color="#2563EB", label=r"Clipped ($\epsilon=0.2$)", **STYLE)
    ax.plot(unclipped["global_step"],
            smooth(unclipped["eval_return_mean"]),
            color="#DC2626", label=r"Unclipped ($\epsilon=\infty$)", **STYLE)
    ax.axhline(1000, color="gray", linestyle=":", linewidth=1.2,
               label="Max return (1000)")

    setup_ax(ax)
    ax.set_title("Q6(c) — Policy Clipping Ablation: Eval Return", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "Q6c_clip_returns")

# ── Plot 4: Q6(c) — Clipping ablation, approx KL ──────────────────────────────
def plot_clip_kl():
    fig, ax = plt.subplots(figsize=(6.5, 4))

    clipped   = load("InvertedPendulum-v5__lam0.95__clip0.2__seed1.csv").dropna(subset=["approx_kl"])
    unclipped = load("InvertedPendulum-v5__lam0.95__clip1000.0__seed1.csv").dropna(subset=["approx_kl"])

    ax.plot(clipped["global_step"],
            smooth(clipped["approx_kl"]),
            color="#2563EB", label=r"Clipped ($\epsilon=0.2$)", **STYLE)
    ax.plot(unclipped["global_step"],
            smooth(unclipped["approx_kl"]),
            color="#DC2626", label=r"Unclipped ($\epsilon=\infty$)", **STYLE)

    setup_ax(ax, ylabel="Approx. KL Divergence")
    ax.set_title("Q6(c) — Policy Clipping Ablation: Approx. KL Divergence", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "Q6c_clip_kl")

# ── Run all ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating plots...")
    plot_benchmark()
    plot_gae_ablation()
    plot_clip_returns()
    plot_clip_kl()
    print(f"\nAll plots saved to: {os.path.abspath(OUT_DIR)}/")