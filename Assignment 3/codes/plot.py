"""
Plots for Q5 and Q6 (ECE 562, Assignment 3).

    python plot.py                 # reads runs/, writes figs/

Each curve is the mean across seeds with a shaded +/- one standard deviation
region, against environment steps.  If a seed stopped early because of
numerical instability (runs/<run>.unstable), the curve uses the seeds still
running at each step and an 'x' marks where each seed stopped.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEEDS = [1, 2, 3]


def load(log_dir, k, loss, metric):
    """Returns ({seed: Series indexed by global_step}, {seed: unstable step or None})."""
    series, unstable = {}, {}
    for seed in SEEDS:
        name = f"k{k}_{loss}__seed{seed}"
        path = os.path.join(log_dir, name + ".csv")
        if not os.path.exists(path):
            print(f"missing {path}")
            continue
        df = pd.read_csv(path)
        s = df.set_index("global_step")[metric].dropna()
        if len(s):
            series[seed] = s
        flag = os.path.join(log_dir, name + ".unstable")
        unstable[seed] = int(open(flag).read().split(",")[1]) if os.path.exists(flag) else None
    return series, unstable


def plot_metric(ax, log_dir, configs, metric, logy=False):
    for k, loss, label in configs:
        series, unstable = load(log_dir, k, loss, metric)
        if not series:
            continue
        df = pd.DataFrame(series).sort_index()      # columns = seeds, NaN after a seed stops
        mean, std = df.mean(axis=1), df.std(axis=1, ddof=0)
        line, = ax.plot(df.index, mean, label=label)
        lower = mean - std
        if logy:
            # mean - std can be <= 0, which a log axis cannot show; keep the band positive
            lower = np.maximum(lower, mean * 1e-2)
        ax.fill_between(df.index, lower, mean + std, color=line.get_color(), alpha=0.2)
        for seed, step in unstable.items():
            if step is not None:
                ax.axvline(step, color=line.get_color(), ls=":", lw=1)
                y = series[seed].iloc[-1] if seed in series else mean.iloc[-1]
                ax.plot(series[seed].index[-1] if seed in series else step, y, "x",
                        color=line.get_color(), ms=8)
                print(f"{label}: seed {seed} unstable at step {step}")


def make_fig(log_dir, out_path, configs, metric, ylabel, title, logy=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_metric(ax, log_dir, configs, metric, logy)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="runs")
    p.add_argument("--out-dir", default="figs")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda f: os.path.join(args.out_dir, f)

    q5 = [(k, "clip", f"k = {k}") for k in (0, 2, 4)]
    make_fig(args.log_dir, out("q5_eval_return.png"), q5, "eval_return_mean",
             "evaluation return", "Q5: evaluation return vs. data lag k")
    make_fig(args.log_dir, out("q5_kl_start.png"), q5, "kl_start",
             "KL(behavior || current) at update start", "Q5: kl_start")
    make_fig(args.log_dir, out("q5_clipfrac_start.png"), q5, "clipfrac_start",
             "clip fraction at update start", "Q5: clipfrac_start")

    q6 = [(2, loss, loss) for loss in ("naive", "ratio", "clip")]
    make_fig(args.log_dir, out("q6_eval_return.png"), q6, "eval_return_mean",
             "evaluation return", "Q6: evaluation return (k = 2)")
    make_fig(args.log_dir, out("q6_kl_end.png"), q6, "kl_end",
             "KL(behavior || current) at update end", "Q6: kl_end (k = 2)", logy=True)


if __name__ == "__main__":
    main()
