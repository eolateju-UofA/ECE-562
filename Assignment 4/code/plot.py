"""Plot mean +/- one standard deviation across seeds for any logged column.

    python plot.py --column eval_return_mean --runs runs/base runs/clip --out return.png
    python plot.py --column eval_return_mean --runs runs/base ../a3/runs/k0 --labels SAC "PPO k=0"

Each entry of --runs is a directory containing seed*/progress.csv. Runs are aligned
on the 'step' column. For files with a different step column name, use --step-col.
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(run_dir, column, step_col):
    frames = []
    for f in sorted(glob.glob(os.path.join(run_dir, "seed*", "*.csv"))):
        df = pd.read_csv(f)
        if column in df.columns:
            frames.append(df[[step_col, column]].dropna().set_index(step_col)[column])
    if not frames:
        raise SystemExit(f"No CSVs with column '{column}' under {run_dir}")
    # Keep only steps present in every seed (runs that diverged early are truncated).
    data = pd.concat(frames, axis=1, join="inner")
    return data.index.values, data.values.mean(axis=1), data.values.std(axis=1), data.shape[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+")
    p.add_argument("--column", default="eval_return_mean")
    p.add_argument("--step-col", default="step")
    p.add_argument("--logy", action="store_true")
    p.add_argument("--out", default="plot.png")
    args = p.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    fig, ax = plt.subplots(figsize=(6, 4))
    for run, label in zip(args.runs, labels):
        x, m, s, n = load(run, args.column, args.step_col)
        ax.plot(x, m, label=f"{label} (n={n})")
        ax.fill_between(x, m - s, m + s, alpha=0.2)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(args.column)
    if args.logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
