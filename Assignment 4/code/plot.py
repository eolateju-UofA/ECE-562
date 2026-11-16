"""Plot mean +/- one standard deviation across seeds for any logged column.

    python plot.py --column eval_return_mean --runs runs/base runs/clip --out return.png
    python plot.py --column eval_return_mean --runs runs/base ../a3/runs/k0 --labels SAC "PPO k=0"

Each entry of --runs is a directory containing seed*/progress.csv. Runs are aligned
on the 'step' column. For files with a different step column name, use --step-col.

Added options (Assignment 4, Section 2):
    --columns c1 c2 ...    one column per run (e.g. kl_beh for SAC, kl_start for PPO)
    --step-cols s1 s2 ...  one step column per run (e.g. step for SAC, global_step for PPO)
    --scale x              divide the plotted values by x (data age -> PPO iterations)
    --hline y [--hline y]  horizontal reference lines, labelled with --hline-labels
    --xmax, --ylabel, --title, --legend-loc
    --symlog t             symmetric log y-axis (linear within [-t, t]) for values spanning many
                           orders of magnitude and both signs
On a log axis, rows whose value is <= 0 are dropped (log(0) is undefined).
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(run_dir, column, step_col, positive_only=False, xmax=None):
    frames = []
    for f in sorted(glob.glob(os.path.join(run_dir, "seed*", "*.csv"))):
        df = pd.read_csv(f)
        if column in df.columns:
            d = df[[step_col, column]].dropna()
            if positive_only:
                d = d[d[column] > 0]
            if xmax is not None:
                d = d[d[step_col] <= xmax]
            frames.append(d.set_index(step_col)[column])
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
    p.add_argument("--columns", nargs="+")
    p.add_argument("--step-cols", nargs="+")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--hline", type=float, action="append", default=[])
    p.add_argument("--hline-labels", nargs="+")
    p.add_argument("--xmax", type=float)
    p.add_argument("--ylabel")
    p.add_argument("--title")
    p.add_argument("--logy", action="store_true")
    p.add_argument("--symlog", type=float, help="symmetric log y-axis with this linear threshold")
    p.add_argument("--legend-loc", default="best")
    p.add_argument("--out", default="plot.png")
    args = p.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    columns = args.columns or [args.column] * len(args.runs)
    step_cols = args.step_cols or [args.step_col] * len(args.runs)
    fig, ax = plt.subplots(figsize=(6, 4))
    for run, label, col, scol in zip(args.runs, labels, columns, step_cols):
        try:
            x, m, s, n = load(run, col, scol, positive_only=args.logy, xmax=args.xmax)
        except SystemExit as e:
            print("skipped:", e)
            continue
        if len(x) == 0:  # every value was <= 0 on a log axis
            ax.plot([], [], " ", label=f"{label}: always 0, not drawn on log axis")
            continue
        m, s = m / args.scale, s / args.scale
        line, = ax.plot(x, m, label=f"{label} (n={n})")
        lo = np.maximum(m - s, m * 1e-3) if args.logy else m - s
        ax.fill_between(x, lo, m + s, alpha=0.2, color=line.get_color())
    hl_labels = args.hline_labels or [f"y = {y:g}" for y in args.hline]
    for y, lab, ls in zip(args.hline, hl_labels, ["--", ":", "-."] * 5):
        ax.axhline(y, color="k", linestyle=ls, linewidth=1, label=lab)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(args.ylabel or columns[0])
    if args.title:
        ax.set_title(args.title)
    if args.xmax:
        ax.set_xlim(right=args.xmax)
    if args.logy:
        ax.set_yscale("log")
    if args.symlog:
        ax.set_yscale("symlog", linthresh=args.symlog)
    ax.grid(alpha=0.3)
    ax.legend(loc=args.legend_loc, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
