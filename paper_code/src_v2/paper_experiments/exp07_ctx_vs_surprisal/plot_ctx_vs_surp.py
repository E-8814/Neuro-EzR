"""
Plot ctx vs surp comparison.

Reads:    results/ctx_vs_surp_results.csv (long form)
          results/ctx_vs_surp_summary.csv (paired stats)
Writes:   results/plot_ctx_vs_surp.pdf
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = Path(_HERE) / "results"
LONG_CSV = RESULTS_DIR / "ctx_vs_surp_results.csv"
PLOT_PDF = RESULTS_DIR / "plot_ctx_vs_surp.pdf"


def main():
    if not LONG_CSV.exists():
        print(f"Run aggregate.py first; missing: {LONG_CSV}")
        return

    df = pd.read_csv(LONG_CSV)
    df = df[df["dataset"] == "geco_test"]
    metrics = ["r_trt", "r_ffd", "r_gaze", "r_skip"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4),
                             sharey=True)
    for ax, m in zip(axes, metrics):
        sub = df[df["metric"] == m]
        means = sub.groupby("variant")["value"].mean()
        stds = sub.groupby("variant")["value"].std(ddof=1)
        ax.bar(means.index, means.values, yerr=stds.values, capsize=4,
               color=["tab:blue", "tab:orange"])
        # Per-seed dots
        for variant, group in sub.groupby("variant"):
            x = list(means.index).index(variant)
            ax.scatter([x] * len(group), group["value"],
                       color="black", s=20, alpha=0.6)
        ax.set_title(m)
        ax.set_ylabel("Pearson r" if m == "r_trt" else "")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("ctx_head vs TinyLlama-surprisal (paper model variants)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_PDF)
    print(f"Wrote {PLOT_PDF}")


if __name__ == "__main__":
    main()
