"""
Plot Table 1 / Figure 1: model comparison bar chart.

Reads:    results/comparison_results.csv
Writes:   results/plot_comparison.pdf

Usage:
    python plot_comparison.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = Path(_HERE) / "results"
COMPARISON_CSV = RESULTS_DIR / "comparison_results.csv"
PLOT_PDF = RESULTS_DIR / "plot_comparison.pdf"


def main():
    if not COMPARISON_CSV.exists():
        print(f"Run aggregate.py first; missing: {COMPARISON_CSV}")
        return

    df = pd.read_csv(COMPARISON_CSV)
    df_geco = df[df["dataset"] == "geco_test"]
    metrics = ["r_trt", "r_ffd", "r_gaze", "r_skip"]
    df_geco = df_geco[df_geco["metric"].isin(metrics)]

    models = df_geco["model"].unique()
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.20

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.5), 5))
    for i, m in enumerate(metrics):
        sub = df_geco[df_geco["metric"] == m].set_index("model")
        means = [sub.loc[mod]["mean"] if mod in sub.index else 0 for mod in models]
        stds = [sub.loc[mod]["std"] if mod in sub.index else 0 for mod in models]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, label=m, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_title("Word-level prediction performance on GECO test")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(PLOT_PDF)
    print(f"Wrote {PLOT_PDF}")


if __name__ == "__main__":
    main()
