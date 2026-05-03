"""
Plot Figure 2 — random-init parameter recovery.

For each of the 7 Reichle-targeted parameters:
  - Light ×'s for init values (5)
  - Dark ●'s for converged values (5)
  - Horizontal line for Reichle 2003 published value

Reads:    results/recovery_results.csv
Writes:   results/plot_recovery.pdf
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = Path(_HERE) / "results"
RECOVERY_CSV = RESULTS_DIR / "recovery_results.csv"
PLOT_PDF = RESULTS_DIR / "plot_recovery.pdf"


def main():
    if not RECOVERY_CSV.exists():
        print(f"Run aggregate.py first; missing: {RECOVERY_CSV}")
        return

    df = pd.read_csv(RECOVERY_CSV)
    params = sorted(df["param"].unique())

    n = len(params)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.5))
    axes = np.atleast_2d(axes).flatten()

    for i, param in enumerate(params):
        ax = axes[i]
        sub = df[df["param"] == param]
        inits = sub["init_value"].values
        convs = sub["converged_value"].values
        reichle = sub["reichle_target"].iloc[0]

        # Plot
        ax.scatter([0] * len(inits), inits, marker="x", color="tab:red",
                   s=60, label="random init", alpha=0.7)
        ax.scatter([1] * len(convs), convs, marker="o", color="tab:blue",
                   s=60, label="converged", alpha=0.85)
        ax.axhline(reichle, linestyle="--", color="black",
                   label=f"Reichle 2003 = {reichle:.2f}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["init", "final"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(param, fontsize=10)
        if i == 0:
            ax.legend(loc="best", fontsize=7)

    # Hide unused subplots
    for j in range(len(params), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Random-init parameter recovery (5 seeds, ±50% jitter)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PLOT_PDF)
    print(f"Wrote {PLOT_PDF}")


if __name__ == "__main__":
    main()
