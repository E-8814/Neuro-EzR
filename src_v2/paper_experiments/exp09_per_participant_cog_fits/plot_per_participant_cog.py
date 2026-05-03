"""
Plot per-participant cog parameter distributions and correlations.

Reads:    results/per_participant_cog_fits.csv
          results/cog_correlations.csv
Writes:   results/plot_per_participant_cog.pdf
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = Path(_HERE) / "results"
FITS_CSV = RESULTS_DIR / "per_participant_cog_fits.csv"
CORR_CSV = RESULTS_DIR / "cog_correlations.csv"
PLOT_PDF = RESULTS_DIR / "plot_per_participant_cog.pdf"

COG_PARAMS = [
    "alpha1_reichle", "alpha2_reichle", "epsilon",
    "M1", "M2_eq_I", "delta", "lambda_refix",
]


def main():
    if not FITS_CSV.exists():
        print(f"Run fit_per_participant.py first; missing: {FITS_CSV}")
        return

    df = pd.read_csv(FITS_CSV)
    cols = [c for c in COG_PARAMS if c in df.columns]
    n_params = len(cols)

    fig, axes = plt.subplots(2, max(n_params, 4), figsize=(3 * n_params, 6))
    axes = np.atleast_2d(axes)

    # Top row: distribution of each parameter across readers
    for i, p in enumerate(cols):
        ax = axes[0, i]
        ax.hist(df[p], bins=8, edgecolor="black", color="tab:blue", alpha=0.7)
        ax.set_title(p, fontsize=9)
        ax.set_xlabel("fitted value")
        ax.set_ylabel("# readers")

    # Bottom row: correlation with mean_RT (proxy for reading speed)
    if "mean_RT" in df.columns:
        for i, p in enumerate(cols):
            ax = axes[1, i]
            ax.scatter(df["mean_RT"], df[p], color="tab:orange", s=40)
            ax.set_xlabel("mean RT (ms)")
            ax.set_ylabel(p)
            # Fit line
            x = df["mean_RT"].values
            y = df[p].values
            if len(x) > 2 and np.std(x) > 0:
                m, b = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, m * xs + b, "k--", alpha=0.5)
                r = np.corrcoef(x, y)[0, 1]
                ax.set_title(f"r = {r:+.2f}", fontsize=9)
    else:
        for i in range(len(cols)):
            axes[1, i].axis("off")

    fig.suptitle(
        "Per-participant cognitive parameter fits\n"
        "Top: distribution across 14 readers. Bottom: correlation with mean reading time.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PLOT_PDF)
    print(f"Wrote {PLOT_PDF}")


if __name__ == "__main__":
    main()
