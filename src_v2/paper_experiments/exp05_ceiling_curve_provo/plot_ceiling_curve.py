"""Plot model-vs-ceiling curve as a function of data fraction."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = Path(_HERE) / "results"
CURVE_CSV = RESULTS_DIR / "ceiling_curve_results.csv"
PLOT_PDF = RESULTS_DIR / "plot_ceiling_curve.pdf"


def main():
    if not CURVE_CSV.exists():
        print(f"Run compute_ceiling_curve.py first; missing: {CURVE_CSV}")
        return

    df = pd.read_csv(CURVE_CSV)
    metrics = sorted(df["metric"].unique())

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4),
                             sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        sub = df[df["metric"] == m].sort_values("data_fraction")
        ax.plot(sub["data_fraction"], sub["model_r"], "o-",
                label="model", color="tab:blue")
        ax.plot(sub["data_fraction"], sub["ceiling_r"], "s--",
                label="ceiling (Spearman-Brown)", color="tab:red")
        ax.set_title(m)
        ax.set_xlabel("Provo participant fraction")
        ax.set_ylabel("Pearson r")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("Model performance vs noise ceiling on Provo", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_PDF)
    print(f"Wrote {PLOT_PDF}")


if __name__ == "__main__":
    main()
