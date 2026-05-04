"""
Plot lesion deltas as horizontal bars.

Reads:    results/lesion_results.csv
Writes:   results/plot_lesion.pdf
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = Path(_HERE) / "results"
LESION_CSV = RESULTS_DIR / "lesion_results.csv"
PLOT_PDF = RESULTS_DIR / "plot_lesion.pdf"


def main():
    if not LESION_CSV.exists():
        print(f"Run run_lesions.py first; missing: {LESION_CSV}")
        return

    df = pd.read_csv(LESION_CSV)
    df = df[df["dataset"] == "geco_test"]
    df = df[df["metric"].isin(["r_trt", "r_ffd", "r_gaze", "r_skip"])]
    df = df[df["lesion"] != "full"]

    metrics = ["r_trt", "r_ffd", "r_gaze", "r_skip"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5), sharey=True)

    for ax, m in zip(axes, metrics):
        sub = df[df["metric"] == m].sort_values("delta_vs_full")
        ax.barh(sub["lesion"], sub["delta_vs_full"])
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(m)
        ax.set_xlabel("Δr vs full model")
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.suptitle("Per-lesion Δr (negative = lesion hurts)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOT_PDF)
    print(f"Wrote {PLOT_PDF}")


if __name__ == "__main__":
    main()
