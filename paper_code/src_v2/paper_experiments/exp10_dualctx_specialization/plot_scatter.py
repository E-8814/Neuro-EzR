"""
Analysis 5 — scatter ctx_FFD vs ctx_skip per word.

If the two heads have collapsed to identity → all points lie on y=x.
If they specialize → points spread off the diagonal.

Color: log_freq (frequent ↔ rare)
Size:  word_length

Reads:    results/per_word_dualctx.csv
Writes:   results/plot_scatter.pdf
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = Path(_HERE) / "results"
PER_WORD_CSV = RESULTS_DIR / "per_word_dualctx.csv"
PLOT_PDF = RESULTS_DIR / "plot_scatter.pdf"


def main():
    if not PER_WORD_CSV.exists():
        print(f"Run extract_per_word_features.py first; missing: {PER_WORD_CSV}")
        return

    df = pd.read_csv(PER_WORD_CSV)
    corpora = sorted(df["corpus"].unique())

    fig, axes = plt.subplots(
        1, len(corpora), figsize=(5.5 * len(corpora), 5),
    )
    if len(corpora) == 1:
        axes = [axes]

    for ax, corpus in zip(axes, corpora):
        sub = df[df["corpus"] == corpus]
        x = sub["ctx_FFD"].values
        y = sub["ctx_skip"].values
        log_freq = sub["log_freq"].values
        wlen = sub["word_length"].values

        # Sizes scaled to word length, capped for plot legibility
        sizes = 5 + (wlen - wlen.min()) * 5
        sizes = np.clip(sizes, 5, 80)

        sc = ax.scatter(
            x, y, c=log_freq, s=sizes, cmap="viridis",
            alpha=0.5, edgecolor="none",
        )

        # y = x diagonal
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6,
                label="y = x")

        r = float(np.corrcoef(x, y)[0, 1])
        ax.set_xlabel("ctx_head_FFD output (ms)")
        ax.set_ylabel("ctx_head_skip output (ms)")
        ax.set_title(f"{corpus}  (r = {r:+.3f}, n = {len(sub):,})")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("log frequency")

    fig.suptitle(
        "Per-word ctx_head_FFD vs ctx_head_skip outputs",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOT_PDF)
    print(f"Wrote {PLOT_PDF}")


if __name__ == "__main__":
    main()
