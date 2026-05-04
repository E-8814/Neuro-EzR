"""
Copy/regenerate paper-ready figures from each experiment's results.

Each experiment already produces a `plot_*.pdf`. This script collects
them all in `results/figures/` with paper-style names.

Usage:
    python make_paper_figures.py
"""

import os
import shutil
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from paper_experiments import config


PAPER_ROOT = Path(_HERE).parent
FIGURES_DIR = config.PAPER_FINAL_FIGURES
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Mapping: experiment plot path -> paper figure name.
FIGURE_MAP = [
    ("exp01_main_comparison/results/plot_comparison.pdf", "fig1_main_comparison.pdf"),
    ("exp03_lesion_study/results/plot_lesion.pdf",        "fig3_lesion.pdf"),
    ("exp07_ctx_vs_surprisal/results/plot_ctx_vs_surp.pdf",      "fig5_ctx_vs_surp.pdf"),
    ("exp09_per_participant_cog_fits/results/plot_per_participant_cog.pdf",
     "fig6_per_participant_cog.pdf"),
]


def main():
    print("Collecting paper figures...")
    for src_rel, dst_name in FIGURE_MAP:
        src = PAPER_ROOT / src_rel
        dst = FIGURES_DIR / dst_name
        if not src.exists():
            print(f"  [missing] {src_rel}")
            continue
        shutil.copy(str(src), str(dst))
        print(f"  {src_rel} -> {dst_name}")

    print(f"\nAll figures in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
