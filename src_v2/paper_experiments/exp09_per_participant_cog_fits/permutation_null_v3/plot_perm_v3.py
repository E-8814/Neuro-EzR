"""
Regenerate the paper's Figure 1 (3x2 panel) from the v3 exact null.

Layout mirrors the published figure:
  columns = parameter groups:
      (1) lexical-rate  alpha1, alpha2   (Reichle 2013's developmental locus)
      (2) lexical movers delta, lambda_refix, epsilon, refix_pivot
      (3) motor/decision M1, M2=I, skip_temperature
  top row    = OBSERVED signed %shift (fast -> slow), value labels
  bottom row = null distribution of signed %shifts over all completed splits
               (boxplots, one per parameter)

Run AFTER perm_v3.py has completed all 1,716 splits (works on partial
results too, with the split count in the subtitle).

Output: results/plot_perm_3x2_v3.pdf

Usage:
    python -u plot_perm_v3.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP09 = os.path.abspath(os.path.join(_HERE, ".."))
PERM_V2 = os.path.join(EXP09, "permutation_null")
SRC_V2 = os.path.abspath(os.path.join(EXP09, "..", ".."))
for p in (SRC_V2, EXP09, PERM_V2, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from _cog_fit import enumerate_balanced_splits, split_index  # noqa: E402
from fit_per_group import FAST_READERS, SLOW_READERS  # noqa: E402

PERMS_DIR = Path(_HERE) / "results" / "perms"
OUT_PDF = Path(_HERE) / "results" / "plot_perm_3x2_v3.pdf"

GROUPS = [
    ("Lexical-rate\n(Reichle 2013 locus)",
     [("alpha1_reichle", r"$\alpha_1$"), ("alpha2_reichle", r"$\alpha_2$")],
     "tab:blue"),
    ("Lexical access /\nrefixation",
     [("delta", r"$\delta$"), ("lambda_refix", r"$\lambda_{refix}$"),
      ("epsilon", r"$\varepsilon$"), ("refix_pivot", "pivot")],
     "tab:orange"),
    ("Motor / decision",
     [("M1", r"$M_1$"), ("M2_eq_I", r"$M_2{=}I$"),
      ("skip_temperature", r"$\tau$")],
     "tab:green"),
]


def pct_shift(fast_val, slow_val):
    denom = abs(fast_val) if abs(fast_val) > 1e-9 else 1.0
    return 100.0 * (slow_val - fast_val) / denom


def main():
    records = {}
    for path in sorted(PERMS_DIR.glob("perm_*.json")):
        if path.name.endswith(".error.json"):
            continue
        d = json.loads(path.read_text())
        records[d["index"]] = d
    if not records:
        print("No completed perms.")
        sys.exit(1)
    n_done = len(records)
    print(f"Loaded {n_done} splits.")

    any_rec = next(iter(records.values()))
    pids = sorted(set(any_rec["group_a"]) | set(any_rec["group_b"]))
    splits = enumerate_balanced_splits(pids)
    obs_idx = split_index(splits, sorted(FAST_READERS), sorted(SLOW_READERS))
    if obs_idx not in records:
        print(f"Observed split (index {obs_idx}) not completed yet.")
        sys.exit(2)
    obs = records[obs_idx]
    if frozenset(obs["group_a"]) == frozenset(FAST_READERS):
        fast_cog, slow_cog = obs["cog_a"], obs["cog_b"]
    else:
        fast_cog, slow_cog = obs["cog_b"], obs["cog_a"]

    fig, axes = plt.subplots(2, 3, figsize=(8.2, 4.6))
    for col, (title, params, color) in enumerate(GROUPS):
        labels = [lab for _, lab in params]

        # top: observed signed shifts
        ax = axes[0, col]
        vals = [pct_shift(fast_cog[k], slow_cog[k]) for k, _ in params]
        ax.bar(range(len(vals)), vals, color=color, alpha=0.85)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:+.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(0, lw=0.6, color="k")
        ax.set_title(title, fontsize=9)
        if col == 0:
            ax.set_ylabel("Observed %shift\n(fast → slow)", fontsize=8)

        # bottom: null distributions (signed)
        ax = axes[1, col]
        null_vals = []
        for k, _ in params:
            nv = [pct_shift(r["cog_a"][k], r["cog_b"][k])
                  for i, r in records.items() if i != obs_idx]
            null_vals.append(nv)
        bp = ax.boxplot(null_vals, showfliers=False, widths=0.6,
                        patch_artist=True)
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.35)
        for i, (k, _) in enumerate(params):
            ax.plot(i + 1, pct_shift(fast_cog[k], slow_cog[k]), marker="*",
                    color="red", markersize=9, zorder=5)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(0, lw=0.6, color="k")
        if col == 0:
            ax.set_ylabel(f"Null %shift\n({n_done} splits)", fontsize=8)

    fig.suptitle(
        f"Cognitive-parameter shifts: observed fast/slow split (red ★) vs "
        f"exact permutation null ({n_done}/{len(splits)} balanced splits), "
        f"v4c_v3 model", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
