"""
Aggregate the per-split JSONs into:
    - results/perm_distribution.csv     long-form (index, T, ...)
    - results/perm_summary.json         {n, p_value, T_obs, T_obs_index, ...}
    - results/perm_histogram.pdf        small histogram panel for the paper

Reads either results/perms/ (cached path) or results/perms_live/ (fallback),
preferring whichever has more completed splits.

The OBSERVED T is computed from the ACTUAL fast/slow split's index in the
canonical enumeration. If that index is among the completed perms, we
reuse its T. Otherwise we fall back to the published T from the
sanity-check JSON.

Usage:
    python -u .../04_aggregate.py
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _HERE)

from paper_experiments.utils.load_data import load_geco_per_participant

from _cog_fit import enumerate_balanced_splits, split_index, dissociation_T
from fit_per_group import FAST_READERS, SLOW_READERS


RESULTS_DIR = Path(_HERE) / "results"
PERMS_CACHED = RESULTS_DIR / "perms"
PERMS_LIVE   = RESULTS_DIR / "perms_live"
SANITY_JSON  = RESULTS_DIR / "sanity_check.json"


def _load_perm_jsons(perm_dir: Path):
    rows = []
    for path in sorted(perm_dir.glob("perm_*.json")):
        if path.name.endswith(".error.json"):
            continue
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  [warn] could not parse {path.name}, skipping")
            continue
        rows.append(d)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["auto", "cached", "live"],
                        default="auto",
                        help=("Which perm directory to aggregate. 'auto' picks "
                              "the one with more completed perms."))
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip matplotlib import / histogram render.")
    args = parser.parse_args()

    # ---- Decide which directory to aggregate ---- #
    n_cached = len(list(PERMS_CACHED.glob("perm_*.json"))) if PERMS_CACHED.exists() else 0
    n_live   = len(list(PERMS_LIVE.glob("perm_*.json")))   if PERMS_LIVE.exists()   else 0
    if args.source == "cached":
        chosen = PERMS_CACHED
    elif args.source == "live":
        chosen = PERMS_LIVE
    else:
        chosen = PERMS_CACHED if n_cached >= n_live else PERMS_LIVE
    print(f"Aggregating from: {chosen}  "
          f"(cached={n_cached}, live={n_live})")

    if not chosen.exists() or not list(chosen.glob("perm_*.json")):
        print("No perm JSONs to aggregate.")
        sys.exit(1)

    rows = _load_perm_jsons(chosen)
    n = len(rows)
    print(f"Loaded {n} perm JSONs.")

    # ---- Build canonical split index -> T ---- #
    by_idx = {r["index"]: r for r in rows}

    # ---- Find observed split's index in the enumeration ---- #
    by_p = load_geco_per_participant(split="train")
    pids = sorted(by_p.keys())
    splits = enumerate_balanced_splits(pids)
    fast = sorted(FAST_READERS & set(pids))
    slow = sorted(SLOW_READERS & set(pids))
    obs_idx = split_index(splits, fast, slow)
    print(f"Observed (fast vs slow) split is canonical index {obs_idx}.")

    # ---- Resolve T_obs ---- #
    T_obs = None
    obs_source = None
    if obs_idx in by_idx:
        T_obs = by_idx[obs_idx]["T"]
        obs_source = "perm_run"
    elif SANITY_JSON.exists():
        sanity = json.loads(SANITY_JSON.read_text())
        T_obs = sanity.get("T_cached", {}).get("T")
        if T_obs is None:
            T_obs = sanity.get("T_published", {}).get("T")
        obs_source = "sanity_check_json"
    if T_obs is None:
        # Last-resort: published proxy
        from _cog_fit import dissociation_T as _dT
        from sanity_constants import PUBLISHED_FAST, PUBLISHED_SLOW  # not present; kept for future
        T_obs = None
    print(f"T_obs = {T_obs!r}  (source: {obs_source})")

    # ---- Build the null distribution (excluding the observed split) ---- #
    null_T = sorted(r["T"] for r in rows if r["index"] != obs_idx)
    null_arr = np.array(null_T, dtype=float)

    # ---- p-value ---- #
    if T_obs is None:
        p_value = None
    else:
        # One-sided: how often does a random split produce a dissociation
        # at least as extreme (large) as the observed?
        n_perm = len(null_arr)
        n_at_or_above = int((null_arr >= float(T_obs)).sum())
        # Conservative (1+x)/(N+1) form to handle the zero-tail case.
        p_value = (1 + n_at_or_above) / (n_perm + 1)

    # ---- Long-form CSV ---- #
    csv_path = RESULTS_DIR / "perm_distribution.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "index", "T",
            "mean_abs_pct_lexical", "mean_abs_pct_motor",
            "abs_pct_delta", "abs_pct_lambda_refix", "abs_pct_epsilon",
            "abs_pct_M1", "abs_pct_M2_eq_I", "abs_pct_skip_temperature",
            "is_observed", "elapsed_seconds",
        ])
        for r in rows:
            tc = r.get("T_components", {})
            writer.writerow([
                r["index"], r["T"],
                tc.get("mean_abs_pct_lexical"),
                tc.get("mean_abs_pct_motor"),
                tc.get("abs_pct_delta"),
                tc.get("abs_pct_lambda_refix"),
                tc.get("abs_pct_epsilon"),
                tc.get("abs_pct_M1"),
                tc.get("abs_pct_M2_eq_I"),
                tc.get("abs_pct_skip_temperature"),
                int(r["index"] == obs_idx),
                r.get("elapsed_seconds"),
            ])
    print(f"Wrote {csv_path}")

    # ---- Summary JSON ---- #
    summary = {
        "n_permutations": n,
        "n_excluding_observed": int(len(null_arr)),
        "T_obs": T_obs,
        "T_obs_source": obs_source,
        "T_obs_canonical_index": obs_idx,
        "p_value_one_sided_geq": p_value,
        "null_min": float(null_arr.min()) if len(null_arr) else None,
        "null_max": float(null_arr.max()) if len(null_arr) else None,
        "null_mean": float(null_arr.mean()) if len(null_arr) else None,
        "null_std": float(null_arr.std()) if len(null_arr) else None,
        "null_quantiles": {
            "q05": float(np.quantile(null_arr, 0.05)) if len(null_arr) else None,
            "q50": float(np.quantile(null_arr, 0.50)) if len(null_arr) else None,
            "q95": float(np.quantile(null_arr, 0.95)) if len(null_arr) else None,
            "q99": float(np.quantile(null_arr, 0.99)) if len(null_arr) else None,
        },
        "fast_readers": fast,
        "slow_readers": slow,
        "source_dir": str(chosen),
    }
    summary_path = RESULTS_DIR / "perm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"Wrote {summary_path}")

    print(f"\nNull distribution (n={len(null_arr)}):")
    print(f"  min:  {summary['null_min']:+8.3f}")
    print(f"  q05:  {summary['null_quantiles']['q05']:+8.3f}")
    print(f"  q50:  {summary['null_quantiles']['q50']:+8.3f}")
    print(f"  q95:  {summary['null_quantiles']['q95']:+8.3f}")
    print(f"  q99:  {summary['null_quantiles']['q99']:+8.3f}")
    print(f"  max:  {summary['null_max']:+8.3f}")
    print(f"\nT_obs = {T_obs!r}")
    print(f"p (T_perm >= T_obs) = {p_value!r}")

    # ---- Histogram panel ---- #
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plot.")
            return

        fig, ax = plt.subplots(figsize=(4.0, 2.8))
        ax.hist(null_arr, bins=40, color="0.6", edgecolor="0.3", linewidth=0.4)
        if T_obs is not None:
            ax.axvline(T_obs, color="crimson", linewidth=1.5,
                       label=f"observed T={T_obs:+.2f}")
        ax.set_xlabel(r"Dissociation $T$ = mean$|\%\Delta_{\mathrm{lex}}| -$ mean$|\%\Delta_{\mathrm{mot}}|$")
        ax.set_ylabel("count (random 7/7 splits)")
        n_label = "all 1,716 balanced splits" if len(null_arr) >= 1000 else f"{len(null_arr)} random splits"
        ax.set_title(f"Permutation null  ({n_label})")
        if T_obs is not None and p_value is not None:
            ax.text(0.97, 0.95,
                    f"p = {p_value:.4g}",
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                              edgecolor="0.3", linewidth=0.5))
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        out_pdf = RESULTS_DIR / "perm_histogram.pdf"
        fig.savefig(str(out_pdf), bbox_inches="tight")
        print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
