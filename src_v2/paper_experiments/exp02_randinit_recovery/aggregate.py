"""
Aggregate randinit checkpoints into the recovery_results.csv used by
plot_recovery.py and the paper Figure 2.

For each seed:
    Load best_model.pt
    Read sampled_init (the perturbed init values, saved at construction)
    Read cog_params (final converged values)
    Compute alpha1_reichle / alpha2_reichle from base_offset / freq_coef
    Emit one row per (seed, parameter)

Writes:
    results/recovery_results.csv     (long-form)
    results/recovery_summary.csv     (per-parameter aggregates)

Usage:
    python aggregate.py
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from paper_experiments import config

RESULTS_DIR = Path(_HERE) / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RECOVERY_CSV = RESULTS_DIR / "recovery_results.csv"
SUMMARY_CSV = RESULTS_DIR / "recovery_summary.csv"


# Map (sampled_init key) → (paper-name, derived?)
# All Reichle-targeted parameters come from REICHLE_TARGETS.
def derive_init_for_named_param(sampled, name):
    """Derive the *paper-named* parameter value from the sampled init dict."""
    if name == "alpha1_reichle":
        return sampled["l1_base_offset"] - 2.0 * sampled["l1_freq_coef"]
    if name == "alpha2_reichle":
        return -sampled["l1_freq_coef"] / 5.0
    if name == "epsilon":
        return 1.0 + sampled["epsilon_minus_1"]
    if name == "M1":
        return sampled["M1"]
    if name == "M2_eq_I":
        return sampled["M2I"]
    if name == "delta":
        return sampled["delta"]
    if name == "lambda_refix":
        return sampled["lambda_refix"]
    raise KeyError(f"Don't know how to derive init for {name}")


def derive_converged_for_named_param(cog_params, name):
    """Read converged value from cog_params dict (saved by training script)."""
    mapping = {
        "alpha1_reichle": "alpha1_reichle",
        "alpha2_reichle": "alpha2_reichle",
        "epsilon": "epsilon",
        "M1": "M1",
        "M2_eq_I": "M2",   # cog_params stores M2 (which == I, tied)
        "delta": "delta",
        "lambda_refix": "lambda_refix",
    }
    return cog_params[mapping[name]]


def main():
    rows = []

    for seed in config.SEEDS:
        ckpt_path = config.randinit_ckpt_path(seed)
        if not ckpt_path.exists():
            print(f"  [missing] seed={seed}: {ckpt_path}")
            continue

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sampled = ckpt.get("sampled_init")
        cog = ckpt.get("cog_params")
        if sampled is None or cog is None:
            print(f"  [malformed] seed={seed}: missing sampled_init or cog_params")
            continue

        for param_name, reichle_target in config.REICHLE_TARGETS.items():
            init_val = derive_init_for_named_param(sampled, param_name)
            conv_val = derive_converged_for_named_param(cog, param_name)
            rows.append({
                "seed": seed,
                "param": param_name,
                "init_value": init_val,
                "converged_value": conv_val,
                "reichle_target": reichle_target,
                "abs_drift_from_init": abs(conv_val - init_val),
                "abs_drift_from_reichle": abs(conv_val - reichle_target),
                "abs_init_drift_from_reichle": abs(init_val - reichle_target),
            })

    # Long-form CSV
    with open(RECOVERY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "param",
            "init_value", "converged_value", "reichle_target",
            "abs_drift_from_init",
            "abs_drift_from_reichle",
            "abs_init_drift_from_reichle",
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows to {RECOVERY_CSV}")

    # Per-parameter summary
    summary_rows = []
    by_param = {}
    for r in rows:
        by_param.setdefault(r["param"], []).append(r)
    for param, rs in sorted(by_param.items()):
        inits = np.array([r["init_value"] for r in rs])
        convs = np.array([r["converged_value"] for r in rs])
        reichle = rs[0]["reichle_target"]
        summary_rows.append({
            "param": param,
            "reichle_target": reichle,
            "n_seeds": len(rs),
            "mean_init": float(inits.mean()),
            "std_init": float(inits.std(ddof=1)) if len(inits) > 1 else 0.0,
            "mean_converged": float(convs.mean()),
            "std_converged": float(convs.std(ddof=1)) if len(convs) > 1 else 0.0,
            "tightening_ratio": float(convs.std(ddof=1) / inits.std(ddof=1))
                if len(inits) > 1 and inits.std(ddof=1) > 0 else 0.0,
            "mean_abs_distance_to_reichle": float(np.mean(np.abs(convs - reichle))),
            "max_abs_distance_to_reichle": float(np.max(np.abs(convs - reichle))),
        })

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(f"Wrote {len(summary_rows)} rows to {SUMMARY_CSV}")

    print("\n=== Recovery summary ===")
    print(f"  {'param':<20s} {'reichle':>10s} {'mean_conv':>10s}±{'std':>5s}  "
          f"{'tighten':>7s}  {'avg_dist':>9s}")
    for r in summary_rows:
        print(f"  {r['param']:<20s} {r['reichle_target']:>10.3f} "
              f"{r['mean_converged']:>10.3f}±{r['std_converged']:>5.3f}  "
              f"{r['tightening_ratio']:>7.3f}  "
              f"{r['mean_abs_distance_to_reichle']:>9.4f}")


if __name__ == "__main__":
    main()
