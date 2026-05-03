"""
Aggregate per-seed evaluation results into long-form CSVs for Table 1.

Reads:
    results/raw/<model>_seed<N>.json     # paper model
    results/raw/baselines/<model>_seed<N>.json   # if produced

Writes:
    results/per_seed_metrics.csv         # long form: (model, seed, dataset, metric, value)
    results/comparison_results.csv       # aggregated: (model, dataset, metric, mean, std, n_seeds)

Usage:
    python aggregate.py
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from paper_experiments import config


RESULTS_DIR = Path(_HERE) / "results"
RAW_DIR = RESULTS_DIR / "raw"
PER_SEED_CSV = RESULTS_DIR / "per_seed_metrics.csv"
COMPARISON_CSV = RESULTS_DIR / "comparison_results.csv"

METRICS = [
    "r_trt", "r_ffd", "r_gaze", "r_skip",
    "mae_trt", "mae_ffd", "mae_gaze",
    "bias_trt", "bias_ffd", "bias_gaze",
    "mean_pred_skip", "mean_human_skip",
]


def load_per_seed_records():
    records = []
    for path in sorted(RAW_DIR.glob("*.json")):
        with open(path) as f:
            payload = json.load(f)
        model = payload["model"]
        seed = payload["seed"]
        for ds_name, summary in payload["datasets"].items():
            for metric in METRICS:
                if metric in summary:
                    records.append({
                        "model": model,
                        "seed": seed,
                        "dataset": ds_name,
                        "metric": metric,
                        "value": summary[metric],
                    })
    return records


def write_per_seed_csv(records):
    PER_SEED_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(PER_SEED_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seed", "dataset", "metric", "value"])
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"Wrote {len(records)} rows to {PER_SEED_CSV}")


def aggregate_to_mean_std(records):
    """Group by (model, dataset, metric) and compute mean/std across seeds."""
    grouped = {}
    for r in records:
        key = (r["model"], r["dataset"], r["metric"])
        grouped.setdefault(key, []).append(r["value"])

    rows = []
    for (model, dataset, metric), values in sorted(grouped.items()):
        arr = np.array(values, dtype=float)
        rows.append({
            "model": model,
            "dataset": dataset,
            "metric": metric,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "n_seeds": len(arr),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        })
    return rows


def write_aggregated_csv(rows):
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "dataset", "metric", "mean", "std", "n_seeds", "min", "max"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows to {COMPARISON_CSV}")


def main():
    records = load_per_seed_records()
    if not records:
        print(f"No raw JSONs found in {RAW_DIR}. Run eval_all_models.py first.")
        return

    write_per_seed_csv(records)
    aggregated = aggregate_to_mean_std(records)
    write_aggregated_csv(aggregated)

    # Summary printout
    print("\n=== Summary (r_TRT on geco_test) ===")
    for r in aggregated:
        if r["dataset"] == "geco_test" and r["metric"] == "r_trt":
            print(f"  {r['model']:<35s}: {r['mean']:.3f} ± {r['std']:.3f}  (n={r['n_seeds']})")


if __name__ == "__main__":
    main()
