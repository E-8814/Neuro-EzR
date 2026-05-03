"""
Aggregate per-seed evaluation results into long-form CSVs for Table 1.

V2 vs v1:
  - Excludes toronto_cl_roberta from the comparison (paper drops that
    baseline since its multi-seed support was added but never fully
    validated).
  - Reads JSONs from BOTH `results/raw/` and `results/raw/baselines/`
    (and any additional dirs passed via --raw_dirs), so paper-model
    seeds trained by partB_parallel are picked up the same way as
    seeds trained by phase B (they all live at canonical checkpoint
    paths and produce JSONs in the same `raw/` location after
    eval_all_models.py runs).
  - Output files have `_v2` suffix so v1 outputs aren't overwritten:
      results/per_seed_metrics_v2.csv
      results/comparison_results_v2.csv

Usage:
    python aggregate_v2.py
    python aggregate_v2.py --raw_dirs results/raw  results/raw/baselines  some/other/dir
"""

import argparse
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
DEFAULT_RAW_DIRS = [
    RESULTS_DIR / "raw",
    RESULTS_DIR / "raw" / "baselines",
]
PER_SEED_CSV = RESULTS_DIR / "per_seed_metrics_v2.csv"
COMPARISON_CSV = RESULTS_DIR / "comparison_results_v2.csv"

METRICS = [
    "r_trt", "r_ffd", "r_gaze", "r_skip",
    "mae_trt", "mae_ffd", "mae_gaze",
    "bias_trt", "bias_ffd", "bias_gaze",
    "mean_pred_skip", "mean_human_skip",
]

# Models to drop from the v2 comparison.
EXCLUDE_MODELS = {"toronto_cl_roberta"}


def _excluded(model_name: str) -> bool:
    """True if this model should not appear in the v2 comparison."""
    return any(name in model_name for name in EXCLUDE_MODELS)


def load_per_seed_records(raw_dirs):
    records = []
    seen_paths = set()
    for d in raw_dirs:
        d = Path(d)
        if not d.exists():
            print(f"  [skip] raw dir does not exist: {d}")
            continue
        for path in sorted(d.glob("*.json")):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                with open(path) as f:
                    payload = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  [warn] could not parse {path}: {e}")
                continue
            model = payload.get("model")
            seed = payload.get("seed")
            if model is None or seed is None:
                print(f"  [warn] {path.name} missing 'model' or 'seed' — skipping")
                continue
            if _excluded(model):
                print(f"  [exclude] {path.name} (model={model})")
                continue
            for ds_name, summary in payload.get("datasets", {}).items():
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
        writer = csv.DictWriter(
            f, fieldnames=["model", "seed", "dataset", "metric", "value"],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"Wrote {len(records)} rows to {PER_SEED_CSV}")


def aggregate_to_mean_std(records):
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
            fieldnames=["model", "dataset", "metric", "mean", "std",
                        "n_seeds", "min", "max"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows to {COMPARISON_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dirs", nargs="+", default=None,
        help="Directories containing per-seed JSONs. Default: "
             "results/raw and results/raw/baselines.",
    )
    args = parser.parse_args()

    raw_dirs = args.raw_dirs or DEFAULT_RAW_DIRS
    print("Looking for raw JSONs in:")
    for d in raw_dirs:
        print(f"  - {d}")
    print(f"Excluding models: {sorted(EXCLUDE_MODELS)}")

    records = load_per_seed_records(raw_dirs)
    if not records:
        print(
            "\nNo raw JSONs found. To produce them:\n"
            "  python /home/u384661/Neuro_EZR/src_v2/paper_experiments/"
            "exp01_main_comparison/eval_all_models.py\n"
            "(That writes paper-model JSONs to results/raw/. Baseline "
            "JSONs come from src_v2/evaluation/eval_all_models_v2.py.)"
        )
        return

    write_per_seed_csv(records)
    aggregated = aggregate_to_mean_std(records)
    write_aggregated_csv(aggregated)

    print("\n=== Summary (r_TRT on geco_test) ===")
    for r in aggregated:
        if r["dataset"] == "geco_test" and r["metric"] == "r_trt":
            print(f"  {r['model']:<35s}: {r['mean']:.3f} ± {r['std']:.3f}  (n={r['n_seeds']})")
    print("\n=== Summary (r_skip on geco_test) ===")
    for r in aggregated:
        if r["dataset"] == "geco_test" and r["metric"] == "r_skip":
            print(f"  {r['model']:<35s}: {r['mean']:.3f} ± {r['std']:.3f}  (n={r['n_seeds']})")


if __name__ == "__main__":
    main()
