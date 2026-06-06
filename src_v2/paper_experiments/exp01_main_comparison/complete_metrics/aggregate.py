"""
Aggregate the augmented-metric JSONs into long-form CSVs for Table 1.

Reads from:
    complete_metrics/results/raw/<model>_seed<N>.json          (paper model + EZ classical)
    complete_metrics/results/raw/baselines/<model>_seed<N>.json (5 baselines)

Writes:
    complete_metrics/results/per_seed_metrics_complete.csv
        long form: one row per (model, seed, dataset, metric, value)
    complete_metrics/results/comparison_results_complete.csv
        aggregated: one row per (model, dataset, metric) with
        mean / std / n_seeds across seeds (single-seed models report n=1)

Metric set covered:
    r_trt, r_ffd, r_gaze, r_skip
    mae_trt, mae_ffd, mae_gaze, mae_skip       (skip in raw [0, 1] units)
    bias_trt, bias_ffd, bias_gaze, bias_skip
    mean_pred_skip, mean_human_skip, n_words

Usage:
    python -u .../aggregate.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = Path(_HERE) / "results" / "raw"
RAW_BASELINES_DIR = RAW_DIR / "baselines"
OUT_DIR = Path(_HERE) / "results"

PER_SEED_CSV = OUT_DIR / "per_seed_metrics_complete.csv"
SUMMARY_CSV = OUT_DIR / "comparison_results_complete.csv"


METRICS_TO_AGGREGATE = [
    "r_trt", "r_ffd", "r_gaze", "r_skip",
    "mae_trt", "mae_ffd", "mae_gaze", "mae_skip",
    "bias_trt", "bias_ffd", "bias_gaze", "bias_skip",
    "mean_pred_skip", "mean_human_skip", "n_words",
]


# Models we expect to find. Other names will still be picked up if they
# follow the JSON convention; these define the canonical row order.
EXPECTED_MODELS = [
    "linear_regression",
    "gpt2_surprisal",
    "lightgbm",
    "bert_regression",
    "ohio_state_roberta",
    "ez_reader_classical",
    "v4c_v2_dualctx",
]


def _scan_json_dir(d: Path):
    """Yield (model_name, seed, payload) for every JSON in dir d."""
    if not d.exists():
        return
    for path in sorted(d.glob("*.json")):
        # filename pattern: <model>_seed<N>.json
        stem = path.stem
        if "_seed" not in stem:
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  [warn] could not parse {path.name}, skipping")
            continue
        model = payload.get("model")
        seed = payload.get("seed")
        if model is None or seed is None:
            print(f"  [warn] {path.name} missing model/seed; skipping")
            continue
        yield model, int(seed), payload


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all (model, seed, dataset, metric) -> value rows
    rows = []  # list of dicts for per_seed CSV
    by_md = defaultdict(list)  # (model, dataset, metric) -> list of values

    for source in (RAW_DIR, RAW_BASELINES_DIR):
        for model, seed, payload in _scan_json_dir(source):
            for dataset, metrics in payload.get("datasets", {}).items():
                for m in METRICS_TO_AGGREGATE:
                    if m not in metrics:
                        continue
                    val = float(metrics[m])
                    rows.append({
                        "model": model,
                        "seed": seed,
                        "dataset": dataset,
                        "metric": m,
                        "value": val,
                    })
                    by_md[(model, dataset, m)].append(val)

    # ---- Write per-seed CSV ---- #
    rows.sort(key=lambda r: (
        EXPECTED_MODELS.index(r["model"]) if r["model"] in EXPECTED_MODELS else 999,
        r["model"], r["dataset"], r["metric"], r["seed"],
    ))
    with open(PER_SEED_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "seed", "dataset", "metric", "value"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {PER_SEED_CSV}  ({len(rows)} rows)")

    # ---- Write aggregated comparison CSV ---- #
    summary_rows = []
    for (model, dataset, metric), vals in by_md.items():
        arr = np.asarray(vals, dtype=float)
        summary_rows.append({
            "model":   model,
            "dataset": dataset,
            "metric":  metric,
            "mean":    float(arr.mean()),
            "std":     float(arr.std(ddof=0)),
            "n_seeds": len(arr),
        })
    summary_rows.sort(key=lambda r: (
        EXPECTED_MODELS.index(r["model"]) if r["model"] in EXPECTED_MODELS else 999,
        r["model"], r["dataset"], r["metric"],
    ))
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dataset", "metric",
                                          "mean", "std", "n_seeds"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"Wrote {SUMMARY_CSV}  ({len(summary_rows)} rows)")

    # ---- Print a compact human-readable table ---- #
    print("\n========== Compact summary ==========")
    print(f"{'model':<22s} {'dataset':<10s} "
          f"{'r_trt':>7s} {'r_ffd':>7s} {'r_gaze':>7s} {'r_skip':>7s}  "
          f"{'mae_trt':>8s} {'mae_ffd':>8s} {'mae_gaze':>8s} {'mae_skip':>9s}")
    print("-" * 110)

    by_model_ds = defaultdict(dict)
    for r in summary_rows:
        by_model_ds[(r["model"], r["dataset"])][r["metric"]] = r["mean"]

    ordered = []
    for m in EXPECTED_MODELS:
        for ds in ("geco_test", "provo"):
            if (m, ds) in by_model_ds:
                ordered.append((m, ds))

    for model, ds in ordered:
        d = by_model_ds[(model, ds)]
        def g(k):
            return d.get(k, float("nan"))
        print(f"{model:<22s} {ds:<10s} "
              f"{g('r_trt'):>+7.3f} {g('r_ffd'):>+7.3f} "
              f"{g('r_gaze'):>+7.3f} {g('r_skip'):>+7.3f}  "
              f"{g('mae_trt'):>8.2f} {g('mae_ffd'):>8.2f} "
              f"{g('mae_gaze'):>8.2f} {g('mae_skip'):>9.4f}")

    print("\nNote: mae_skip is in raw [0, 1] fraction-of-readers units.")
    print("      mae_trt/ffd/gaze in milliseconds.")


if __name__ == "__main__":
    main()
