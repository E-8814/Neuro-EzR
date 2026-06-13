"""
Aggregate exp11 results into the fair-comparison table.

Reads:
    results/raw/v4c_v3_dualctx_next*_seed*.json
    results/raw/baselines/*_seed*.json

Writes:
    results/fair_comparison.csv
and prints a markdown table: per model, seed-mean (and std where >1 seed)
of time metrics (all words) and skip metrics on the comparable
population (words 1..L-1), plus skip-all-words for the baselines as the
sanity link to the published numbers.

Usage:
    python -u aggregate.py
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RAW = Path(_HERE) / "results" / "raw"
OUT_CSV = Path(_HERE) / "results" / "fair_comparison.csv"

TIME_KEYS = ["r_trt", "r_ffd", "r_gaze", "mae_trt", "mae_ffd", "mae_gaze"]
SKIP_KEYS = ["r_skip", "skip_auc", "skip_brier", "mae_skip"]


def load_all():
    rows = defaultdict(lambda: defaultdict(list))  # model -> corpus -> list of dict
    for path in sorted(RAW.rglob("*_seed*.json")):
        d = json.loads(path.read_text())
        model = d["model"]
        for corpus, block in d["datasets"].items():
            flat = {}
            for k in TIME_KEYS:
                if k in block:
                    flat[k] = block[k]
            if "skip_cmp" in block:        # baselines: nested
                for k in SKIP_KEYS:
                    flat[f"cmp_{k}"] = block["skip_cmp"][k]
                for k in SKIP_KEYS:
                    flat[f"all_{k}"] = block["skip_all"][k]
            else:                          # v4c_v3: skip fields at top level
                for k in SKIP_KEYS:
                    if k in block:
                        flat[f"cmp_{k}"] = block[k]
            rows[model][corpus].append(flat)
    return rows


def fmt(vals, prec=3):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return "—"
    m = np.mean(vals)
    if len(vals) > 1:
        return f"{m:.{prec}f}±{np.std(vals):.{prec}f}"
    return f"{m:.{prec}f}"


def main():
    rows = load_all()
    order = ["linear_regression", "gpt2_surprisal", "lightgbm",
             "bert_regression", "ohio_state_roberta",
             "v4c_v3_dualctx_next_no_ai", "v4c_v3_dualctx_next"]
    cols = (["r_trt", "r_ffd", "r_gaze"]
            + [f"cmp_{k}" for k in SKIP_KEYS]
            + ["all_r_skip"])

    csv_rows = []
    for corpus in ("geco_test", "provo"):
        print(f"\n### {corpus}  (skip = words 1..L-1; 'all_r_skip' = legacy all-words)\n")
        header = "| model | " + " | ".join(cols) + " |"
        print(header)
        print("|" + "---|" * (len(cols) + 1))
        for model in order:
            if model not in rows or corpus not in rows[model]:
                continue
            seed_dicts = rows[model][corpus]
            cells = []
            for c in cols:
                vals = [d.get(c) for d in seed_dicts if c in d]
                cells.append(fmt(vals))
                csv_rows.append({
                    "model": model, "corpus": corpus, "metric": c,
                    "mean": (np.mean([v for v in vals if v is not None])
                             if vals else None),
                    "std": (np.std([v for v in vals if v is not None])
                            if len(vals) > 1 else None),
                    "n_seeds": len(vals),
                })
            print(f"| {model} | " + " | ".join(cells) + " |")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "corpus", "metric",
                                          "mean", "std", "n_seeds"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
