"""
Compute model-vs-ceiling curve on Provo.

For multiple data fractions f ∈ [0.1, 0.2, ..., 1.0]:
  - subsample participants to f × n_participants
  - compute split-half reliability of the subsample (= ceiling at f)
  - evaluate paper model on the same subsample (= model at f)
  - record gap

The curve shows: as you have more data, ceiling rises, model rises;
the persistent gap (if any) is feature-limited (not data-limited).

Usage:
    python compute_ceiling_curve.py
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "archive", "original_ezreader"))

from paper_experiments import config
from paper_experiments.utils.load_model import load_paper_model
from paper_experiments.utils.load_data import load_provo_aggregated, load_subtlex
from paper_experiments.utils.eval_metrics import corr, eval_predictions_on_aggregated
from data_loader import load_provo, aggregate_by_sentence


RESULTS_DIR = Path(_HERE) / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CURVE_CSV = RESULTS_DIR / "ceiling_curve_results.csv"


def _spearman_brown(r_half):
    if r_half <= -1.0 or r_half >= 1.0:
        return r_half
    return 2.0 * r_half / (1.0 + r_half)


def split_half_at_subsample(raw_provo, participant_subset, n_splits=50, seed=42):
    """Split-half reliability with only `participant_subset` selected."""
    metrics = ["trt", "ffd", "gaze", "skip"]
    rng = random.Random(seed)

    word_index = defaultdict(dict)
    for sd in raw_provo:
        if sd.participant_id not in participant_subset:
            continue
        for i, w in enumerate(sd.words):
            key = (sd.text_id, sd.sentence_number, i)
            word_index[key][sd.participant_id] = w

    participants = sorted(participant_subset)
    n_p = len(participants)
    half = n_p // 2
    if half < 2:
        return {m: [0.0] for m in metrics}

    min_coverage = max(2, n_p - 1)
    usable = [k for k, v in word_index.items() if len(v) >= min_coverage]

    half_corrs = {m: [] for m in metrics}
    for _ in range(n_splits):
        perm = list(range(n_p))
        rng.shuffle(perm)
        ga = set(participants[i] for i in perm[:half])
        gb = set(participants[i] for i in perm[half:half * 2])

        for m in metrics:
            a, b = [], []
            for key in usable:
                pdata = word_index[key]
                ad = [pdata[p] for p in pdata if p in ga]
                bd = [pdata[p] for p in pdata if p in gb]
                if not ad or not bd:
                    continue
                if m == "trt":
                    a.append(np.mean([w.total_reading_time for w in ad]))
                    b.append(np.mean([w.total_reading_time for w in bd]))
                elif m == "ffd":
                    a.append(np.mean([w.first_fixation_duration for w in ad]))
                    b.append(np.mean([w.first_fixation_duration for w in bd]))
                elif m == "gaze":
                    a.append(np.mean([w.gaze_duration for w in ad]))
                    b.append(np.mean([w.gaze_duration for w in bd]))
                elif m == "skip":
                    a.append(np.mean([1.0 if w.was_skipped else 0.0 for w in ad]))
                    b.append(np.mean([1.0 if w.was_skipped else 0.0 for w in bd]))
            if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
                half_corrs[m].append(float(np.corrcoef(a, b)[0, 1]))
    return half_corrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--n_splits", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading paper model + data...")
    model, _ = load_paper_model(seed=args.seed, device=device)
    subtlex = load_subtlex()

    raw_provo = load_provo(str(config.PROVO_FILE))
    all_participants = sorted(set(sd.participant_id for sd in raw_provo))
    print(f"  Provo: {len(raw_provo)} obs, {len(all_participants)} participants")

    rng = random.Random(args.seed)
    rows = []

    for frac in args.fractions:
        n_keep = max(2, int(round(len(all_participants) * frac)))
        keep = set(rng.sample(all_participants, n_keep))
        print(f"\n>> fraction={frac:.2f} → {n_keep} participants")

        # Aggregate Provo with only these participants
        kept_raw = [sd for sd in raw_provo if sd.participant_id in keep]
        ds = aggregate_by_sentence(kept_raw, min_participants=2)
        if not ds:
            print("    (skipping; no aggregated sentences at this fraction)")
            continue

        # Model evaluation
        _, summary = eval_predictions_on_aggregated(model, ds, device, subtlex)

        # Noise ceiling at this fraction
        half_corrs = split_half_at_subsample(
            raw_provo, keep, n_splits=args.n_splits, seed=args.seed,
        )

        for metric_short in ["trt", "ffd", "gaze", "skip"]:
            metric_full = f"r_{metric_short}"
            model_r = summary.get(metric_full, 0.0)
            half_arr = np.array(half_corrs.get(metric_short, [0.0]))
            half_mean = float(half_arr.mean()) if len(half_arr) else 0.0
            ceiling = _spearman_brown(half_mean)
            rows.append({
                "data_fraction": frac,
                "n_participants": n_keep,
                "metric": metric_full,
                "model_r": model_r,
                "ceiling_r": ceiling,
                "gap": ceiling - model_r,
            })
            print(f"   {metric_full}: model={model_r:.3f}  ceiling={ceiling:.3f}  gap={ceiling - model_r:+.3f}")

    with open(CURVE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["data_fraction", "n_participants", "metric",
                        "model_r", "ceiling_r", "gap"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {CURVE_CSV}")


if __name__ == "__main__":
    main()
