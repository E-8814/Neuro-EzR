"""
Compute split-half reliability (noise ceiling) for GECO eye-tracking
metrics.

Wraps the existing src_v2/break_the_ceiling/noise_ceiling.py logic
(which we re-import) and emits structured output for the paper pipeline.

Usage:
    python compute_noise_ceiling.py
    python compute_noise_ceiling.py --n_splits 200 --seed 42
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from paper_experiments import config
from paper_experiments.utils.load_data import _load_geco_raw


RESULTS_DIR = Path(_HERE) / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NOISE_CSV = RESULTS_DIR / "noise_ceiling_results.csv"


def _spearman_brown(r_half: float) -> float:
    """Spearman-Brown prophecy: r_full = 2 r_half / (1 + r_half).

    NB: this corrects from "split-half" (each half is half-sample) to
    "full sample" reliability. For our 14-participant corpus, more
    accurate would be a generalized version, but split-half is the
    canonical measure."""
    if r_half <= -1.0 or r_half >= 1.0:
        return r_half
    return 2.0 * r_half / (1.0 + r_half)


def compute_split_half(raw_dataset, n_splits: int = 200, seed: int = 42):
    """
    For each random split of participants into two equal halves:
      - average TRT/FFD/Gaze/Skip in each half (per word)
      - correlate the two halves
    Returns mean ± std of half correlation, plus Spearman-Brown estimate.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    metrics = ["trt", "ffd", "gaze", "skip"]

    # Build word_index[(text_id, sentence_number, word_position)] = {pid: word_data}
    word_index = defaultdict(dict)
    for sd in raw_dataset:
        for i, w in enumerate(sd.words):
            key = (sd.text_id, sd.sentence_number, i)
            word_index[key][sd.participant_id] = w

    participants = sorted(set(sd.participant_id for sd in raw_dataset))
    n_p = len(participants)
    half_size = n_p // 2

    # Filter to words seen by enough participants
    min_coverage = max(10, n_p - 2)
    usable_keys = [k for k, v in word_index.items() if len(v) >= min_coverage]

    half_corrs = {m: [] for m in metrics}

    for _ in range(n_splits):
        perm = list(range(n_p))
        rng.shuffle(perm)
        group_a = set(participants[i] for i in perm[:half_size])
        group_b = set(participants[i] for i in perm[half_size:half_size * 2])

        a_arrays = {m: [] for m in metrics}
        b_arrays = {m: [] for m in metrics}

        for key in usable_keys:
            pdata = word_index[key]
            a_data = [pdata[p] for p in pdata if p in group_a]
            b_data = [pdata[p] for p in pdata if p in group_b]
            if not a_data or not b_data:
                continue

            for m in metrics:
                if m == "trt":
                    av = np.mean([w.total_reading_time for w in a_data])
                    bv = np.mean([w.total_reading_time for w in b_data])
                elif m == "ffd":
                    av = np.mean([w.first_fixation_duration for w in a_data])
                    bv = np.mean([w.first_fixation_duration for w in b_data])
                elif m == "gaze":
                    av = np.mean([w.gaze_duration for w in a_data])
                    bv = np.mean([w.gaze_duration for w in b_data])
                elif m == "skip":
                    av = np.mean([1.0 if w.was_skipped else 0.0 for w in a_data])
                    bv = np.mean([1.0 if w.was_skipped else 0.0 for w in b_data])
                a_arrays[m].append(av)
                b_arrays[m].append(bv)

        for m in metrics:
            a = np.array(a_arrays[m])
            b = np.array(b_arrays[m])
            if len(a) > 2 and a.std() > 0 and b.std() > 0:
                r = float(np.corrcoef(a, b)[0, 1])
                half_corrs[m].append(r)

    return half_corrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_splits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading GECO raw data...")
    raw = _load_geco_raw()
    print(f"  {len(raw):,} per-participant observations")

    print(f"Computing split-half reliability over {args.n_splits} splits...")
    half_corrs = compute_split_half(raw, n_splits=args.n_splits, seed=args.seed)

    rows = []
    print("\n=== Noise ceiling on GECO (full-sample reliability via Spearman-Brown) ===")
    for metric, vals in half_corrs.items():
        if not vals:
            continue
        arr = np.array(vals)
        half_mean = float(arr.mean())
        full_est = _spearman_brown(half_mean)
        rows.append({
            "corpus": "geco",
            "metric": f"r_{metric}",
            "half_corr_mean": half_mean,
            "half_corr_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "full_corr_estimate": full_est,
            "n_splits": len(arr),
            "n_participants": 14,
        })
        print(f"  r_{metric:<5s}: split-half = {half_mean:.3f} ± "
              f"{arr.std(ddof=1):.3f}, Spearman-Brown full = {full_est:.3f}")

    with open(NOISE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "corpus", "metric",
                "half_corr_mean", "half_corr_std",
                "full_corr_estimate", "n_splits", "n_participants",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {NOISE_CSV}")


if __name__ == "__main__":
    main()
