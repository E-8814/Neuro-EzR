"""
Compute split-half reliability (noise ceiling) for GECO eye-tracking
metrics — corrected methodology (v2).

Fix vs v1:
  v1 included w.total_reading_time / w.first_fixation_duration / w.gaze_duration
  for ALL participants in a half, including those who skipped the word
  (where these RTs are 0). Because skip patterns are highly consistent
  across participants, these zero-padded means inflate the per-word
  half/half correlation, especially for FFD.

  v2 mirrors the original methodology in
  src_v2/break_the_ceiling/noise_ceiling.py (lines 96-142):
    - skip is computed from all participants in the half (1 if skipped, else 0)
    - RT/FFD/Gaze are computed only from participants who DID NOT skip
      and who have a positive value for the metric
    - a word is included for a metric only if BOTH halves have at least
      one valid (non-skipped, positive) measurement for that metric

Usage:
    python compute_noise_ceiling_v2.py
    python compute_noise_ceiling_v2.py --n_splits 200 --seed 42
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
NOISE_CSV = RESULTS_DIR / "noise_ceiling_results_v2.csv"


def _spearman_brown(r_half: float) -> float:
    if r_half <= -1.0 or r_half >= 1.0:
        return r_half
    return 2.0 * r_half / (1.0 + r_half)


def compute_split_half(raw_dataset, n_splits: int = 200, seed: int = 42):
    rng = random.Random(seed)
    metrics = ["trt", "ffd", "gaze", "skip"]

    word_index = defaultdict(dict)
    for sd in raw_dataset:
        for i, w in enumerate(sd.words):
            key = (sd.text_id, sd.sentence_number, i)
            word_index[key][sd.participant_id] = w

    participants = sorted(set(sd.participant_id for sd in raw_dataset))
    n_p = len(participants)
    half_size = n_p // 2

    min_coverage = max(10, n_p - 2)
    usable_keys = [k for k, v in word_index.items() if len(v) >= min_coverage]
    print(f"  participants: {n_p}, half_size: {half_size}")
    print(f"  usable words (>= {min_coverage} participants): {len(usable_keys):,}")

    half_corrs = {m: [] for m in metrics}

    for split_idx in range(n_splits):
        perm = list(range(n_p))
        rng.shuffle(perm)
        group_a = set(participants[i] for i in perm[:half_size])
        group_b = set(participants[i] for i in perm[half_size:half_size * 2])

        a_arrays = {m: [] for m in metrics}
        b_arrays = {m: [] for m in metrics}

        for key in usable_keys:
            pdata = word_index[key]

            a_trts, a_ffds, a_gazes, a_skips = [], [], [], []
            for pid in group_a:
                if pid not in pdata:
                    continue
                w = pdata[pid]
                if w.was_skipped:
                    a_skips.append(1.0)
                else:
                    a_skips.append(0.0)
                    if w.total_reading_time > 0:
                        a_trts.append(w.total_reading_time)
                    if w.first_fixation_duration > 0:
                        a_ffds.append(w.first_fixation_duration)
                    if w.gaze_duration > 0:
                        a_gazes.append(w.gaze_duration)

            b_trts, b_ffds, b_gazes, b_skips = [], [], [], []
            for pid in group_b:
                if pid not in pdata:
                    continue
                w = pdata[pid]
                if w.was_skipped:
                    b_skips.append(1.0)
                else:
                    b_skips.append(0.0)
                    if w.total_reading_time > 0:
                        b_trts.append(w.total_reading_time)
                    if w.first_fixation_duration > 0:
                        b_ffds.append(w.first_fixation_duration)
                    if w.gaze_duration > 0:
                        b_gazes.append(w.gaze_duration)

            if a_trts and b_trts:
                a_arrays["trt"].append(float(np.mean(a_trts)))
                b_arrays["trt"].append(float(np.mean(b_trts)))
            if a_ffds and b_ffds:
                a_arrays["ffd"].append(float(np.mean(a_ffds)))
                b_arrays["ffd"].append(float(np.mean(b_ffds)))
            if a_gazes and b_gazes:
                a_arrays["gaze"].append(float(np.mean(a_gazes)))
                b_arrays["gaze"].append(float(np.mean(b_gazes)))
            if a_skips and b_skips:
                a_arrays["skip"].append(float(np.mean(a_skips)))
                b_arrays["skip"].append(float(np.mean(b_skips)))

        for m in metrics:
            a = np.array(a_arrays[m])
            b = np.array(b_arrays[m])
            if len(a) > 2 and a.std() > 0 and b.std() > 0:
                r = float(np.corrcoef(a, b)[0, 1])
                half_corrs[m].append(r)

        if (split_idx + 1) % 50 == 0:
            print(f"  split {split_idx + 1}/{n_splits}")

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
    print("\n=== Noise ceiling on GECO (corrected: skipped words filtered) ===")
    n_p_all = len(set(sd.participant_id for sd in raw))
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
            "n_participants": n_p_all,
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
