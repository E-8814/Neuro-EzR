"""
Compute the noise ceiling for GECO eye-tracking data.

Split-half reliability: divide the 14 participants into two groups of 7,
average each group separately, and correlate their word-level TRT/FFD/Gaze/Skip.
Repeat with many random splits and report mean + std of the correlation.

This gives the theoretical maximum correlation any model can achieve,
since the model is predicting averaged data that itself has noise.

Also applies Spearman-Brown correction:
    r_full = 2 * r_half / (1 + r_half)
to estimate the reliability of the full 14-participant average.

Usage:
    python3 -u src_v2/break_the_ceiling/noise_ceiling.py
"""

import os
import sys
import random
import itertools
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'archive', 'original_ezreader'))
from geco_loader import load_geco, split_geco


def compute_split_half_reliability(raw_dataset, n_splits=200, seed=42):
    """
    Compute split-half reliability of GECO eye-tracking averages.

    For each random split of participants into two halves:
      1. Average TRT/FFD/Gaze/Skip within each half
      2. Correlate the two halves at word level
      3. Apply Spearman-Brown correction for full-sample reliability

    Returns dict with per-metric results.
    """
    rng = random.Random(seed)

    # Get all unique participants
    participants = sorted(set(sd.participant_id for sd in raw_dataset))
    n_participants = len(participants)
    half_size = n_participants // 2

    print(f"Participants: {n_participants} ({participants})")
    print(f"Split size: {half_size} vs {n_participants - half_size}")

    # Index: (text_id, sentence_number, word_position) -> {participant_id: WordData}
    print("Building word-level index...")
    word_index = defaultdict(dict)
    for sd in raw_dataset:
        for i, w in enumerate(sd.words):
            key = (sd.text_id, sd.sentence_number, i)
            word_index[key][sd.participant_id] = w

    # Filter to words seen by all participants (for clean comparison)
    full_coverage_keys = [
        key for key, pdata in word_index.items()
        if len(pdata) >= n_participants
    ]
    print(f"Words with full participant coverage: {len(full_coverage_keys):,}")
    print(f"Total word positions in index: {len(word_index):,}")

    # Use words with at least 10 participants for robustness
    min_coverage = max(10, n_participants - 2)
    usable_keys = [
        key for key, pdata in word_index.items()
        if len(pdata) >= min_coverage
    ]
    print(f"Words with >= {min_coverage} participants: {len(usable_keys):,}")

    # Run splits
    metrics = ['trt', 'ffd', 'gaze', 'skip']
    half_corrs = {m: [] for m in metrics}
    full_corrs = {m: [] for m in metrics}

    print(f"\nRunning {n_splits} random splits...")
    for split_idx in range(n_splits):
        shuffled = participants.copy()
        rng.shuffle(shuffled)
        group_a = set(shuffled[:half_size])
        group_b = set(shuffled[half_size:])

        # Compute averages for each group at each word position
        a_trt, b_trt = [], []
        a_ffd, b_ffd = [], []
        a_gaze, b_gaze = [], []
        a_skip, b_skip = [], []

        for key in usable_keys:
            pdata = word_index[key]

            # Group A averages
            a_trts, a_ffds, a_gazes, a_skips_list = [], [], [], []
            for pid in group_a:
                if pid not in pdata:
                    continue
                w = pdata[pid]
                if w.was_skipped:
                    a_skips_list.append(1.0)
                else:
                    a_skips_list.append(0.0)
                    if w.total_reading_time > 0:
                        a_trts.append(w.total_reading_time)
                    if w.first_fixation_duration > 0:
                        a_ffds.append(w.first_fixation_duration)
                    if w.gaze_duration > 0:
                        a_gazes.append(w.gaze_duration)

            # Group B averages
            b_trts, b_ffds, b_gazes, b_skips_list = [], [], [], []
            for pid in group_b:
                if pid not in pdata:
                    continue
                w = pdata[pid]
                if w.was_skipped:
                    b_skips_list.append(1.0)
                else:
                    b_skips_list.append(0.0)
                    if w.total_reading_time > 0:
                        b_trts.append(w.total_reading_time)
                    if w.first_fixation_duration > 0:
                        b_ffds.append(w.first_fixation_duration)
                    if w.gaze_duration > 0:
                        b_gazes.append(w.gaze_duration)

            # Only include words where both groups have data
            if a_trts and b_trts:
                a_trt.append(np.mean(a_trts))
                b_trt.append(np.mean(b_trts))
            if a_ffds and b_ffds:
                a_ffd.append(np.mean(a_ffds))
                b_ffd.append(np.mean(b_ffds))
            if a_gazes and b_gazes:
                a_gaze.append(np.mean(a_gazes))
                b_gaze.append(np.mean(b_gazes))
            if a_skips_list and b_skips_list:
                a_skip.append(np.mean(a_skips_list))
                b_skip.append(np.mean(b_skips_list))

        # Correlate the two halves
        def corr(x, y):
            x, y = np.array(x), np.array(y)
            if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0:
                return np.corrcoef(x, y)[0, 1]
            return 0.0

        pairs = [
            ('trt', a_trt, b_trt),
            ('ffd', a_ffd, b_ffd),
            ('gaze', a_gaze, b_gaze),
            ('skip', a_skip, b_skip),
        ]

        for metric, a_vals, b_vals in pairs:
            r_half = corr(a_vals, b_vals)
            # Spearman-Brown correction: estimate reliability of the full (14-person) average
            r_full = 2 * r_half / (1 + abs(r_half)) if abs(r_half) > 0 else 0.0
            half_corrs[metric].append(r_half)
            full_corrs[metric].append(r_full)

        if (split_idx + 1) % 50 == 0:
            print(f"  Split {split_idx + 1}/{n_splits} done")

    # Summary
    print("\n" + "=" * 80)
    print("NOISE CEILING RESULTS (GECO)")
    print("=" * 80)
    print(f"\nSplit-half reliability ({n_splits} random splits of {n_participants} participants)")
    print(f"  Words used: {len(usable_keys):,}")
    print()

    print(f"{'Metric':<8s} | {'r_half (mean)':>13s} {'r_half (std)':>13s} | "
          f"{'r_full (SB)':>12s} {'r_full (std)':>13s} | {'r_full^2 (var%)':>15s}")
    print("-" * 80)

    results = {}
    for metric in metrics:
        r_h = np.array(half_corrs[metric])
        r_f = np.array(full_corrs[metric])
        r_h_mean, r_h_std = np.mean(r_h), np.std(r_h)
        r_f_mean, r_f_std = np.mean(r_f), np.std(r_f)
        var_explained = r_f_mean ** 2 * 100

        print(f"{metric.upper():<8s} | {r_h_mean:13.4f} {r_h_std:13.4f} | "
              f"{r_f_mean:12.4f} {r_f_std:13.4f} | {var_explained:14.1f}%")

        results[metric] = {
            'r_half_mean': r_h_mean,
            'r_half_std': r_h_std,
            'r_full_mean': r_f_mean,
            'r_full_std': r_f_std,
        }

    print()
    print("Interpretation:")
    print("  r_half  = correlation between two independent 7-person averages")
    print("  r_full  = Spearman-Brown corrected: estimated reliability of the 14-person average")
    print("  r_full  = the MAXIMUM correlation ANY model can achieve when predicting these averages")
    print()
    print("Your current best model:")
    print("  r_TRT  = 0.467  (ceiling: {:.3f})  → {:.0f}% of ceiling".format(
        results['trt']['r_full_mean'],
        100 * 0.467 / max(results['trt']['r_full_mean'], 0.001)))
    print("  r_FFD  = 0.207  (ceiling: {:.3f})  → {:.0f}% of ceiling".format(
        results['ffd']['r_full_mean'],
        100 * 0.207 / max(results['ffd']['r_full_mean'], 0.001)))
    print("  r_Skip = 0.699  (ceiling: {:.3f})  → {:.0f}% of ceiling".format(
        results['skip']['r_full_mean'],
        100 * 0.699 / max(results['skip']['r_full_mean'], 0.001)))
    print("  r_Gaze = 0.388  (ceiling: {:.3f})  → {:.0f}% of ceiling".format(
        results['gaze']['r_full_mean'],
        100 * 0.388 / max(results['gaze']['r_full_mean'], 0.001)))

    # Also compute by train/val/test split to check if ceiling differs
    print("\n" + "=" * 80)
    print("NOISE CEILING BY SPLIT")
    print("=" * 80)

    train_raw, val_raw, test_raw = split_geco(raw_dataset)
    train_text_ids = set(sd.text_id for sd in train_raw)
    val_text_ids = set(sd.text_id for sd in val_raw)

    for split_name, split_text_ids in [("train", train_text_ids), ("val", val_text_ids)]:
        split_keys = [k for k in usable_keys if k[0] in split_text_ids]
        if len(split_keys) < 100:
            print(f"\n  {split_name}: too few words ({len(split_keys)}), skipping")
            continue

        # One representative split for each
        split_corrs = {m: [] for m in metrics}
        for _ in range(100):
            shuffled = participants.copy()
            rng.shuffle(shuffled)
            group_a = set(shuffled[:half_size])
            group_b = set(shuffled[half_size:])

            a_trt, b_trt = [], []
            a_skip, b_skip = [], []

            for key in split_keys:
                pdata = word_index[key]
                a_trts, b_trts = [], []
                a_skips_list, b_skips_list = [], []

                for pid in group_a:
                    if pid not in pdata:
                        continue
                    w = pdata[pid]
                    a_skips_list.append(1.0 if w.was_skipped else 0.0)
                    if not w.was_skipped and w.total_reading_time > 0:
                        a_trts.append(w.total_reading_time)

                for pid in group_b:
                    if pid not in pdata:
                        continue
                    w = pdata[pid]
                    b_skips_list.append(1.0 if w.was_skipped else 0.0)
                    if not w.was_skipped and w.total_reading_time > 0:
                        b_trts.append(w.total_reading_time)

                if a_trts and b_trts:
                    a_trt.append(np.mean(a_trts))
                    b_trt.append(np.mean(b_trts))
                if a_skips_list and b_skips_list:
                    a_skip.append(np.mean(a_skips_list))
                    b_skip.append(np.mean(b_skips_list))

            def corr(x, y):
                x, y = np.array(x), np.array(y)
                if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0:
                    return np.corrcoef(x, y)[0, 1]
                return 0.0

            r_h = corr(a_trt, b_trt)
            r_f = 2 * r_h / (1 + abs(r_h)) if abs(r_h) > 0 else 0.0
            split_corrs['trt'].append(r_f)

            r_h = corr(a_skip, b_skip)
            r_f = 2 * r_h / (1 + abs(r_h)) if abs(r_h) > 0 else 0.0
            split_corrs['skip'].append(r_f)

        print(f"\n  {split_name} ({len(split_keys):,} words):")
        print(f"    TRT ceiling:  r = {np.mean(split_corrs['trt']):.4f} +/- {np.std(split_corrs['trt']):.4f}")
        print(f"    Skip ceiling: r = {np.mean(split_corrs['skip']):.4f} +/- {np.std(split_corrs['skip']):.4f}")

    return results


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
    material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
    pred_path = os.path.join(data_dir, "geco_predictability.pkl")

    print("Loading GECO Corpus...")
    raw_dataset = load_geco(reading_path, material_path, pred_path)
    print(f"Loaded {len(raw_dataset):,} observations\n")

    results = compute_split_half_reliability(raw_dataset, n_splits=200, seed=42)
