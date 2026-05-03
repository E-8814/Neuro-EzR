"""
Compute model-vs-ceiling curve on Provo — corrected methodology (v2).

Fix vs v1: same bug as exp04. v1 took np.mean over ALL participants
in each half for RT/FFD/Gaze, including participants who skipped
(where the metric is 0). v2 filters skipped words for RT-based metrics
and only includes a word for a metric when both halves have at least
one valid measurement, mirroring src_v2/break_the_ceiling/noise_ceiling.py.

For multiple data fractions f ∈ [0.2, 0.4, 0.6, 0.8, 1.0]:
  - subsample participants to f × n_participants
  - compute corrected split-half reliability of the subsample (= ceiling at f)
  - evaluate paper model on the same subsample (= model at f)
  - record gap

Usage:
    python compute_ceiling_curve_v2.py
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
CURVE_CSV = RESULTS_DIR / "ceiling_curve_results_v2.csv"


def _spearman_brown(r_half):
    if r_half <= -1.0 or r_half >= 1.0:
        return r_half
    return 2.0 * r_half / (1.0 + r_half)


def split_half_at_subsample(raw_provo, participant_subset, n_splits=50, seed=42):
    """Corrected split-half reliability for the subset of participants."""
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

        a_arr = {m: [] for m in metrics}
        b_arr = {m: [] for m in metrics}

        for key in usable:
            pdata = word_index[key]

            a_trts, a_ffds, a_gazes, a_skips = [], [], [], []
            for pid in ga:
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
            for pid in gb:
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
                a_arr["trt"].append(float(np.mean(a_trts)))
                b_arr["trt"].append(float(np.mean(b_trts)))
            if a_ffds and b_ffds:
                a_arr["ffd"].append(float(np.mean(a_ffds)))
                b_arr["ffd"].append(float(np.mean(b_ffds)))
            if a_gazes and b_gazes:
                a_arr["gaze"].append(float(np.mean(a_gazes)))
                b_arr["gaze"].append(float(np.mean(b_gazes)))
            if a_skips and b_skips:
                a_arr["skip"].append(float(np.mean(a_skips)))
                b_arr["skip"].append(float(np.mean(b_skips)))

        for m in metrics:
            a = np.array(a_arr[m])
            b = np.array(b_arr[m])
            if len(a) > 2 and a.std() > 0 and b.std() > 0:
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

        kept_raw = [sd for sd in raw_provo if sd.participant_id in keep]
        ds = aggregate_by_sentence(kept_raw, min_participants=2)
        if not ds:
            print("    (skipping; no aggregated sentences at this fraction)")
            continue

        _, summary = eval_predictions_on_aggregated(model, ds, device, subtlex)

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
