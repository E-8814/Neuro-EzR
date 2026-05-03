"""
Per-participant evaluation (exp08).

Loads paper model (seed=42 by default). For each GECO participant,
evaluate on all sentences (test split) and report metrics.

Each participant's "ground truth" comes from THEIR fixation data (not
aggregated). We aggregate predictions per word per sentence (model
predictions are deterministic), and compare against this participant's
own per-word measures.

Usage:
    python eval_per_participant.py --seed 42
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from paper_experiments import config
from paper_experiments.utils.load_data import (
    load_geco_per_participant, load_subtlex, word_frequency,
)
from paper_experiments.utils.load_model import load_paper_model
from paper_experiments.utils.eval_metrics import corr, mae, bias


RESULTS_DIR = Path(_HERE) / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RESULTS_DIR / "per_participant_eval.csv"


def evaluate_participant(model, participant_data, device, subtlex):
    """Evaluate model against one participant's reading data."""
    pt, ht = [], []
    pf, hf = [], []
    pg, hg = [], []
    ps, hs = [], []

    with torch.no_grad():
        for sd in participant_data:
            tokens = sd.tokens
            n = len(tokens)
            freqs = torch.tensor(
                [float(word_frequency(t, subtlex)) for t in tokens],
                dtype=torch.float32,
            ).unsqueeze(0).to(device)
            wlens = torch.tensor(
                [len(t) for t in tokens], dtype=torch.float32,
            ).unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                p = model([tokens], freqs, wlens)

            for i in range(n):
                pt.append(p['conditional_trt'][0, i].item())
                pf.append(p['first_fixation'][0, i].item())
                pg.append(p['gaze_duration'][0, i].item())
                ps.append(p['skip_prob'][0, i].item())
                ht.append(sd.total_reading_times[i])
                hf.append(sd.first_fixation_durations[i])
                hg.append(sd.gaze_durations[i])
                hs.append(1.0 if sd.skip_flags[i] else 0.0)

    return {
        "n_sentences": len(participant_data),
        "n_words": len(pt),
        "r_TRT": corr(pt, ht), "r_FFD": corr(pf, hf),
        "r_Gaze": corr(pg, hg), "r_skip": corr(ps, hs),
        "MAE_TRT": mae(pt, ht), "MAE_FFD": mae(pf, hf), "MAE_Gaze": mae(pg, hg),
        "bias_TRT": bias(pt, ht), "bias_FFD": bias(pf, hf),
        "mean_RT": float(np.mean(ht)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading paper model (seed={args.seed})...")
    model, _ = load_paper_model(seed=args.seed, device=device)

    print(f"Loading per-participant GECO ({args.split} split)...")
    by_p = load_geco_per_participant(split=args.split)
    subtlex = load_subtlex()

    rows = []
    for pid in sorted(by_p.keys()):
        data = by_p[pid]
        print(f"  {pid}: {len(data)} sentences ...", flush=True)
        metrics = evaluate_participant(model, data, device, subtlex)
        rows.append({"participant_id": pid, **metrics})
        print(f"    r_TRT={metrics['r_TRT']:.3f}  r_FFD={metrics['r_FFD']:.3f}  "
              f"r_skip={metrics['r_skip']:.3f}")

    # Add summary row at the end (mean ± std across readers)
    arrs = {
        col: np.array([r[col] for r in rows], dtype=float)
        for col in rows[0] if col != "participant_id"
    }
    summary_row = {"participant_id": "mean"}
    summary_row.update({
        col: float(np.mean(vals)) for col, vals in arrs.items()
    })
    rows.append(summary_row)
    std_row = {"participant_id": "std"}
    std_row.update({
        col: float(np.std(vals, ddof=1)) for col, vals in arrs.items()
    })
    rows.append(std_row)

    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nWrote {CSV_PATH}")
    print(f"Per-reader r_TRT: mean = {summary_row['r_TRT']:.3f} ± {std_row['r_TRT']:.3f}")


if __name__ == "__main__":
    main()
