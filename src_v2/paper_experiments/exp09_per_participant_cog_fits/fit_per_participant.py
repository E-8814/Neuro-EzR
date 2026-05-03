"""
Per-participant cognitive parameter fits (exp09).

For each GECO reader:
    1. Load paper model (frozen backbone)
    2. Fine-tune ONLY cog scalars on their reading data
    3. Save fitted scalars

Usage:
    python fit_per_participant.py --seed 42 --epochs 3 --lr 3e-5
"""

import argparse
import copy
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from paper_experiments import config
from paper_experiments.utils.load_data import (
    load_geco_per_participant, load_subtlex, word_frequency,
)
from paper_experiments.utils.load_model import (
    load_paper_model, freeze_neural_layers, get_cog_param_list,
    collect_cog_params,
)
from paper_experiments.utils.eval_metrics import corr


SIGMA2_TRT = 10000.0
SIGMA2_FFD = 1500.0
SIGMA2_GAZE = 4500.0


RESULTS_DIR = Path(_HERE) / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FITS_CSV = RESULTS_DIR / "per_participant_cog_fits.csv"


def collate(batch, device, subtlex):
    word_lists = [sd.tokens for sd in batch]
    freqs = pad_sequence(
        [torch.tensor([float(word_frequency(t, subtlex)) for t in sd.tokens],
                      dtype=torch.float32) for sd in batch],
        batch_first=True, padding_value=1.0,
    ).to(device)
    wlens = pad_sequence(
        [torch.tensor([len(t) for t in sd.tokens], dtype=torch.float32)
         for sd in batch],
        batch_first=True,
    ).to(device)
    h_trt = pad_sequence(
        [torch.tensor(sd.total_reading_times, dtype=torch.float32) for sd in batch],
        batch_first=True,
    ).to(device)
    h_ffd = pad_sequence(
        [torch.tensor(sd.first_fixation_durations, dtype=torch.float32) for sd in batch],
        batch_first=True,
    ).to(device)
    h_gaze = pad_sequence(
        [torch.tensor(sd.gaze_durations, dtype=torch.float32) for sd in batch],
        batch_first=True,
    ).to(device)
    h_skip = pad_sequence(
        [torch.tensor([1.0 if s else 0.0 for s in sd.skip_flags], dtype=torch.float32) for sd in batch],
        batch_first=True,
    ).to(device)
    return word_lists, freqs, wlens, h_trt, h_ffd, h_gaze, h_skip


def loss_fn(pred, h_trt, h_ffd, h_gaze, h_skip):
    pred_trt = pred['conditional_trt'].float()
    pred_ffd = pred['first_fixation'].float()
    pred_gaze = pred['gaze_duration'].float()
    pred_skip = pred['skip_prob'].float()
    fixated = (h_skip < 0.5)
    if fixated.sum() > 0:
        trt_mse = F.mse_loss(pred_trt[fixated], h_trt[fixated])
        ffd_mse = F.mse_loss(pred_ffd[fixated], h_ffd[fixated])
        gaze_mse = F.mse_loss(pred_gaze[fixated], h_gaze[fixated])
    else:
        zero = torch.tensor(0.0, device=pred_trt.device)
        trt_mse = ffd_mse = gaze_mse = zero
    skip_pred = pred_skip.clamp(1e-6, 1 - 1e-6)
    skip_loss = F.binary_cross_entropy(skip_pred, h_skip)
    return (
        trt_mse / SIGMA2_TRT + ffd_mse / SIGMA2_FFD
        + gaze_mse / SIGMA2_GAZE + skip_loss
    )


def fit_one_participant(model, participant_data, device, subtlex,
                        epochs, lr, batch_size):
    """Frozen-backbone fine-tune of cog scalars on one reader's data."""
    freeze_neural_layers(model)
    cog_params = get_cog_param_list(model)
    optimizer = torch.optim.AdamW(cog_params, lr=lr)

    losses = []
    n_batches = (len(participant_data) + batch_size - 1) // batch_size

    for epoch in range(epochs):
        # Shuffle reader's data
        import random as _r
        _r.shuffle(participant_data)

        epoch_loss = 0.0
        n_seen = 0
        for step in range(n_batches):
            batch = participant_data[step * batch_size:(step + 1) * batch_size]
            if not batch:
                continue
            word_lists, freqs, wlens, h_trt, h_ffd, h_gaze, h_skip = \
                collate(batch, device, subtlex)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(word_lists, freqs, wlens)
            loss = loss_fn(pred, h_trt, h_ffd, h_gaze, h_skip)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cog_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(batch)
            n_seen += len(batch)
        losses.append(epoch_loss / max(1, n_seen))

    return collect_cog_params(model), losses[-1] if losses else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=config.PER_PARTICIPANT_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.PER_PARTICIPANT_COG_LR)
    parser.add_argument("--batch_size", type=int, default=config.PER_PARTICIPANT_BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading per-participant data (train+val splits)...")
    by_p_train = load_geco_per_participant(split="train")
    by_p_val = load_geco_per_participant(split="val")
    # Combine train + val per participant for fitting
    by_p_fit = {}
    for pid, sds in by_p_train.items():
        by_p_fit[pid] = list(sds)
    for pid, sds in by_p_val.items():
        by_p_fit.setdefault(pid, []).extend(sds)

    subtlex = load_subtlex()

    print(f"Loading paper model (seed={args.seed})...")
    base_model, _ = load_paper_model(seed=args.seed, device=device)
    base_state = copy.deepcopy(base_model.state_dict())

    rows = []
    for pid in sorted(by_p_fit.keys()):
        data = by_p_fit[pid]
        n_words = sum(len(sd.tokens) for sd in data)
        if n_words < 100:
            print(f"  {pid}: only {n_words} words, skipping.")
            continue
        print(f"  {pid}: fitting on {len(data)} sentences ({n_words} words)...",
              flush=True)
        # Reset to base model state before each participant fit
        base_model.load_state_dict(base_state)

        t0 = time.time()
        cog, final_loss = fit_one_participant(
            base_model, list(data), device, subtlex,
            args.epochs, args.lr, args.batch_size,
        )
        elapsed = time.time() - t0
        print(f"    α1R={cog['alpha1_reichle']:.2f}  α2R={cog['alpha2_reichle']:.3f}  "
              f"ε={cog['epsilon']:.3f}  M1={cog['M1']:.1f}  "
              f"M2=I={cog['M2_eq_I']:.1f}  δ={cog['delta']:.3f}  "
              f"({elapsed:.1f}s, loss={final_loss:.4f})")

        # Compute mean RT for this participant for downstream correlation analysis
        mean_RT = float(np.mean([rt for sd in data for rt in sd.total_reading_times]))

        row = {
            "participant_id": pid,
            "n_train_sentences": len(data),
            "n_train_words": n_words,
            "fit_loss": final_loss,
            "fit_time_seconds": elapsed,
            "mean_RT": mean_RT,
            **cog,
        }
        rows.append(row)

    if not rows:
        print("No participants fit.")
        return

    fieldnames = list(rows[0].keys())
    with open(FITS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {len(rows)} rows to {FITS_CSV}")


if __name__ == "__main__":
    main()
