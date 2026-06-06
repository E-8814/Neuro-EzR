"""
Cache the frozen-neural outputs per participant so that subsequent
per-group cog-scalar fits become pure cog-scalar SGD on tiny tensors.

What we cache, per word, per participant:
    log_freq_norm        scalar (recomputed input to L1 formula)
    word_lengths         scalar (input to cascade)
    ctx_FFD              scalar (output of frozen ctx_head_FFD)
    ctx_skip             scalar (output of frozen ctx_head_skip)
    residual_skip_logit  scalar (output of frozen skip_residual_head)

Plus the four targets:
    h_trt, h_ffd, h_gaze, h_skip

These are the only neural-net-side quantities the cog cascade reads. Caching
them makes 1,716 permuted fits cost ~ 1 GPU-hour instead of ~ 100s of hours.

The cache is a torch.save() file per participant, holding a dict of cpu
fp32 tensors aligned by sentence and padded by sentence-batch.

Usage (inside Slurm or interactively, neuro_ezr env, GPU available):
    python -u .../permutation_null/01_cache_features.py
    python -u .../permutation_null/01_cache_features.py --seed 42 --batch_size 8

Idempotent: skips participants already cached.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_HERE, ".."))  # so we can import fit_per_participant

from paper_experiments import config
from paper_experiments.utils.load_data import (
    load_geco_per_participant, load_subtlex, word_frequency,
)
from paper_experiments.utils.load_model import load_paper_model

from fit_per_participant import collate  # reuse exactly the same collate


CACHE_DIR = Path(_HERE) / "results" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _participant_cache_path(pid: str) -> Path:
    return CACHE_DIR / f"{pid}.pt"


@torch.no_grad()
def cache_one_participant(model, pid, sentences, device, subtlex, batch_size):
    """Run the model forward in eval mode (autocast on CUDA) and capture
    everything the cog cascade needs as fp32 cpu tensors.

    Returns a list of dicts, one per (padded) batch. Each dict holds:
        log_freq_norm, word_lengths, ctx_FFD, ctx_skip, residual_skip_logit,
        h_trt, h_ffd, h_gaze, h_skip
    all of shape [B, T_max] where T_max is per-batch.
    """
    model.eval()
    cached = []
    n_batches = (len(sentences) + batch_size - 1) // batch_size
    for step in range(n_batches):
        batch = sentences[step * batch_size:(step + 1) * batch_size]
        if not batch:
            continue
        word_lists, freqs, wlens, h_trt, h_ffd, h_gaze, h_skip = collate(
            batch, device, subtlex,
        )
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred = model(word_lists, freqs, wlens)
        # Recompute log_freq_norm exactly as the model does (line 328-329 of model)
        log_freq = torch.log(freqs.clamp(min=1.0))
        log_freq_norm = (log_freq - 10.0) / 5.0

        cached.append({
            "log_freq_norm":      log_freq_norm.detach().to(torch.float32).cpu(),
            "word_lengths":       wlens.detach().to(torch.float32).cpu(),
            "ctx_FFD":            pred["ctx_FFD"].detach().to(torch.float32).cpu(),
            "ctx_skip":           pred["ctx_skip"].detach().to(torch.float32).cpu(),
            "residual_skip_logit": pred["residual_skip_logit"].detach().to(torch.float32).cpu(),
            "h_trt":              h_trt.detach().to(torch.float32).cpu(),
            "h_ffd":              h_ffd.detach().to(torch.float32).cpu(),
            "h_gaze":             h_gaze.detach().to(torch.float32).cpu(),
            "h_skip":             h_skip.detach().to(torch.float32).cpu(),
        })
    return cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--batch_size", type=int,
                        default=config.PER_PARTICIPANT_BATCH_SIZE)
    parser.add_argument("--force", action="store_true",
                        help="Recache even if files exist.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading per-participant data (train+val splits)...")
    by_p_train = load_geco_per_participant(split="train")
    by_p_val = load_geco_per_participant(split="val")
    by_p_fit = {}
    for pid, sds in by_p_train.items():
        by_p_fit[pid] = list(sds)
    for pid, sds in by_p_val.items():
        by_p_fit.setdefault(pid, []).extend(sds)

    subtlex = load_subtlex()

    print(f"Loading paper model (seed={args.seed})...")
    model, _ = load_paper_model(seed=args.seed, device=device)
    model.eval()

    pids = sorted(by_p_fit.keys())
    print(f"Participants to cache ({len(pids)}): {pids}")

    for pid in pids:
        out_path = _participant_cache_path(pid)
        if out_path.exists() and not args.force:
            print(f"  {pid}: cache exists -> {out_path}, skipping.")
            continue

        t0 = time.time()
        cached = cache_one_participant(
            model, pid, list(by_p_fit[pid]), device, subtlex, args.batch_size,
        )
        torch.save({
            "participant_id": pid,
            "n_sentences": len(by_p_fit[pid]),
            "n_batches": len(cached),
            "batch_size": args.batch_size,
            "seed": args.seed,
            "model_recipe": config.PAPER_MODEL_RECIPE,
            "batches": cached,
        }, str(out_path))
        elapsed = time.time() - t0
        print(f"  {pid}: cached {len(cached)} batches "
              f"({len(by_p_fit[pid])} sentences) -> {out_path}  ({elapsed:.1f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
