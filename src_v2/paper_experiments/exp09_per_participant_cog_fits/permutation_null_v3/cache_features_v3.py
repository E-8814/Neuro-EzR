"""
Cache frozen-neural features per participant for the v3 model, mirroring
../permutation_null/01_cache_features.py but loading the v4c_v3_dualctx
(skip_align=next) checkpoint.

Writes one .pt per participant into results/cache/.

Usage:
    python -u cache_features_v3.py            # seed 42 (paper default)
    python -u cache_features_v3.py --force
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP09 = os.path.abspath(os.path.join(_HERE, ".."))
SRC_V2 = os.path.abspath(os.path.join(EXP09, "..", ".."))

for p in (SRC_V2, EXP09, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from paper_experiments import config  # noqa: E402
from paper_experiments.utils.load_data import (  # noqa: E402
    load_geco_per_participant, load_subtlex,
)
from fit_per_participant import collate  # noqa: E402  (same collate as v2)
from _cog_fit_v3 import load_v3_model  # noqa: E402


CACHE_DIR = Path(_HERE) / "results" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def cache_one_participant(model, pid, sentences, device, subtlex, batch_size):
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
    parser.add_argument("--force", action="store_true")
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

    print(f"Loading v4c_v3_dualctx_next model (seed={args.seed})...")
    model = load_v3_model(seed=args.seed, device=device)

    pids = sorted(by_p_fit.keys())
    print(f"Participants to cache ({len(pids)}): {pids}")

    for pid in pids:
        out_path = CACHE_DIR / f"{pid}.pt"
        if out_path.exists() and not args.force:
            print(f"  {pid}: cache exists, skipping.")
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
            "model_recipe": "v4c_v3_dualctx_next",
            "batches": cached,
        }, str(out_path))
        print(f"  {pid}: cached {len(cached)} batches ({time.time()-t0:.1f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
