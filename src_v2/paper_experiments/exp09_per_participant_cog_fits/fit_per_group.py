"""
Per-group cognitive parameter fits (exp09 follow-up).

Splits the 14 GECO readers into 2 groups by median mean_RT (7 fast + 7 slow)
and fine-tunes cog scalars per group on the COMBINED data of all readers
in that group. With ~38k words per group instead of ~5k per reader, the
fits have enough signal to support meaningful parameter estimation.

This addresses the core finding of fit_per_participant.py: with only
9 trainable scalars and ~5k words per reader, cog params barely move
across readers. The 2-group design gives 5x more data per fit and
asks a tighter question: do reading-style groups have systematically
different cog parameters?

Reader split (from per_participant_eval.csv mean_RT):
    fast: pp23 pp34 pp25 pp29 pp32 pp22 pp33  (mean_RT 107-142)
    slow: pp31 pp26 pp27 pp28 pp35 pp21 pp30  (mean_RT 185-220)

Usage (from byzantium srun, neuro_ezr env):
    python -u src_v2/paper_experiments/exp09_per_participant_cog_fits/fit_per_group.py
    python -u .../fit_per_group.py --epochs 5 --lr 1e-4
"""

import argparse
import copy
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from paper_experiments import config
from paper_experiments.utils.load_data import (
    load_geco_per_participant, load_subtlex,
)
from paper_experiments.utils.load_model import (
    load_paper_model, freeze_neural_layers, get_cog_param_list,
    collect_cog_params,
)

# Reuse the per-participant collate + loss + fit loop.
sys.path.insert(0, _HERE)
from fit_per_participant import collate, loss_fn, fit_one_participant


RESULTS_DIR = Path(_HERE) / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GROUP_CSV = RESULTS_DIR / "per_group_cog_fits.csv"
COMPARISON_CSV = RESULTS_DIR / "per_group_comparison.csv"


# Pre-computed split (median mean_RT ~ 163ms).
FAST_READERS = {"pp23", "pp34", "pp25", "pp29", "pp32", "pp22", "pp33"}
SLOW_READERS = {"pp31", "pp26", "pp27", "pp28", "pp35", "pp21", "pp30"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=config.PER_PARTICIPANT_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.PER_PARTICIPANT_COG_LR)
    parser.add_argument("--batch_size", type=int,
                        default=config.PER_PARTICIPANT_BATCH_SIZE)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Settings: epochs={args.epochs}  lr={args.lr}  batch={args.batch_size}")

    # ---- Load per-reader data, combine train+val ---- #
    print("Loading per-participant data (train+val splits)...")
    by_p_train = load_geco_per_participant(split="train")
    by_p_val   = load_geco_per_participant(split="val")
    by_p_fit = {}
    for pid, sds in by_p_train.items():
        by_p_fit[pid] = list(sds)
    for pid, sds in by_p_val.items():
        by_p_fit.setdefault(pid, []).extend(sds)

    available = set(by_p_fit.keys())
    fast = sorted(FAST_READERS & available)
    slow = sorted(SLOW_READERS & available)
    print(f"\nGroups:")
    print(f"  fast (n={len(fast)}): {fast}")
    print(f"  slow (n={len(slow)}): {slow}")

    subtlex = load_subtlex()

    # ---- Load model once, snapshot for resets ---- #
    print(f"\nLoading paper model (seed={args.seed})...")
    model, _ = load_paper_model(seed=args.seed, device=device)
    base_state = copy.deepcopy(model.state_dict())

    rows = []
    for label, readers in (("fast", fast), ("slow", slow)):
        # Pool all sentences in this group into one list.
        group_data = []
        for pid in readers:
            group_data.extend(by_p_fit[pid])
        random.shuffle(group_data)

        n_words = sum(len(sd.tokens) for sd in group_data)
        n_sents = len(group_data)
        mean_RT = float(np.mean(
            [rt for sd in group_data for rt in sd.total_reading_times]
        ))
        print(f"\n>> Fitting {label} group: {len(readers)} readers, "
              f"{n_sents} sentences, {n_words} words, mean_RT={mean_RT:.1f}ms")

        # Reset to base model.
        model.load_state_dict(base_state)

        t0 = time.time()
        cog, final_loss = fit_one_participant(
            model, list(group_data), device, subtlex,
            args.epochs, args.lr, args.batch_size,
        )
        elapsed = time.time() - t0

        print(f"   final_loss={final_loss:.4f}  ({elapsed:.1f}s)")
        for k in ("alpha1_reichle", "alpha2_reichle", "epsilon",
                  "M1", "M2_eq_I", "delta", "lambda_refix",
                  "refix_pivot", "skip_temperature"):
            if k in cog:
                print(f"   {k:<22s} {cog[k]:.4f}")

        rows.append({
            "group": label,
            "n_readers": len(readers),
            "n_sentences": n_sents,
            "n_words": n_words,
            "mean_RT": mean_RT,
            "fit_loss": final_loss,
            "fit_time_seconds": elapsed,
            **cog,
        })

    # ---- Write per-group fits ---- #
    fieldnames = list(rows[0].keys())
    with open(GROUP_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} group fits to {GROUP_CSV}")

    # ---- Comparison: fast vs slow per parameter ---- #
    fast_row = next(r for r in rows if r["group"] == "fast")
    slow_row = next(r for r in rows if r["group"] == "slow")

    cog_keys = [k for k in fast_row
                if k in slow_row and isinstance(fast_row[k], (int, float))
                and k not in {"n_readers", "n_sentences", "n_words",
                              "fit_loss", "fit_time_seconds", "mean_RT"}]

    comparison = []
    for k in cog_keys:
        f_val, s_val = fast_row[k], slow_row[k]
        denom = abs(f_val) if abs(f_val) > 1e-9 else 1.0
        rel_pct = 100.0 * (s_val - f_val) / denom
        comparison.append({
            "param": k,
            "fast": f_val, "slow": s_val,
            "abs_diff": s_val - f_val,
            "rel_pct_change": rel_pct,
        })

    with open(COMPARISON_CSV, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["param", "fast", "slow", "abs_diff", "rel_pct_change"],
        )
        w.writeheader()
        for r in comparison:
            w.writerow(r)
    print(f"Wrote per-param fast vs slow comparison to {COMPARISON_CSV}")

    print(f"\n{'='*72}")
    print(f"{'param':<22s} {'fast':>10s} {'slow':>10s} {'abs Δ':>10s} {'rel Δ%':>8s}")
    print('-' * 72)
    for r in comparison:
        sign = '+' if r["abs_diff"] >= 0 else ''
        print(f"{r['param']:<22s} {r['fast']:>10.4f} {r['slow']:>10.4f} "
              f"{sign}{r['abs_diff']:>+10.4f} {r['rel_pct_change']:>+7.1f}%")


if __name__ == "__main__":
    main()
