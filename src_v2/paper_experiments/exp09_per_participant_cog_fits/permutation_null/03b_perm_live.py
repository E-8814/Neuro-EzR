"""
Fallback permutation null using the LIVE model (no caching), for use
when 02_sanity_check.py fails.

Same JSON-per-split scheme as 03a_perm_cached.py — random-order
selection, atomic writes, resumable across submissions. Default budget
is 300 random splits (out of 1,716); reviewer-defensibility for
permutation p ~ 0.005 is bounded by SE ~ sqrt(p(1-p)/300) ~ 0.004.

Each split is much more expensive than in the cached path because we
re-run the full TinyLlama forward per batch (no caching). Expect ~2-5
minutes per group on a single RTX-class GPU; ~5-10 minutes per split.

Usage:
    python -u .../03b_perm_live.py
    python -u .../03b_perm_live.py --max_perms 100 --max_runtime_minutes 240
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from paper_experiments import config
from paper_experiments.utils.load_data import (
    load_geco_per_participant, load_subtlex,
)
from paper_experiments.utils.load_model import load_paper_model

from fit_per_participant import fit_one_participant
from _cog_fit import (
    enumerate_balanced_splits, dissociation_T,
)


PERMS_DIR = Path(_HERE) / "results" / "perms_live"
PERMS_DIR.mkdir(parents=True, exist_ok=True)


def _perm_path(idx: int) -> Path:
    return PERMS_DIR / f"perm_{idx:04d}.json"


def _completed_indices() -> set:
    return {
        int(p.stem.split("_")[1])
        for p in PERMS_DIR.glob("perm_*.json")
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=config.PER_PARTICIPANT_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.PER_PARTICIPANT_COG_LR)
    parser.add_argument("--batch_size", type=int,
                        default=config.PER_PARTICIPANT_BATCH_SIZE)
    parser.add_argument("--num_perms", type=int, default=300,
                        help="Total target number of random permutations to run "
                             "(across all submissions, NOT per-job).")
    parser.add_argument("--max_perms", type=int, default=10**9,
                        help="Stop after this many perms in THIS run.")
    parser.add_argument("--max_runtime_minutes", type=float, default=10**9)
    parser.add_argument("--pick_seed", type=int, default=None,
                        help="Seed for which random subset of 1,716 we draw.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load data + subtlex ---- #
    print("Loading per-participant data (train+val splits)...")
    by_p_train = load_geco_per_participant(split="train")
    by_p_val   = load_geco_per_participant(split="val")
    by_p_fit = {}
    for pid, sds in by_p_train.items():
        by_p_fit[pid] = list(sds)
    for pid, sds in by_p_val.items():
        by_p_fit.setdefault(pid, []).extend(sds)
    subtlex = load_subtlex()

    pids = sorted(by_p_fit.keys())
    splits = enumerate_balanced_splits(pids)
    n_splits = len(splits)

    # Choose a deterministic random subset of `--num_perms` indices to
    # target across all submissions. The pick_seed determines which
    # subset; default uses a fixed seed (=0) so re-submissions agree on
    # the target subset and converge.
    target_rng = random.Random(args.pick_seed if args.pick_seed is not None else 0)
    target_indices = sorted(target_rng.sample(range(n_splits), args.num_perms))
    print(f"Targeting {len(target_indices)} of {n_splits} balanced splits")
    print(f"  (pick_seed={args.pick_seed if args.pick_seed is not None else 0})")

    completed = _completed_indices()
    missing = [i for i in target_indices if i not in completed]
    print(f"Already completed: {len(target_indices) - len(missing)} / {len(target_indices)}")
    if not missing:
        print("All target perms done. Nothing to do.")
        return

    # Pick order WITHIN this run is random.
    pick_rng = random.Random()  # fresh each run

    # ---- Load model once ---- #
    print(f"Loading paper model (seed={args.seed})...")
    model, _ = load_paper_model(seed=args.seed, device=device)
    base_state = copy.deepcopy(model.state_dict())

    # ---- Loop ---- #
    t_start = time.time()
    n_done = 0
    while True:
        completed = _completed_indices()
        missing = [i for i in target_indices if i not in completed]
        if not missing:
            print("\nAll target perms complete.")
            break
        if n_done >= args.max_perms:
            print(f"\nHit --max_perms={args.max_perms}.")
            break
        if (time.time() - t_start) / 60.0 >= args.max_runtime_minutes:
            print(f"\nHit --max_runtime_minutes={args.max_runtime_minutes}.")
            break

        idx = pick_rng.choice(missing)
        out_path = _perm_path(idx)
        if out_path.exists():
            continue

        group_a, group_b = splits[idx]
        t0 = time.time()
        try:
            # Group A
            model.load_state_dict(base_state)
            ga_data = []
            for pid in group_a:
                ga_data.extend(by_p_fit[pid])
            local_rng = random.Random(args.seed * 7919 + idx)
            local_rng.shuffle(ga_data)
            cog_a, _ = fit_one_participant(
                model, list(ga_data), device, subtlex,
                args.epochs, args.lr, args.batch_size,
            )
            # Group B
            model.load_state_dict(base_state)
            gb_data = []
            for pid in group_b:
                gb_data.extend(by_p_fit[pid])
            local_rng = random.Random(args.seed * 7919 + idx + 1)
            local_rng.shuffle(gb_data)
            cog_b, _ = fit_one_participant(
                model, list(gb_data), device, subtlex,
                args.epochs, args.lr, args.batch_size,
            )
            T_components = dissociation_T(cog_a, cog_b)
        except Exception as exc:
            elapsed = time.time() - t0
            err_payload = {
                "index": idx,
                "group_a": list(group_a),
                "group_b": list(group_b),
                "error": repr(exc),
                "elapsed_seconds": elapsed,
                "seed": args.seed,
            }
            err_path = PERMS_DIR / f"perm_{idx:04d}.error.json"
            _atomic_write_json(err_path, err_payload)
            print(f"  [error] split {idx}: {exc!r} -> {err_path.name}")
            n_done += 1
            continue

        elapsed = time.time() - t0
        payload = {
            "index": idx,
            "group_a": list(group_a),
            "group_b": list(group_b),
            "cog_a": cog_a,
            "cog_b": cog_b,
            "T": T_components["T"],
            "T_components": T_components,
            "elapsed_seconds": elapsed,
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "live": True,
        }
        _atomic_write_json(out_path, payload)
        n_done += 1
        print(f"  [{n_done:>4d}] split {idx:>4d}  T={T_components['T']:+8.3f}  "
              f"({elapsed:.1f}s)")

    print(f"\nThis run: completed {n_done} perms.")
    cumulative = sum(1 for i in target_indices if i in _completed_indices())
    print(f"Cumulative: {cumulative} / {len(target_indices)} target.")


if __name__ == "__main__":
    main()
