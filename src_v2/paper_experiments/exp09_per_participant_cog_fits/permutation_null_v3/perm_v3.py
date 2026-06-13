"""
Run the EXACT permutation null for the v3 model: all 1,716 balanced 7/7
splits of the 14 GECO readers, refitting the cognitive scalars per group
on cached frozen-neural features.

The observed fast/slow split is one of the 1,716, so the exact test
needs no separate observed run — aggregate_v3.py locates it by index.

Resumable and parallel-safe (one JSON per split, atomic writes, random
pick among missing indices), mirroring ../permutation_null/03a_perm_cached.py.

Usage:
    python -u perm_v3.py                          # run until done
    python -u perm_v3.py --max_runtime_minutes 60
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
EXP09 = os.path.abspath(os.path.join(_HERE, ".."))
PERM_V2 = os.path.join(EXP09, "permutation_null")
SRC_V2 = os.path.abspath(os.path.join(EXP09, "..", ".."))

for p in (SRC_V2, EXP09, PERM_V2, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from paper_experiments import config  # noqa: E402
from _cog_fit import (  # noqa: E402  (v2 helpers, model-agnostic)
    enumerate_balanced_splits, dissociation_T,
    load_cached_participants, pool_group_batches,
)
from _cog_fit_v3 import load_v3_model, fit_group_cached_v3  # noqa: E402


CACHE_DIR = Path(_HERE) / "results" / "cache"
PERMS_DIR = Path(_HERE) / "results" / "perms"
PERMS_DIR.mkdir(parents=True, exist_ok=True)


def _perm_path(idx: int) -> Path:
    return PERMS_DIR / f"perm_{idx:04d}.json"


def _completed_indices():
    done = set()
    for p in PERMS_DIR.glob("perm_*.json"):
        if p.name.endswith(".error.json"):
            continue
        try:
            done.add(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return done


def _atomic_write_json(path: Path, payload: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--epochs", type=int,
                        default=config.PER_PARTICIPANT_EPOCHS)
    parser.add_argument("--lr", type=float,
                        default=config.PER_PARTICIPANT_COG_LR)
    parser.add_argument("--max_perms", type=int, default=10_000)
    parser.add_argument("--max_runtime_minutes", type=float, default=2_000.0)
    parser.add_argument("--pick_seed", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    pick_rng = random.Random(args.pick_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not CACHE_DIR.exists() or not list(CACHE_DIR.glob("*.pt")):
        print("ERROR: no v3 cache. Run cache_features_v3.py first.")
        sys.exit(2)
    cache = load_cached_participants(CACHE_DIR)
    pids = sorted(cache.keys())
    if len(pids) != 14:
        print(f"WARNING: cache has {len(pids)} participants, expected 14: {pids}")

    splits = enumerate_balanced_splits(pids)
    n_splits = len(splits)
    print(f"Total balanced splits: {n_splits}")

    completed = _completed_indices()
    print(f"Already completed: {len(completed)} / {n_splits}")

    print(f"Loading v4c_v3_dualctx_next model (seed={args.seed})...")
    model = load_v3_model(seed=args.seed, device=device)
    base_state = copy.deepcopy(model.state_dict())

    t_start = time.time()
    n_done = 0
    while True:
        completed = _completed_indices()
        missing = [i for i in range(n_splits) if i not in completed]
        if not missing:
            print("\nAll splits complete.")
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
            model.load_state_dict(base_state)
            ba = pool_group_batches(cache, list(group_a))
            cog_a, _ = fit_group_cached_v3(
                model, ba, device,
                epochs=args.epochs, lr=args.lr,
                rng_seed=args.seed * 7919 + idx,
            )
            model.load_state_dict(base_state)
            bb = pool_group_batches(cache, list(group_b))
            cog_b, _ = fit_group_cached_v3(
                model, bb, device,
                epochs=args.epochs, lr=args.lr,
                rng_seed=args.seed * 7919 + idx + 1,
            )
            T_components = dissociation_T(cog_a, cog_b)
        except Exception as exc:
            err_path = PERMS_DIR / f"perm_{idx:04d}.error.json"
            _atomic_write_json(err_path, {
                "index": idx, "group_a": list(group_a), "group_b": list(group_b),
                "error": repr(exc), "elapsed_seconds": time.time() - t0,
                "seed": args.seed,
            })
            print(f"  [error] split {idx}: {exc!r}")
            n_done += 1
            continue

        elapsed = time.time() - t0
        _atomic_write_json(out_path, {
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
            "model_recipe": "v4c_v3_dualctx_next",
        })
        n_done += 1
        if n_done % 25 == 0 or n_done <= 5:
            total_done = len(_completed_indices())
            rate = n_done / max(1e-9, (time.time() - t_start) / 3600.0)
            print(f"  [{n_done:>4d}] split {idx:>4d}  T={T_components['T']:+8.3f} "
                  f"({elapsed:.1f}s) | cumulative {total_done}/{n_splits} "
                  f"| {rate:.0f}/h")

    print(f"\nThis run: completed {n_done} perms.")
    print(f"Cumulative: {len(_completed_indices())} / {n_splits}.")


if __name__ == "__main__":
    main()
