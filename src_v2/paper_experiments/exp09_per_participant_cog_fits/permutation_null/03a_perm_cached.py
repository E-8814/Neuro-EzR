"""
Cached-feature permutation null: enumerate ALL 1,716 balanced 7/7 splits
in random order, with one JSON per split for resumability.

Loop:
    1. Find which split indices already have a JSON in results/perms/.
    2. Of the remaining indices, pick one uniformly at random.
    3. Run the cached cog-scalar fit for both groups of that split.
    4. Write the JSON atomically (write to .tmp, then rename).
    5. Repeat until all 1,716 are done OR --max_runtime_minutes elapsed
       OR --max_perms reached.

Re-running the script picks up where the last invocation stopped.
Multiple instances on different GPUs are also safe (each instance picks
a random missing index; collisions are rare and harmless — at worst one
JSON is recomputed).

Usage (single-GPU slurm):
    python -u .../03a_perm_cached.py
    python -u .../03a_perm_cached.py --max_runtime_minutes 240
    python -u .../03a_perm_cached.py --max_perms 200

Per-perm output (results/perms/perm_<idx:04d>.json):
    {
      "index": int,                 canonical split index (0..1715)
      "group_a": [pp...],
      "group_b": [pp...],
      "cog_a": {param: value, ...},
      "cog_b": {param: value, ...},
      "T": float,
      "T_components": { ... per-param |%Δ| ... },
      "elapsed_seconds": float,
      "seed": int
    }
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
sys.path.insert(0, _HERE)

from paper_experiments import config
from paper_experiments.utils.load_model import load_paper_model

from _cog_fit import (
    enumerate_balanced_splits, fit_group_cached, dissociation_T,
    load_cached_participants, pool_group_batches,
)


CACHE_DIR = Path(_HERE) / "results" / "cache"
PERMS_DIR = Path(_HERE) / "results" / "perms"
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
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED,
                        help="Seed for paper-model load and per-fit shuffle.")
    parser.add_argument("--epochs", type=int, default=config.PER_PARTICIPANT_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.PER_PARTICIPANT_COG_LR)
    parser.add_argument("--max_perms", type=int, default=10**9,
                        help="Stop after this many perms in this run.")
    parser.add_argument("--max_runtime_minutes", type=float, default=10**9,
                        help="Stop after this many minutes of wall clock.")
    parser.add_argument(
        "--pick_seed", type=int, default=None,
        help=("Seed for the RANDOM ORDER in which we pick missing indices. "
              "Default: a fresh os.urandom seed each run, so two re-submissions "
              "don't pick the same indices first."),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    pick_rng = random.Random(args.pick_seed)  # None -> os.urandom seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load cache ---- #
    if not CACHE_DIR.exists() or not list(CACHE_DIR.glob("*.pt")):
        print(f"ERROR: no cache. Run 01_cache_features.py first.")
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
    missing = [i for i in range(n_splits) if i not in completed]
    if not missing:
        print("All splits done. Nothing to do.")
        return
    print(f"Remaining: {len(missing)}")

    # ---- Load model once ---- #
    print(f"Loading paper model (seed={args.seed})...")
    model, _ = load_paper_model(seed=args.seed, device=device)
    base_state = copy.deepcopy(model.state_dict())

    # ---- Loop ---- #
    t_start = time.time()
    n_done = 0
    while True:
        # Refresh missing each iteration to handle parallel instances.
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
        # Race-safety: if another instance just wrote it, skip.
        if out_path.exists():
            continue

        group_a, group_b = splits[idx]
        t0 = time.time()
        T_components = None
        cog_a = cog_b = None
        try:
            model.load_state_dict(base_state)
            ba = pool_group_batches(cache, list(group_a))
            cog_a, _ = fit_group_cached(
                model, ba, device,
                epochs=args.epochs, lr=args.lr,
                rng_seed=args.seed * 7919 + idx,
            )
            model.load_state_dict(base_state)
            bb = pool_group_batches(cache, list(group_b))
            cog_b, _ = fit_group_cached(
                model, bb, device,
                epochs=args.epochs, lr=args.lr,
                rng_seed=args.seed * 7919 + idx + 1,
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
        }
        _atomic_write_json(out_path, payload)
        n_done += 1
        print(f"  [{n_done:>4d}] split {idx:>4d}  T={T_components['T']:+8.3f}  "
              f"({elapsed:.1f}s)")

    print(f"\nThis run: completed {n_done} perms.")
    print(f"Cumulative: {len(_completed_indices())} / {n_splits}.")


if __name__ == "__main__":
    main()
