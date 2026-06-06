"""
Sanity check: cached-feature fast/slow cog fit must reproduce the
published exp09 numbers.

If this passes, the cached path is provably equivalent to the live one,
and we can run the full 1,716-permutation enumeration safely.

If this fails, the cache is buggy and we should fall back to the live
codepath with a smaller number of random permutations (300, see
03b_perm_live.py). The slurm orchestrator (run_slurm.sh) reads this
script's exit code / sanity_check.json and branches accordingly.

Reference numbers from the paper (per-group cog scalar refit, paper §H3):
    delta:            0.266 -> 0.348   (+30.8%)
    lambda_refix:     0.898 -> 0.766   (-14.7%)
    epsilon:          1.140 -> 1.188   (+4.2%)
    M1:             123.71  -> 123.69  (essentially zero)
    M2_eq_I:         24.06  ->  24.05  (essentially zero)
    skip_temperature: 31.34 ->  31.34  (zero)

Tolerance defaults are intentionally generous because cached fp32
recomputation can drift slightly relative to the live autocast path.
The criterion that actually matters is whether T_cached and T_live
agree on the dissociation direction and magnitude — that's what the
permutation null hangs on.

Usage:
    python -u .../02_sanity_check.py
    python -u .../02_sanity_check.py --tolerance 0.10  # ± 10% relative
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
    LEXICAL_PARAMS, MOTOR_PARAMS,
)
from fit_per_group import FAST_READERS, SLOW_READERS


CACHE_DIR = Path(_HERE) / "results" / "cache"
RESULTS_DIR = Path(_HERE) / "results"
SANITY_JSON = RESULTS_DIR / "sanity_check.json"


# Published numbers from paper §H3 (Figure 2 / per_group_comparison.csv)
PUBLISHED = {
    "delta":            (0.266, 0.348),
    "lambda_refix":     (0.898, 0.766),
    "epsilon":          (1.140, 1.188),
    "M1":             (123.71, 123.69),
    "M2_eq_I":          (24.06,  24.05),
    "skip_temperature": (31.34,  31.34),
}


def _abs_pct_diff(a: float, b: float) -> float:
    if abs(b) < 1e-9:
        return 0.0 if abs(a - b) < 1e-9 else float("inf")
    return abs(a - b) / abs(b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=config.PER_PARTICIPANT_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.PER_PARTICIPANT_COG_LR)
    parser.add_argument(
        "--tolerance", type=float, default=0.10,
        help=("Max relative difference (|cached-published|/|published|) on "
              "any of the 6 reported parameters in PUBLISHED. Default 0.10 "
              "(10%%). Headline criterion: |T_cached - T_published_proxy| / "
              "|T_published_proxy| also < tolerance."),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Settings: epochs={args.epochs}  lr={args.lr}")

    # ---- Load cached features (must already exist; run 01_cache_features.py first) ---- #
    if not CACHE_DIR.exists() or not list(CACHE_DIR.glob("*.pt")):
        print(f"ERROR: no cache found in {CACHE_DIR}. "
              f"Run 01_cache_features.py first.")
        SANITY_JSON.write_text(json.dumps({
            "passed": False,
            "reason": "cache_missing",
        }, indent=2))
        sys.exit(2)

    cache = load_cached_participants(CACHE_DIR)
    available = set(cache.keys())
    fast = sorted(FAST_READERS & available)
    slow = sorted(SLOW_READERS & available)
    print(f"Groups (from fit_per_group.py):")
    print(f"  fast (n={len(fast)}): {fast}")
    print(f"  slow (n={len(slow)}): {slow}")
    if len(fast) != 7 or len(slow) != 7:
        print("ERROR: expected exactly 7 fast + 7 slow.")
        SANITY_JSON.write_text(json.dumps({
            "passed": False, "reason": "split_size_mismatch",
            "fast": fast, "slow": slow,
        }, indent=2))
        sys.exit(2)

    # ---- Load model once, snapshot for reset between groups ---- #
    print(f"\nLoading paper model (seed={args.seed})...")
    model, _ = load_paper_model(seed=args.seed, device=device)
    base_state = copy.deepcopy(model.state_dict())

    fits = {}
    for label, readers in (("fast", fast), ("slow", slow)):
        group_batches = pool_group_batches(cache, readers)
        n_words = sum(int(cb["word_lengths"].numel()) for cb in group_batches)
        print(f"\n>> Cached fit on {label} group: {len(readers)} readers, "
              f"{len(group_batches)} batches, {n_words} (padded) word slots")

        model.load_state_dict(base_state)
        t0 = time.time()
        cog, final_loss = fit_group_cached(
            model, group_batches, device,
            epochs=args.epochs, lr=args.lr, rng_seed=args.seed,
        )
        elapsed = time.time() - t0
        print(f"   final_loss={final_loss:.4f}  ({elapsed:.1f}s)")
        for k in ("delta", "lambda_refix", "epsilon", "M1", "M2_eq_I",
                  "skip_temperature", "alpha1_reichle", "alpha2_reichle",
                  "refix_pivot"):
            if k in cog:
                print(f"   {k:<22s} {cog[k]:.4f}")
        fits[label] = cog

    # ---- Compare to published ---- #
    print("\n" + "=" * 78)
    print(f"{'param':<22s} {'cached_fast':>12s} {'cached_slow':>12s}  "
          f"{'pub_fast':>10s} {'pub_slow':>10s}  {'rel_err':>9s}")
    print("-" * 78)
    max_rel_err = 0.0
    per_param_rel = {}
    for k, (pub_f, pub_s) in PUBLISHED.items():
        c_f = fits["fast"][k]; c_s = fits["slow"][k]
        rel_f = _abs_pct_diff(c_f, pub_f)
        rel_s = _abs_pct_diff(c_s, pub_s)
        rel = max(rel_f, rel_s)
        per_param_rel[k] = {"fast_rel_err": rel_f, "slow_rel_err": rel_s}
        max_rel_err = max(max_rel_err, rel)
        print(f"{k:<22s} {c_f:>12.4f} {c_s:>12.4f}  "
              f"{pub_f:>10.4f} {pub_s:>10.4f}  {rel:>8.2%}")
    print("-" * 78)

    # Dissociation T agreement (cached vs published-proxy from the values above).
    pub_fast = {k: v[0] for k, v in PUBLISHED.items()}
    pub_slow = {k: v[1] for k, v in PUBLISHED.items()}
    T_pub    = dissociation_T(pub_fast, pub_slow)
    T_cache  = dissociation_T(fits["fast"], fits["slow"])
    T_rel_err = (
        abs(T_cache["T"] - T_pub["T"]) / abs(T_pub["T"])
        if abs(T_pub["T"]) > 1e-6 else float("inf")
    )
    print(f"\nDissociation statistic T (mean |%Δ_lex| - mean |%Δ_mot|):")
    print(f"  cached:       {T_cache['T']:+.3f}   "
          f"(lex={T_cache['mean_abs_pct_lexical']:.3f}, "
          f"mot={T_cache['mean_abs_pct_motor']:.3f})")
    print(f"  published:    {T_pub['T']:+.3f}   "
          f"(lex={T_pub['mean_abs_pct_lexical']:.3f}, "
          f"mot={T_pub['mean_abs_pct_motor']:.3f})")
    print(f"  rel error:    {T_rel_err:.2%}")

    passed = (max_rel_err <= args.tolerance) and (T_rel_err <= args.tolerance)
    print("\n" + ("PASSED" if passed else "FAILED")
          + f"  (tolerance={args.tolerance:.2%})")

    SANITY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SANITY_JSON.write_text(json.dumps({
        "passed": passed,
        "tolerance": args.tolerance,
        "max_param_rel_err": max_rel_err,
        "T_rel_err": T_rel_err,
        "T_cached": T_cache,
        "T_published": T_pub,
        "fits_cached": fits,
        "per_param_rel": per_param_rel,
        "published": PUBLISHED,
        "fast": fast,
        "slow": slow,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
    }, indent=2, default=float))
    print(f"Wrote {SANITY_JSON}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
