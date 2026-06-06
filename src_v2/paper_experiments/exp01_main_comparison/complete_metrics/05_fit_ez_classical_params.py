"""
Fit a small set of classical-E-Z-Reader parameters to GECO train.

We optimize 8 parameters that most directly affect predicted reading times:
    alpha1, alpha2, alpha3   (familiarity-check formula)
    eccentricity             (visual decline factor)
    delta                    (lexical-access ratio L2 = delta * L1)
    lambda                   (refixation parameter)
    saccade_programming      (M1)
    saccade_finishing        (M2)

Loss: normalized weighted sum of MAEs across FFD, Gaze, TRT, and skip on
a subsample of GECO train sentences. Each loss evaluation runs the
simulator with N_MC Monte Carlo runs per sentence; predictions are
averaged across runs and compared to human aggregated values.

Outputs:
    complete_metrics/ez_classical/fitted_params.json

The eval script 04_eval_ez_classical.py auto-loads this JSON if present.

Usage:
    python -u .../05_fit_ez_classical_params.py
    python -u .../05_fit_ez_classical_params.py --n_sentences 200 --n_mc 50 --maxiter 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC_V2 = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
REPO_ROOT = os.path.dirname(SRC_V2)
ARCHIVE_BASELINES = os.path.join(REPO_ROOT, "archive", "baselines")
ORIG_EZ = os.path.join(REPO_ROOT, "archive", "original_ezreader")

for p in (SRC_V2, ARCHIVE_BASELINES, ORIG_EZ, _HERE,
          os.path.join(_HERE, "ez_classical")):
    if p not in sys.path:
        sys.path.insert(0, p)

from paper_experiments.utils.load_data import (  # noqa: E402
    load_geco_aggregated, load_subtlex, word_frequency,
)

from ez_classical.wrapper_with_gaze import run_classical_averaged  # noqa: E402


OUT_DIR = Path(_HERE) / "ez_classical"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FITTED_JSON = OUT_DIR / "fitted_params.json"


# --------------------------------------------------------------------------- #
#  Parameter set
# --------------------------------------------------------------------------- #

# (name, default, lower bound, upper bound)
PARAMS = [
    ("alpha1",              104.0,  50.0,  250.0),
    ("alpha2",                3.4,   0.5,   12.0),
    ("alpha3",               39.0,   5.0,   90.0),
    ("eccentricity",          1.15,  1.00,   1.50),
    ("delta",                 0.34,  0.05,   0.95),
    ("lambda",                0.16,  0.01,   0.80),
    ("saccade_programming", 125.0,  60.0,  220.0),
    ("saccade_finishing",    25.0,  10.0,   80.0),
]
PARAM_NAMES = [p[0] for p in PARAMS]
DEFAULTS    = np.array([p[1] for p in PARAMS], dtype=float)
LOWERS      = np.array([p[2] for p in PARAMS], dtype=float)
UPPERS      = np.array([p[3] for p in PARAMS], dtype=float)


def _vec_to_dict(x: np.ndarray) -> Dict[str, float]:
    return {name: float(v) for name, v in zip(PARAM_NAMES, x)}


def _clip(x: np.ndarray) -> np.ndarray:
    return np.minimum(UPPERS, np.maximum(LOWERS, x))


# --------------------------------------------------------------------------- #
#  Worker for multiprocessing
# --------------------------------------------------------------------------- #

# Module-level so multiprocessing can pickle.
def _simulate_one(args):
    """args = (sent_idx, tokens, freqs, preds, n_mc, model_params_dict)."""
    sent_idx, tokens, freqs, preds, n_mc, model_params = args
    out = run_classical_averaged(
        tokens=list(tokens),
        frequencies=list(freqs),
        predictabilities=list(preds),
        num_runs=n_mc,
        model_params=model_params,
    )
    return sent_idx, out


# --------------------------------------------------------------------------- #
#  Loss function
# --------------------------------------------------------------------------- #


def _build_tasks(agg_list, subtlex):
    """Pre-compute the per-sentence (tokens, frequencies, predictabilities) data."""
    tasks_meta = []
    for agg in agg_list:
        tokens = list(agg.tokens)
        freqs = [float(word_frequency(t, subtlex)) for t in tokens]
        preds = []
        for i in range(len(tokens)):
            try:
                p = float(agg.predictabilities[i])
            except (TypeError, IndexError, ValueError):
                p = 0.05
            if p != p:
                p = 0.05
            preds.append(max(0.0, min(1.0, p)))
        tasks_meta.append((tokens, freqs, preds))
    # Pre-compute human targets for all sentences combined
    h_trt = np.array([v for agg in agg_list for v in agg.mean_trt],  dtype=float)
    h_ffd = np.array([v for agg in agg_list for v in agg.mean_ffd],  dtype=float)
    h_gaze = np.array([v for agg in agg_list for v in agg.mean_gaze], dtype=float)
    h_skip = np.array([v for agg in agg_list for v in agg.skip_rate], dtype=float)
    return tasks_meta, (h_trt, h_ffd, h_gaze, h_skip)


def _evaluate(params_vec, tasks_meta, humans, n_mc, pool, weights):
    """Run simulator with these params, return scalar loss."""
    params_dict = _vec_to_dict(params_vec)
    tasks = [
        (i, t, f, p, n_mc, params_dict)
        for i, (t, f, p) in enumerate(tasks_meta)
    ]

    pred_per_sent = [None] * len(tasks)
    for sent_idx, out in pool.imap_unordered(_simulate_one, tasks, chunksize=4):
        pred_per_sent[sent_idx] = out

    # Flatten predictions in the same order as humans
    pred_trt, pred_ffd, pred_gaze, pred_skip = [], [], [], []
    for out, (tokens, _, _) in zip(pred_per_sent, tasks_meta):
        n = len(tokens)
        if out is None or not out.get("success"):
            pred_trt.extend([0.0] * n)
            pred_ffd.extend([0.0] * n)
            pred_gaze.extend([0.0] * n)
            pred_skip.extend([1.0] * n)
        else:
            pred_trt.extend(out["total_reading_time"])
            pred_ffd.extend(out["first_fixation_duration"])
            pred_gaze.extend(out["gaze_duration"])
            pred_skip.extend(out["skip_rate"])
    pred_trt  = np.asarray(pred_trt,  dtype=float)
    pred_ffd  = np.asarray(pred_ffd,  dtype=float)
    pred_gaze = np.asarray(pred_gaze, dtype=float)
    pred_skip = np.asarray(pred_skip, dtype=float)

    h_trt, h_ffd, h_gaze, h_skip = humans

    # Normalize each MAE by the mean of the human target so all four
    # contributions live on roughly the same scale.
    def _mae_rel(p, h):
        denom = max(np.mean(np.abs(h)), 1e-6)
        return float(np.mean(np.abs(p - h)) / denom)

    L_trt  = _mae_rel(pred_trt,  h_trt)
    L_ffd  = _mae_rel(pred_ffd,  h_ffd)
    L_gaze = _mae_rel(pred_gaze, h_gaze)
    L_skip = _mae_rel(pred_skip, h_skip)

    w_trt, w_ffd, w_gaze, w_skip = weights
    return (w_trt * L_trt + w_ffd * L_ffd
            + w_gaze * L_gaze + w_skip * L_skip), \
           {"L_trt": L_trt, "L_ffd": L_ffd, "L_gaze": L_gaze, "L_skip": L_skip}


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sentences", type=int, default=200,
                        help="GECO train sentences to use per loss eval (default 200).")
    parser.add_argument("--n_mc", type=int, default=50,
                        help="MC simulations per sentence per loss eval (default 50).")
    parser.add_argument("--maxiter", type=int, default=200,
                        help="Nelder-Mead max iterations (default 200).")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--sample_seed", type=int, default=0,
                        help="Seed for selecting the GECO-train subset.")
    parser.add_argument("--weights", type=float, nargs=4,
                        default=[1.0, 1.0, 1.0, 1.0],
                        help="Weights on (TRT, FFD, Gaze, skip) MAE_rel terms.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if FITTED_JSON.exists() and not args.force:
        print(f"{FITTED_JSON} already exists; pass --force to refit.")
        return

    print("Loading GECO train + SUBTLEX...")
    train_agg = load_geco_aggregated("train")
    subtlex = load_subtlex()
    print(f"  GECO train: {len(train_agg)} sentences total")

    rng = random.Random(args.sample_seed)
    subset = rng.sample(train_agg, min(args.n_sentences, len(train_agg)))
    print(f"  Using random subset of {len(subset)} sentences for fitting "
          f"(seed={args.sample_seed})")
    print(f"  Settings: n_mc={args.n_mc}  maxiter={args.maxiter}  "
          f"workers={args.workers}  weights={args.weights}")

    tasks_meta, humans = _build_tasks(subset, subtlex)
    print(f"  Total words in subset: {len(humans[0])}")

    pool = Pool(processes=args.workers)
    try:
        # ---- Initial loss at defaults ---- #
        t0 = time.time()
        loss0, comp0 = _evaluate(DEFAULTS, tasks_meta, humans, args.n_mc, pool,
                                  args.weights)
        print(f"\nInitial loss (Reichle 2003 defaults): {loss0:.4f}")
        print(f"  components: {comp0}")
        print(f"  initial eval took {time.time()-t0:.1f}s")

        # ---- Nelder-Mead in [0, 1] normalized space ---- #
        # Map x_norm in [0,1] -> param value in [lower, upper]
        def _to_real(xn):  return LOWERS + np.clip(xn, 0.0, 1.0) * (UPPERS - LOWERS)
        def _to_norm(xr): return (xr - LOWERS) / (UPPERS - LOWERS)

        history = []
        n_evals = [0]

        def loss_norm(xn):
            xr = _to_real(xn)
            loss, comp = _evaluate(xr, tasks_meta, humans, args.n_mc, pool,
                                    args.weights)
            n_evals[0] += 1
            history.append({"eval": n_evals[0], "loss": loss,
                            "params": _vec_to_dict(xr), "components": comp})
            if n_evals[0] % 10 == 0 or n_evals[0] == 1:
                print(f"  eval {n_evals[0]:>4d}  loss={loss:.4f}  "
                      f"trt={comp['L_trt']:.3f} ffd={comp['L_ffd']:.3f} "
                      f"gaze={comp['L_gaze']:.3f} skip={comp['L_skip']:.3f}")
            return loss

        x0_norm = _to_norm(DEFAULTS)
        print(f"\nStarting Nelder-Mead optimization...\n")
        t0 = time.time()
        res = minimize(
            loss_norm, x0_norm,
            method="Nelder-Mead",
            options=dict(maxiter=args.maxiter, xatol=0.005, fatol=0.001,
                          adaptive=True, disp=True),
        )
        elapsed = time.time() - t0
        x_final = _to_real(np.clip(res.x, 0.0, 1.0))
        loss_final, comp_final = _evaluate(x_final, tasks_meta, humans,
                                            args.n_mc, pool, args.weights)
        print(f"\nOptimization finished in {elapsed:.0f}s, {n_evals[0]} evals.")
        print(f"  final loss: {loss_final:.4f}")
        print(f"  components: {comp_final}")
        print(f"  improvement vs defaults: "
              f"{100*(loss0 - loss_final)/max(loss0, 1e-9):+.1f}%")

        # ---- Print before/after ---- #
        print("\n" + "=" * 72)
        print(f"{'param':<22s}  {'default':>10s}  {'fitted':>10s}  {'%Δ':>8s}")
        print("-" * 72)
        for name, default, fitted in zip(PARAM_NAMES, DEFAULTS, x_final):
            pct = 100 * (fitted - default) / max(abs(default), 1e-9)
            print(f"{name:<22s}  {default:>10.3f}  {fitted:>10.3f}  {pct:>+7.1f}%")

        # ---- Save ---- #
        payload = {
            "fitted_params":  _vec_to_dict(x_final),
            "default_params": _vec_to_dict(DEFAULTS),
            "loss_default":   loss0,
            "loss_fitted":    loss_final,
            "components_default": comp0,
            "components_fitted":  comp_final,
            "n_sentences":    len(subset),
            "n_mc":           args.n_mc,
            "n_evals":        n_evals[0],
            "fit_seconds":    elapsed,
            "weights":        list(args.weights),
            "sample_seed":    args.sample_seed,
            "history":        history,
        }
        FITTED_JSON.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\nWrote {FITTED_JSON}")

    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
