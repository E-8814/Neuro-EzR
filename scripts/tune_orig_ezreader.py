"""
Tune the Original (discrete) EZ Reader on GECO training data.

Uses scipy.optimize to find the best formula parameters (alpha1, alpha2,
alpha3, delta, eccentricity) that minimize MSE on GECO training data.

The discrete simulation is non-differentiable and stochastic, so we use
Nelder-Mead optimization on a subset of sentences with averaged runs.

Parameters tuned:
  - alpha1:       base processing time
  - alpha2:       frequency scaling
  - alpha3:       predictability scaling
  - delta:        L2/L1 ratio
  - eccentricity: distance scaling exponent

Usage:
    python3 -u src_diff_gpu/tune_orig_ezreader.py
    python3 -u src_diff_gpu/tune_orig_ezreader.py --n-sentences 500 --n-runs 10
"""

import os
import sys
import csv
import json
import math
import time
import random
import argparse
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'archive', 'original_ezreader'))

from data_loader import aggregate_by_sentence
from geco_loader import load_geco, split_geco
from utilities import time_familiarity_check, time_lexical_access
from ez_wrapper import run_simulation_averaged
from ez_reader_engine import Simulation


# --------------------------------------------------------------------------- #
#  Frequency helpers
# --------------------------------------------------------------------------- #

def load_subtlexus(path):
    freq = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            freq[row['Word'].lower()] = int(row['FREQcount'])
    return freq


def get_real_frequency(word, subtlex):
    w = word.lower().strip(".,;:!?\"'()[]{}").replace("\u2019", "'")
    if w in subtlex:
        return max(1, subtlex[w])
    for variant in [w.replace("'", ""), w.split("'")[0], w.split("-")[0]]:
        if variant in subtlex:
            return max(1, subtlex[variant])
    length = len(w)
    if length <= 3:   return 50000
    elif length <= 5: return 10000
    elif length <= 7: return 2000
    else:             return 500


# --------------------------------------------------------------------------- #
#  Compute L1/L2 with given parameters
# --------------------------------------------------------------------------- #

def compute_l1_l2(tokens, predictabilities, subtlex, alpha1, alpha2, alpha3,
                  delta, eccentricity):
    """Compute L1/L2 for each word using the given formula parameters."""
    l1_list, l2_list = [], []
    for token, pred in zip(tokens, predictabilities):
        freq = get_real_frequency(token, subtlex)
        wordlen = len(token)
        tL1 = time_familiarity_check(
            0, wordlen, freq, pred, eccentricity, alpha1, alpha2, alpha3
        )
        tL1 = max(1.0, tL1)
        tL2 = time_lexical_access(freq, pred, delta, alpha1, alpha2, alpha3)
        tL2 = max(1.0, tL2)
        l1_list.append(tL1)
        l2_list.append(tL2)
    return l1_list, l2_list


# --------------------------------------------------------------------------- #
#  Objective function
# --------------------------------------------------------------------------- #

class ObjectiveFunction:
    """
    Callable objective for scipy.optimize.

    Evaluates the Original EZ Reader simulation with given parameters
    on a subset of sentences.
    """

    def __init__(self, sentences, subtlex, n_runs=10, verbose=True):
        self.sentences = sentences
        self.subtlex = subtlex
        self.n_runs = n_runs
        self.verbose = verbose
        self.eval_count = 0
        self.best_loss = float('inf')
        self.best_params = None

    def __call__(self, x):
        """
        x = [alpha1, alpha2, alpha3, delta, eccentricity]
        Returns: loss (MSE on TRT + correlation penalty)
        """
        alpha1, alpha2, alpha3, delta, eccentricity = x

        # Enforce parameter bounds via penalty
        penalty = 0.0
        if alpha1 < 20 or alpha1 > 300:
            penalty += 1e6
        if alpha2 < 0.5 or alpha2 > 10:
            penalty += 1e6
        if alpha3 < 0 or alpha3 > 100:
            penalty += 1e6
        if delta < 0.05 or delta > 1.0:
            penalty += 1e6
        if eccentricity < 1.0 or eccentricity > 2.0:
            penalty += 1e6

        if penalty > 0:
            self.eval_count += 1
            return penalty

        all_h_trt, all_p_trt = [], []
        all_h_ffd, all_p_ffd = [], []

        for agg in self.sentences:
            tokens = agg.tokens
            preds = agg.predictabilities

            l1, l2 = compute_l1_l2(
                tokens, preds, self.subtlex,
                alpha1, alpha2, alpha3, delta, eccentricity
            )

            result = run_simulation_averaged(
                tokens, l1, l2, preds,
                num_runs=self.n_runs,
                timeout_seconds=3.0,
            )

            if result['success']:
                all_h_trt.extend(agg.mean_trt)
                all_p_trt.extend(result['total_reading_time'])
                all_h_ffd.extend(agg.mean_ffd)
                all_p_ffd.extend(result['first_fixation_duration'])

        if len(all_h_trt) < 10:
            self.eval_count += 1
            return 1e6

        h_trt = np.array(all_h_trt)
        p_trt = np.array(all_p_trt)
        h_ffd = np.array(all_h_ffd)
        p_ffd = np.array(all_p_ffd)

        # Loss: MSE on TRT + MSE on FFD - correlation bonus
        mse_trt = np.mean((h_trt - p_trt) ** 2)
        mse_ffd = np.mean((h_ffd - p_ffd) ** 2)

        r_trt = np.corrcoef(h_trt, p_trt)[0, 1] if np.std(p_trt) > 0 else 0.0
        r_ffd = np.corrcoef(h_ffd, p_ffd)[0, 1] if np.std(p_ffd) > 0 else 0.0

        # Combined loss: MSE with correlation penalty
        # We want to minimize MSE and maximize correlation
        loss = mse_trt + mse_ffd - 5000.0 * (r_trt + r_ffd)

        self.eval_count += 1

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = x.copy()

        if self.verbose and self.eval_count % 5 == 0:
            mae_trt = np.mean(np.abs(h_trt - p_trt))
            print(f"  Eval {self.eval_count:4d} | loss={loss:10.0f} | "
                  f"r_TRT={r_trt:.3f}  r_FFD={r_ffd:.3f} | "
                  f"MAE_TRT={mae_trt:.1f}ms | "
                  f"a1={alpha1:.1f} a2={alpha2:.2f} a3={alpha3:.1f} "
                  f"d={delta:.3f} ecc={eccentricity:.3f}")

        return loss


# --------------------------------------------------------------------------- #
#  Full evaluation on a dataset
# --------------------------------------------------------------------------- #

def full_evaluate(sentences, subtlex, alpha1, alpha2, alpha3, delta,
                  eccentricity, n_runs=20, label=""):
    """Run full evaluation with given parameters."""
    all_h_trt, all_p_trt = [], []
    all_h_ffd, all_p_ffd = [], []

    for agg in sentences:
        tokens = agg.tokens
        preds = agg.predictabilities

        l1, l2 = compute_l1_l2(
            tokens, preds, subtlex,
            alpha1, alpha2, alpha3, delta, eccentricity
        )

        result = run_simulation_averaged(
            tokens, l1, l2, preds,
            num_runs=n_runs,
            timeout_seconds=5.0,
        )

        if result['success']:
            all_h_trt.extend(agg.mean_trt)
            all_p_trt.extend(result['total_reading_time'])
            all_h_ffd.extend(agg.mean_ffd)
            all_p_ffd.extend(result['first_fixation_duration'])

    h_trt = np.array(all_h_trt)
    p_trt = np.array(all_p_trt)
    h_ffd = np.array(all_h_ffd)
    p_ffd = np.array(all_p_ffd)

    r_trt = np.corrcoef(h_trt, p_trt)[0, 1] if np.std(p_trt) > 0 else 0.0
    r_ffd = np.corrcoef(h_ffd, p_ffd)[0, 1] if np.std(p_ffd) > 0 else 0.0
    mae_trt = np.mean(np.abs(h_trt - p_trt))
    mae_ffd = np.mean(np.abs(h_ffd - p_ffd))

    print(f"\n  {label} ({len(all_h_trt)} words):")
    print(f"    r_TRT={r_trt:.3f}  r_FFD={r_ffd:.3f}")
    print(f"    MAE_TRT={mae_trt:.1f}ms  MAE_FFD={mae_ffd:.1f}ms")
    print(f"    Mean pred TRT={np.mean(p_trt):.1f}ms  (human={np.mean(h_trt):.1f}ms)")
    print(f"    Mean pred FFD={np.mean(p_ffd):.1f}ms  (human={np.mean(h_ffd):.1f}ms)")

    return {'r_trt': r_trt, 'r_ffd': r_ffd, 'mae_trt': mae_trt, 'mae_ffd': mae_ffd}


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sentences", type=int, default=300,
                        help="Number of training sentences to use for optimization")
    parser.add_argument("--n-runs", type=int, default=10,
                        help="Simulation runs per sentence during optimization")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Max optimizer iterations")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints_tuned_orig')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load data
    print("Loading GECO Corpus...")
    reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
    material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
    pred_path = os.path.join(data_dir, "geco_predictability.pkl")

    geco_raw = load_geco(reading_path, material_path, pred_path)
    train_raw, val_raw, test_raw = split_geco(geco_raw)

    all_agg = aggregate_by_sentence(geco_raw, min_participants=5)
    train_ids = set(sd.text_id for sd in train_raw)
    val_ids = set(sd.text_id for sd in val_raw)

    train_agg = [a for a in all_agg if a.text_id in train_ids]
    val_agg = [a for a in all_agg if a.text_id in val_ids]
    test_agg = [a for a in all_agg if a.text_id not in train_ids and a.text_id not in val_ids]

    print(f"  Train: {len(train_agg)} sentences")
    print(f"  Val:   {len(val_agg)} sentences")
    print(f"  Test:  {len(test_agg)} sentences")

    # Load frequency data
    print("Loading SUBTLEXus...")
    subtlex = load_subtlexus(os.path.join(data_dir, 'SUBTLEXus.txt'))
    print(f"  {len(subtlex):,} entries")

    # Subsample training data for optimization (full set is too slow)
    n_opt = min(args.n_sentences, len(train_agg))
    rng = random.Random(42)
    opt_sentences = rng.sample(train_agg, n_opt)
    print(f"\nUsing {n_opt} sentences for optimization ({args.n_runs} runs each)")

    # ---------------------------------------------------------------------- #
    #  Step 1: Evaluate with literature defaults
    # ---------------------------------------------------------------------- #
    print(f"\n{'='*80}")
    print("  LITERATURE DEFAULTS")
    print(f"{'='*80}")
    print("  alpha1=104, alpha2=3.4, alpha3=39, delta=0.34, eccentricity=1.15")

    full_evaluate(val_agg, subtlex, 104, 3.4, 39, 0.34, 1.15,
                  n_runs=20, label="Val set (defaults)")

    # ---------------------------------------------------------------------- #
    #  Step 2: Optimize with Nelder-Mead
    # ---------------------------------------------------------------------- #
    print(f"\n{'='*80}")
    print("  NELDER-MEAD OPTIMIZATION")
    print(f"{'='*80}")

    objective = ObjectiveFunction(
        opt_sentences, subtlex,
        n_runs=args.n_runs,
        verbose=True,
    )

    # Starting point: literature defaults
    x0 = np.array([104.0, 3.4, 39.0, 0.34, 1.15])

    print(f"\n  Starting optimization (max {args.max_iter} iterations)...")
    print(f"  Initial: alpha1={x0[0]}, alpha2={x0[1]}, alpha3={x0[2]}, "
          f"delta={x0[3]}, ecc={x0[4]}")

    t0 = time.time()

    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={
            'maxiter': args.max_iter,
            'maxfev': args.max_iter * 10,
            'xatol': 0.5,
            'fatol': 100,
            'adaptive': True,
        },
    )

    elapsed = time.time() - t0

    print(f"\n  Optimization finished in {elapsed:.0f}s ({objective.eval_count} evaluations)")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")

    # Extract best parameters
    best_alpha1, best_alpha2, best_alpha3, best_delta, best_ecc = result.x

    print(f"\n  Optimized parameters:")
    print(f"    alpha1       = {best_alpha1:.2f}   (default: 104)")
    print(f"    alpha2       = {best_alpha2:.4f}   (default: 3.4)")
    print(f"    alpha3       = {best_alpha3:.2f}   (default: 39)")
    print(f"    delta        = {best_delta:.4f}   (default: 0.34)")
    print(f"    eccentricity = {best_ecc:.4f}   (default: 1.15)")

    # ---------------------------------------------------------------------- #
    #  Step 3: Evaluate optimized parameters on val and test
    # ---------------------------------------------------------------------- #
    print(f"\n{'='*80}")
    print("  EVALUATION WITH OPTIMIZED PARAMETERS")
    print(f"{'='*80}")

    val_tuned = full_evaluate(
        val_agg, subtlex,
        best_alpha1, best_alpha2, best_alpha3, best_delta, best_ecc,
        n_runs=20, label="Val set (tuned)"
    )

    test_tuned = full_evaluate(
        test_agg, subtlex,
        best_alpha1, best_alpha2, best_alpha3, best_delta, best_ecc,
        n_runs=20, label="Test set (tuned)"
    )

    # Also re-evaluate defaults on test for comparison
    print(f"\n  --- Comparison on Test Set ---")
    test_default = full_evaluate(
        test_agg, subtlex, 104, 3.4, 39, 0.34, 1.15,
        n_runs=20, label="Test set (defaults)"
    )

    print(f"\n  Improvement on test set:")
    print(f"    r_TRT: {test_default['r_trt']:.3f} → {test_tuned['r_trt']:.3f} "
          f"(+{test_tuned['r_trt'] - test_default['r_trt']:.3f})")
    print(f"    r_FFD: {test_default['r_ffd']:.3f} → {test_tuned['r_ffd']:.3f} "
          f"(+{test_tuned['r_ffd'] - test_default['r_ffd']:.3f})")
    print(f"    MAE TRT: {test_default['mae_trt']:.1f} → {test_tuned['mae_trt']:.1f}ms "
          f"({test_tuned['mae_trt'] - test_default['mae_trt']:+.1f})")

    # ---------------------------------------------------------------------- #
    #  Save results
    # ---------------------------------------------------------------------- #
    results = {
        'optimized_params': {
            'alpha1': float(best_alpha1),
            'alpha2': float(best_alpha2),
            'alpha3': float(best_alpha3),
            'delta': float(best_delta),
            'eccentricity': float(best_ecc),
        },
        'default_params': {
            'alpha1': 104.0,
            'alpha2': 3.4,
            'alpha3': 39.0,
            'delta': 0.34,
            'eccentricity': 1.15,
        },
        'val_metrics_tuned': {k: float(v) for k, v in val_tuned.items()},
        'test_metrics_tuned': {k: float(v) for k, v in test_tuned.items()},
        'test_metrics_default': {k: float(v) for k, v in test_default.items()},
        'optimization': {
            'n_sentences': n_opt,
            'n_runs': args.n_runs,
            'n_evaluations': objective.eval_count,
            'time_seconds': elapsed,
        },
    }

    out_path = os.path.join(ckpt_dir, 'tuned_orig_params.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
