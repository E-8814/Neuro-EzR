"""
Evaluate the trained LLaMA+DiffEZR model by running its L1/L2/skip
predictions through the full discrete-event EZ Reader simulation.

This bridges:
  Training:   differentiable EZR approximation (parallel, smooth)
  Inference:  full EZR simulation (sequential, stochastic, cognitively plausible)

Compares three conditions:
  1. Human data (ground truth)
  2. Diff EZR (trained, parallel — what we already have)
  3. Full Simulation (sequential, with neural L1/L2/skip — the new thing)

Also runs the original formula-based EZ Reader for reference.

Usage:
    python eval_simulation.py [--gpu 0] [--num_runs 50] [--model_name TinyLlama/...]
"""

import os
import sys
import csv
import time
import argparse
from collections import defaultdict

import torch
import numpy as np
from scipy.stats import pearsonr

# --- Path setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_V2_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_V2_DIR)
EZR_DIR = os.path.join(ROOT_DIR, 'ez_reader')

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, SRC_V2_DIR)
sys.path.insert(0, EZR_DIR)

from model_llama import NeuralEZReaderLLaMA
from data_loader import load_provo, aggregate_by_sentence, split_aggregated
from geco_loader import load_geco, split_geco

from simulation_wrapper import (
    run_simulation_averaged,
    run_original_simulation_averaged,
)


# --------------------------------------------------------------------------- #
#  Logger
# --------------------------------------------------------------------------- #

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


# --------------------------------------------------------------------------- #
#  SUBTLEXus frequency
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
    return subtlex.get(w, 1)


# --------------------------------------------------------------------------- #
#  Extract neural predictions
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract_neural_predictions(model, tokens, predictabilities, device):
    """Run trained LLaMA model on one sentence. Returns L1, L2, skip lists."""
    wlens = [float(len(t)) for t in tokens]
    pred = model(
        [tokens],
        torch.tensor([predictabilities], dtype=torch.float32, device=device),
        torch.tensor([wlens], dtype=torch.float32, device=device),
    )
    l1 = pred['L1'][0].cpu().tolist()
    l2 = pred['L2'][0].cpu().tolist()
    skip = pred['skip_prob'][0].cpu().tolist()

    # Also get diff EZR predictions for comparison
    diff = {
        'trt': pred['total_reading_time'][0].cpu().tolist(),
        'ffd': pred['first_fixation'][0].cpu().tolist(),
        'gaze': pred['gaze_duration'][0].cpu().tolist(),
        'skip': pred['skip_prob'][0].cpu().tolist(),
    }
    return l1, l2, skip, diff


# --------------------------------------------------------------------------- #
#  Evaluate on a corpus
# --------------------------------------------------------------------------- #

def safe_pearsonr(x, y):
    """Pearson r that returns 0 on failure."""
    try:
        if len(x) < 3:
            return 0.0
        r, _ = pearsonr(x, y)
        return r if np.isfinite(r) else 0.0
    except Exception:
        return 0.0


def evaluate_corpus(model, aggregated_data, subtlex, device, num_runs=50,
                    run_original=True, label="Corpus"):
    """
    Run all models on a corpus and collect per-word predictions.

    Returns dict with human/simulation/diff/original arrays.
    """
    h = {'trt': [], 'ffd': [], 'gaze': [], 'skip': []}
    sim = {'trt': [], 'ffd': [], 'gaze': [], 'skip': []}
    diff = {'trt': [], 'ffd': [], 'gaze': [], 'skip': []}
    orig = {'trt': [], 'ffd': [], 'gaze': [], 'skip': []}

    n_sentences = len(aggregated_data)
    t_start = time.time()

    for idx, agg in enumerate(aggregated_data):
        tokens = agg.tokens
        preds = agg.predictabilities
        wlens = [len(t) for t in tokens]

        # Human data
        h['trt'].extend(agg.mean_trt)
        h['ffd'].extend(agg.mean_ffd)
        h['gaze'].extend(agg.mean_gaze)
        h['skip'].extend(agg.skip_rate)

        # Neural predictions from trained model
        l1, l2, skip_probs, diff_pred = extract_neural_predictions(
            model, tokens, preds, device
        )

        # Diff EZR (parallel, trained)
        diff['trt'].extend(diff_pred['trt'])
        diff['ffd'].extend(diff_pred['ffd'])
        diff['gaze'].extend(diff_pred['gaze'])
        diff['skip'].extend(diff_pred['skip'])

        # Full simulation (sequential, stochastic)
        # Use raw predictabilities for skip (calibrated for EZR's skip mechanism),
        # not neural skip_probs (calibrated for diff EZR's soft weighting)
        sim_result = run_simulation_averaged(
            tokens, l1, l2,
            skip_probs=None,
            predictabilities=preds,
            num_runs=num_runs,
        )
        sim['trt'].extend(sim_result['total_reading_time'])
        sim['ffd'].extend(sim_result['first_fixation_duration'])
        sim['gaze'].extend(sim_result['gaze_duration'])
        sim['skip'].extend(sim_result['skip_rate'])

        # Original EZ Reader (formula-based)
        if run_original:
            freqs = [get_real_frequency(t, subtlex) for t in tokens]
            orig_result = run_original_simulation_averaged(
                tokens, freqs, preds, num_runs=num_runs,
            )
            orig['trt'].extend(orig_result['total_reading_time'])
            orig['ffd'].extend(orig_result['first_fixation_duration'])
            orig['gaze'].extend(orig_result['gaze_duration'])
            orig['skip'].extend(orig_result['skip_rate'])

        # Free GPU memory between sentences
        if torch.cuda.is_available() and (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

        # Progress
        if (idx + 1) % 50 == 0 or idx == n_sentences - 1:
            elapsed = time.time() - t_start
            per_sent = elapsed / (idx + 1)
            remaining = per_sent * (n_sentences - idx - 1)
            print(f"  [{label}] {idx+1}/{n_sentences} sentences "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    # Convert to arrays
    for d in [h, sim, diff, orig]:
        for k in d:
            d[k] = np.array(d[k])

    return h, sim, diff, orig


# --------------------------------------------------------------------------- #
#  Print results
# --------------------------------------------------------------------------- #

def print_comparison(h, sim, diff, orig, label, W=80):
    """Print comparison table: Human vs Diff EZR vs Full Simulation vs Original."""
    measures = ['trt', 'ffd', 'gaze', 'skip']
    names = {'trt': 'TRT', 'ffd': 'FFD', 'gaze': 'Gaze', 'skip': 'Skip'}

    print(f"\n{'=' * W}")
    print(f"  {label}")
    print(f"  {len(h['trt'])} words")
    print(f"{'=' * W}")

    has_orig = len(orig['trt']) > 0

    # Header
    header = f"  {'Measure':<8} {'Human':>8}"
    header += f"  {'DiffEZR':>10} {'r_diff':>8}"
    header += f"  {'FullSim':>10} {'r_sim':>8}"
    if has_orig:
        header += f"  {'OrigEZR':>10} {'r_orig':>8}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for m in measures:
        hm = h[m]
        sm = sim[m]
        dm = diff[m]

        r_sim = safe_pearsonr(hm, sm)
        r_diff = safe_pearsonr(hm, dm)

        row = f"  {names[m]:<8} {np.mean(hm):>8.1f}"
        row += f"  {np.mean(dm):>10.1f} {r_diff:>8.3f}"
        row += f"  {np.mean(sm):>10.1f} {r_sim:>8.3f}"

        if has_orig:
            om = orig[m]
            r_orig = safe_pearsonr(hm, om)
            row += f"  {np.mean(om):>10.1f} {r_orig:>8.3f}"

        print(row)

    # Summary comparison: Diff EZR vs Full Simulation
    print(f"\n  {'Measure':<8} {'r_diff':>8} {'r_sim':>8} {'delta':>8} {'winner':>10}")
    print(f"  {'-' * 42}")
    for m in measures:
        r_sim = safe_pearsonr(h[m], sim[m])
        r_diff = safe_pearsonr(h[m], diff[m])
        delta = r_sim - r_diff
        winner = "Simulation" if delta > 0.005 else ("DiffEZR" if delta < -0.005 else "Tie")
        print(f"  {names[m]:<8} {r_diff:>8.3f} {r_sim:>8.3f} {delta:>+8.3f} {winner:>10}")

    # MAE comparison
    print(f"\n  MAE (ms):")
    print(f"  {'Measure':<8} {'DiffEZR':>10} {'FullSim':>10}", end="")
    if has_orig:
        print(f" {'OrigEZR':>10}", end="")
    print()
    for m in ['trt', 'ffd', 'gaze']:
        mae_diff = np.mean(np.abs(h[m] - diff[m]))
        mae_sim = np.mean(np.abs(h[m] - sim[m]))
        row = f"  {names[m]:<8} {mae_diff:>10.1f} {mae_sim:>10.1f}"
        if has_orig:
            mae_orig = np.mean(np.abs(h[m] - orig[m]))
            row += f" {mae_orig:>10.1f}"
        print(row)


def print_l1_l2_analysis(model, aggregated_data, device, label=""):
    """Show L1/L2 statistics from the trained model."""
    all_l1, all_l2, all_skip = [], [], []
    for agg in aggregated_data:
        l1, l2, skip, _ = extract_neural_predictions(
            model, agg.tokens, agg.predictabilities, device
        )
        all_l1.extend(l1)
        all_l2.extend(l2)
        all_skip.extend(skip)

    all_l1 = np.array(all_l1)
    all_l2 = np.array(all_l2)
    all_skip = np.array(all_skip)

    print(f"\n  Neural L1/L2 Statistics {label}:")
    print(f"    L1:   mean={np.mean(all_l1):.1f}ms  std={np.std(all_l1):.1f}  "
          f"range=[{np.min(all_l1):.1f}, {np.max(all_l1):.1f}]")
    print(f"    L2:   mean={np.mean(all_l2):.1f}ms  std={np.std(all_l2):.1f}  "
          f"range=[{np.min(all_l2):.1f}, {np.max(all_l2):.1f}]")
    print(f"    Skip: mean={np.mean(all_skip):.3f}  "
          f"range=[{np.min(all_skip):.3f}, {np.max(all_skip):.3f}]")
    print(f"    L2/L1 ratio: {np.mean(all_l2)/np.mean(all_l1):.3f}")
    if len(all_l1) > 2:
        r_l1_l2, _ = pearsonr(all_l1, all_l2)
        print(f"    r(L1, L2): {r_l1_l2:.3f}")


def print_sample_predictions(model, aggregated_data, device, num_runs=50, n_samples=3):
    """Show detailed per-word predictions for a few sentences."""
    print(f"\n  Sample Predictions (first {n_samples} sentences):")
    for idx, agg in enumerate(aggregated_data[:n_samples]):
        tokens = agg.tokens
        preds = agg.predictabilities
        l1, l2, skip_probs, diff_pred = extract_neural_predictions(
            model, tokens, preds, device
        )
        sim_result = run_simulation_averaged(
            tokens, l1, l2, skip_probs=None,
            predictabilities=preds, num_runs=num_runs,
        )

        print(f"\n    Sentence {idx+1}: {' '.join(tokens[:10])}{'...' if len(tokens)>10 else ''}")
        print(f"    {'Word':<12} {'L1':>5} {'L2':>5} {'Pskip':>6} "
              f"{'H_TRT':>6} {'S_TRT':>6} {'D_TRT':>6} "
              f"{'H_FFD':>6} {'S_FFD':>6} "
              f"{'H_Gaz':>6} {'S_Gaz':>6} "
              f"{'H_Skp':>6} {'S_Skp':>6}")
        print(f"    {'-' * 100}")

        for i, tok in enumerate(tokens):
            print(f"    {tok:<12} "
                  f"{l1[i]:>5.0f} {l2[i]:>5.0f} {skip_probs[i]:>6.3f} "
                  f"{agg.mean_trt[i]:>6.0f} {sim_result['total_reading_time'][i]:>6.0f} "
                  f"{diff_pred['trt'][i]:>6.0f} "
                  f"{agg.mean_ffd[i]:>6.0f} {sim_result['first_fixation_duration'][i]:>6.0f} "
                  f"{agg.mean_gaze[i]:>6.0f} {sim_result['gaze_duration'][i]:>6.0f} "
                  f"{agg.skip_rate[i]:>6.2f} {sim_result['skip_rate'][i]:>6.2f}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained LLaMA+DiffEZR via full EZR simulation"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=50,
                        help="Number of simulation runs to average")
    parser.add_argument("--model_name", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (auto-detected if not given)")
    parser.add_argument("--no_original", action="store_true",
                        help="Skip original EZ Reader (slow)")
    args = parser.parse_args()

    # Output
    output_path = os.path.join(ROOT_DIR, 'results', 'eval_simulation_results.txt')
    sys.stdout = Logger(output_path)

    W = 80
    print(f"{'=' * W}")
    print(f"  Neural EZ Reader: Full Simulation Evaluation")
    print(f"  Train with differentiable approximation, evaluate with real simulation")
    print(f"{'=' * W}")
    print(f"  Simulation runs per sentence: {args.num_runs}")
    print(f"  Model: {args.model_name}")

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    # --- Load data ---
    data_dir = os.path.join(ROOT_DIR, 'data')

    print("\nLoading GECO Corpus...")
    geco_raw = load_geco(
        os.path.join(data_dir, "Geco_MonolingualReadingData.csv"),
        os.path.join(data_dir, "Geco_EnglishMaterial.csv"),
        os.path.join(data_dir, "geco_predictability.pkl"),
    )
    train_raw, val_raw, test_raw = split_geco(geco_raw)
    geco_agg = aggregate_by_sentence(geco_raw, min_participants=5)
    train_ids = set(sd.text_id for sd in train_raw)
    val_ids = set(sd.text_id for sd in val_raw)
    geco_test = [a for a in geco_agg if a.text_id not in train_ids and a.text_id not in val_ids]
    print(f"  GECO test: {len(geco_test)} sentences, "
          f"{sum(len(s) for s in geco_test)} words")

    print("Loading Provo Corpus...")
    provo_raw = load_provo(os.path.join(data_dir, 'Provo_Corpus-Eyetracking_Data.csv'))
    provo_all = aggregate_by_sentence(provo_raw, min_participants=10)
    print(f"  Provo:     {len(provo_all)} sentences, "
          f"{sum(len(s) for s in provo_all)} words")

    subtlex = None
    if not args.no_original:
        print("Loading SUBTLEXus...")
        subtlex = load_subtlexus(os.path.join(data_dir, 'SUBTLEXus.txt'))
        print(f"  {len(subtlex):,} entries")

    # --- Load model ---
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # Auto-detect from model_name
        safe_name = args.model_name.replace('/', '_')
        ckpt_path = os.path.join(ROOT_DIR, f'checkpoints_v2/geco_{safe_name}', 'best_model.pt')

    print(f"\nLoading model from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = NeuralEZReaderLLaMA(
        model_name=ckpt.get('model_name', args.model_name),
        freeze_layers=ckpt.get('freeze_layers', 16),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    print(f"  Loaded (epoch {ckpt.get('epoch', '?')})")

    # Print learned EZR parameters
    ezr = model.ezreader
    print(f"\n  Learned EZR parameters:")
    print(f"    saccade_time:     {ezr.saccade_time.item():.1f} ms")
    print(f"    attention_shift:  {ezr.attention_shift.item():.1f} ms")
    print(f"    eccentricity:     {ezr.eccentricity.item():.4f}")
    print(f"    skip_sharpness:   {ezr.skip_sharpness.item():.2f}")
    print(f"    l2_contribution:  {ezr.l2_contribution.item():.4f}")
    print(f"    regr_threshold:   {ezr.regression_threshold.item():.1f} ms")
    print(f"    regr_sharpness:   {ezr.regression_sharpness.item():.4f}")
    print(f"    regr_cost_scale:  {ezr.regression_cost_scale.item():.4f}")

    # --- L1/L2 analysis (small sample to avoid OOM) ---
    print_l1_l2_analysis(model, geco_test[:20], device, "(GECO test, first 20)")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ================================================================
    #  GECO Test Set Evaluation
    # ================================================================
    print(f"\n{'=' * W}")
    print(f"  EVALUATION: GECO Test Set (in-distribution)")
    print(f"{'=' * W}")

    t0 = time.time()
    h_geco, sim_geco, diff_geco, orig_geco = evaluate_corpus(
        model, geco_test, subtlex, device,
        num_runs=args.num_runs,
        run_original=not args.no_original,
        label="GECO",
    )
    print(f"\n  GECO evaluation took {time.time() - t0:.0f}s")
    print_comparison(h_geco, sim_geco, diff_geco, orig_geco, "GECO Test Results")

    # ================================================================
    #  Provo Evaluation (cross-corpus)
    # ================================================================
    print(f"\n{'=' * W}")
    print(f"  EVALUATION: Provo Corpus (cross-corpus generalization)")
    print(f"{'=' * W}")

    t0 = time.time()
    h_provo, sim_provo, diff_provo, orig_provo = evaluate_corpus(
        model, provo_all, subtlex, device,
        num_runs=args.num_runs,
        run_original=not args.no_original,
        label="Provo",
    )
    print(f"\n  Provo evaluation took {time.time() - t0:.0f}s")
    print_comparison(h_provo, sim_provo, diff_provo, orig_provo, "Provo Results (cross-corpus)")

    # ================================================================
    #  Sample predictions
    # ================================================================
    print_sample_predictions(model, provo_all, device,
                             num_runs=args.num_runs, n_samples=5)

    # ================================================================
    #  Summary
    # ================================================================
    print(f"\n{'=' * W}")
    print(f"  GRAND SUMMARY")
    print(f"{'=' * W}")

    print(f"\n  {'':15s} {'--- GECO Test ---':^30s} {'--- Provo (cross) ---':^30s}")
    print(f"  {'Measure':<15s} {'r_diff':>8} {'r_sim':>8} {'delta':>8}"
          f"   {'r_diff':>8} {'r_sim':>8} {'delta':>8}")
    print(f"  {'-' * 73}")

    for m, name in [('trt', 'TRT'), ('ffd', 'FFD'), ('gaze', 'Gaze'), ('skip', 'Skip')]:
        rd_g = safe_pearsonr(h_geco[m], diff_geco[m])
        rs_g = safe_pearsonr(h_geco[m], sim_geco[m])
        rd_p = safe_pearsonr(h_provo[m], diff_provo[m])
        rs_p = safe_pearsonr(h_provo[m], sim_provo[m])
        print(f"  {name:<15s} {rd_g:>8.3f} {rs_g:>8.3f} {rs_g-rd_g:>+8.3f}"
              f"   {rd_p:>8.3f} {rs_p:>8.3f} {rs_p-rd_p:>+8.3f}")

    print(f"\n  Key: r_diff = Differentiable EZR (parallel, trained)")
    print(f"       r_sim  = Full Simulation (sequential, stochastic)")
    print(f"       delta  = r_sim - r_diff (positive = simulation is better)")

    print(f"\n  The full simulation produces cognitively plausible sequential")
    print(f"  reading behavior while achieving word-level predictions.")
    print(f"\n{'=' * W}")


if __name__ == "__main__":
    main()
