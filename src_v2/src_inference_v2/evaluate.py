"""
Evaluate the trained Neural E-Z Reader via the discrete simulation.

Loads a checkpoint trained with the aligned diff EZR, extracts L1/L2/skip,
configures the discrete simulation with the learned parameters, and compares:
  1. Human data (ground truth)
  2. Diff EZR predictions (parallel, from the model's forward pass)
  3. Full Simulation (sequential, stochastic, with learned params)
  4. Original EZ Reader (formula-based, for reference)

Evaluates on GECO test set (in-distribution) and Provo (cross-corpus).

Usage:
    python evaluate.py [--model_name TinyLlama/...] [--num_runs 50]
"""

import os
import sys
import time
import argparse

import torch
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import NeuralEZReader
from simulation import (
    run_simulation_averaged,
    run_original_simulation_averaged,
    compute_integration_failures,
)
from data import (
    load_geco, split_geco, load_provo,
    aggregate_by_sentence, get_data_dir,
    load_subtlexus, get_frequency,
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
#  Extract neural predictions
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract_predictions(model, tokens, predictabilities, device):
    """Run trained model on one sentence. Returns L1, L2, skip, diff preds."""
    wlens = [float(len(t)) for t in tokens]
    pred = model(
        [tokens],
        torch.tensor([predictabilities], dtype=torch.float32, device=device),
        torch.tensor([wlens], dtype=torch.float32, device=device),
    )
    n = len(tokens)
    l1 = pred['L1'][0, :n].cpu().tolist()
    l2 = pred['L2'][0, :n].cpu().tolist()
    skip = pred['skip_prob'][0, :n].cpu().tolist()
    int_fail = pred['integration_failure_prob'][0, :n].cpu().tolist()

    diff = {
        'trt': pred['total_reading_time'][0, :n].cpu().tolist(),
        'ffd': pred['first_fixation'][0, :n].cpu().tolist(),
        'gaze': pred['gaze_duration'][0, :n].cpu().tolist(),
        'skip': skip,
    }
    return l1, l2, skip, int_fail, diff


# --------------------------------------------------------------------------- #
#  Evaluate on a corpus
# --------------------------------------------------------------------------- #

def safe_pearsonr(x, y):
    try:
        if len(x) < 3:
            return 0.0
        r, _ = pearsonr(x, y)
        return r if np.isfinite(r) else 0.0
    except Exception:
        return 0.0


def evaluate_corpus(model, aggregated_data, sim_params, subtlex, device,
                    num_runs=50, run_original=True, label="Corpus"):
    """Run all models on a corpus and collect per-word predictions."""
    h = {'trt': [], 'ffd': [], 'gaze': [], 'skip': []}
    sim = {'trt': [], 'ffd': [], 'gaze': [], 'skip': []}
    diff = {'trt': [], 'ffd': [], 'gaze': [], 'skip': []}
    orig = {'trt': [], 'ffd': [], 'gaze': [], 'skip': []}

    n_sentences = len(aggregated_data)
    t_start = time.time()

    for idx, agg in enumerate(aggregated_data):
        tokens = agg.tokens
        preds = agg.predictabilities

        # Human data (conditional on fixation)
        h['trt'].extend(agg.mean_trt)
        h['ffd'].extend(agg.mean_ffd)
        h['gaze'].extend(agg.mean_gaze)
        h['skip'].extend(agg.skip_rate)

        # Neural predictions
        l1, l2, skip_probs, int_fail, diff_pred = extract_predictions(
            model, tokens, preds, device,
        )

        # Diff EZR (parallel, trained) — conditional on fixation
        diff['trt'].extend(diff_pred['trt'])
        diff['ffd'].extend(diff_pred['ffd'])
        diff['gaze'].extend(diff_pred['gaze'])
        diff['skip'].extend(diff_pred['skip'])

        # Full simulation (sequential, stochastic)
        # Uses neural skip_probs, learned sim_params, per-word integration
        sim_result = run_simulation_averaged(
            tokens, l1, l2,
            skip_probs=skip_probs,
            sim_params=sim_params,
            integration_failures=int_fail,
            num_runs=num_runs,
        )
        # Use raw averages (including zeros for skipped runs).
        # The skip decision is where per-word variance comes from in the
        # simulation — fixation durations are dominated by motor overhead.
        sim['trt'].extend(sim_result['total_reading_time'])
        sim['ffd'].extend(sim_result['first_fixation_duration'])
        sim['gaze'].extend(sim_result['gaze_duration'])
        sim['skip'].extend(sim_result['skip_rate'])

        # Original EZ Reader (formula-based)
        if run_original and subtlex is not None:
            freqs = [get_frequency(t, subtlex) for t in tokens]
            orig_result = run_original_simulation_averaged(
                tokens, freqs, preds, num_runs=num_runs,
            )
            orig['trt'].extend(orig_result['total_reading_time'])
            orig['ffd'].extend(orig_result['first_fixation_duration'])
            orig['gaze'].extend(orig_result['gaze_duration'])
            orig['skip'].extend(orig_result['skip_rate'])

        if torch.cuda.is_available() and (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

        if (idx + 1) % 50 == 0 or idx == n_sentences - 1:
            elapsed = time.time() - t_start
            per_sent = elapsed / (idx + 1)
            remaining = per_sent * (n_sentences - idx - 1)
            print(f"  [{label}] {idx+1}/{n_sentences} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    for d in [h, sim, diff, orig]:
        for k in d:
            d[k] = np.array(d[k])

    return h, sim, diff, orig


# --------------------------------------------------------------------------- #
#  Print results
# --------------------------------------------------------------------------- #

def print_comparison(h, sim, diff, orig, label):
    measures = ['trt', 'ffd', 'gaze', 'skip']
    names = {'trt': 'TRT', 'ffd': 'FFD', 'gaze': 'Gaze', 'skip': 'Skip'}
    W = 80

    print(f"\n{'=' * W}")
    print(f"  {label} ({len(h['trt'])} words)")
    print(f"{'=' * W}")

    has_orig = len(orig['trt']) > 0

    header = f"  {'Measure':<8} {'Human':>8}"
    header += f"  {'DiffEZR':>10} {'r_diff':>8}"
    header += f"  {'FullSim':>10} {'r_sim':>8}"
    if has_orig:
        header += f"  {'OrigEZR':>10} {'r_orig':>8}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for m in measures:
        r_diff = safe_pearsonr(h[m], diff[m])
        r_sim = safe_pearsonr(h[m], sim[m])
        row = f"  {names[m]:<8} {np.mean(h[m]):>8.1f}"
        row += f"  {np.mean(diff[m]):>10.1f} {r_diff:>8.3f}"
        row += f"  {np.mean(sim[m]):>10.1f} {r_sim:>8.3f}"
        if has_orig:
            r_orig = safe_pearsonr(h[m], orig[m])
            row += f"  {np.mean(orig[m]):>10.1f} {r_orig:>8.3f}"
        print(row)

    # Summary: DiffEZR vs Simulation
    print(f"\n  {'Measure':<8} {'r_diff':>8} {'r_sim':>8} "
          f"{'delta':>8} {'winner':>10}")
    print(f"  {'-' * 42}")
    for m in measures:
        r_diff = safe_pearsonr(h[m], diff[m])
        r_sim = safe_pearsonr(h[m], sim[m])
        delta = r_sim - r_diff
        winner = ("Simulation" if delta > 0.005
                  else ("DiffEZR" if delta < -0.005 else "Tie"))
        print(f"  {names[m]:<8} {r_diff:>8.3f} {r_sim:>8.3f} "
              f"{delta:>+8.3f} {winner:>10}")

    # MAE
    print(f"\n  MAE (ms):")
    row_header = f"  {'Measure':<8} {'DiffEZR':>10} {'FullSim':>10}"
    if has_orig:
        row_header += f" {'OrigEZR':>10}"
    print(row_header)
    for m in ['trt', 'ffd', 'gaze']:
        mae_diff = np.mean(np.abs(h[m] - diff[m]))
        mae_sim = np.mean(np.abs(h[m] - sim[m]))
        row = f"  {names[m]:<8} {mae_diff:>10.1f} {mae_sim:>10.1f}"
        if has_orig:
            mae_orig = np.mean(np.abs(h[m] - orig[m]))
            row += f" {mae_orig:>10.1f}"
        print(row)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=50)
    parser.add_argument("--model_name", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no_original", action="store_true")
    args = parser.parse_args()

    # Output
    root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    output_path = os.path.join(
        root_dir, 'results', 'eval_simulation_v2_results.txt'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sys.stdout = Logger(output_path)

    W = 80
    print(f"{'=' * W}")
    print(f"  Neural E-Z Reader: Aligned Simulation Evaluation")
    print(f"  Diff EZR and simulation share parameter semantics")
    print(f"{'=' * W}")
    print(f"  Simulation runs: {args.num_runs}")
    print(f"  Model: {args.model_name}")

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    # --- Load data ---
    data_dir = get_data_dir()

    print("\nLoading GECO...")
    geco_raw = load_geco(
        os.path.join(data_dir, "Geco_MonolingualReadingData.csv"),
        os.path.join(data_dir, "Geco_EnglishMaterial.csv"),
        os.path.join(data_dir, "geco_predictability.pkl"),
    )
    train_raw, val_raw, test_raw = split_geco(geco_raw)
    geco_agg = aggregate_by_sentence(geco_raw, min_participants=5)
    train_ids = set(sd.text_id for sd in train_raw)
    val_ids = set(sd.text_id for sd in val_raw)
    geco_test = [a for a in geco_agg
                 if a.text_id not in train_ids and a.text_id not in val_ids]
    print(f"  GECO test: {len(geco_test)} sentences, "
          f"{sum(len(s) for s in geco_test)} words")

    print("Loading Provo...")
    provo_raw = load_provo(
        os.path.join(data_dir, 'Provo_Corpus-Eyetracking_Data.csv')
    )
    provo_all = aggregate_by_sentence(provo_raw, min_participants=10)
    print(f"  Provo: {len(provo_all)} sentences, "
          f"{sum(len(s) for s in provo_all)} words")

    subtlex = None
    if not args.no_original:
        print("Loading SUBTLEXus...")
        subtlex = load_subtlexus()
        print(f"  {len(subtlex):,} entries")

    # --- Load model ---
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        safe_name = args.model_name.replace('/', '_')
        ckpt_path = os.path.join(
            root_dir, f'checkpoints_v4/geco_{safe_name}', 'best_model.pt'
        )

    print(f"\nLoading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = NeuralEZReader(
        model_name=ckpt.get('model_name', args.model_name),
        freeze_layers=ckpt.get('freeze_layers', 16),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    print(f"  Loaded (epoch {ckpt.get('epoch', '?')})")

    # Get sim_params from checkpoint
    sim_params = ckpt.get('sim_params', model.get_sim_params())
    print(f"\n  Simulation parameters (from checkpoint):")
    for k, v in sim_params.items():
        print(f"    {k}: {v}")

    # ================================================================
    #  GECO Test
    # ================================================================
    print(f"\n{'=' * W}")
    print(f"  GECO Test Set (in-distribution)")
    print(f"{'=' * W}")

    t0 = time.time()
    h_geco, sim_geco, diff_geco, orig_geco = evaluate_corpus(
        model, geco_test, sim_params, subtlex, device,
        num_runs=args.num_runs,
        run_original=not args.no_original,
        label="GECO",
    )
    print(f"\n  GECO took {time.time() - t0:.0f}s")
    print_comparison(h_geco, sim_geco, diff_geco, orig_geco, "GECO Test")

    # ================================================================
    #  Provo (cross-corpus)
    # ================================================================
    print(f"\n{'=' * W}")
    print(f"  Provo Corpus (cross-corpus)")
    print(f"{'=' * W}")

    t0 = time.time()
    h_provo, sim_provo, diff_provo, orig_provo = evaluate_corpus(
        model, provo_all, sim_params, subtlex, device,
        num_runs=args.num_runs,
        run_original=not args.no_original,
        label="Provo",
    )
    print(f"\n  Provo took {time.time() - t0:.0f}s")
    print_comparison(
        h_provo, sim_provo, diff_provo, orig_provo,
        "Provo (cross-corpus)",
    )

    # ================================================================
    #  Grand Summary
    # ================================================================
    print(f"\n{'=' * W}")
    print(f"  GRAND SUMMARY")
    print(f"{'=' * W}")
    print(f"\n  {'':15s} {'--- GECO Test ---':^30s} "
          f"{'--- Provo (cross) ---':^30s}")
    print(f"  {'Measure':<15s} {'r_diff':>8} {'r_sim':>8} {'delta':>8}"
          f"   {'r_diff':>8} {'r_sim':>8} {'delta':>8}")
    print(f"  {'-' * 73}")

    for m, name in [('trt', 'TRT'), ('ffd', 'FFD'),
                     ('gaze', 'Gaze'), ('skip', 'Skip')]:
        rd_g = safe_pearsonr(h_geco[m], diff_geco[m])
        rs_g = safe_pearsonr(h_geco[m], sim_geco[m])
        rd_p = safe_pearsonr(h_provo[m], diff_provo[m])
        rs_p = safe_pearsonr(h_provo[m], sim_provo[m])
        print(f"  {name:<15s} {rd_g:>8.3f} {rs_g:>8.3f} "
              f"{rs_g-rd_g:>+8.3f}   {rd_p:>8.3f} {rs_p:>8.3f} "
              f"{rs_p-rd_p:>+8.3f}")

    print(f"\n  r_diff = Differentiable EZR (parallel, trained)")
    print(f"  r_sim  = Full Simulation (sequential, learned params)")
    print(f"  delta  = r_sim - r_diff (positive = simulation is better)")
    print(f"\n{'=' * W}")


if __name__ == "__main__":
    main()
