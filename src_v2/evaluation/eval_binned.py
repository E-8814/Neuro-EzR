"""
Bin-level evaluation: the way E-Z Reader was originally validated.

Instead of word-level Pearson correlations, bins words by frequency (5 bins),
predictability (5 bins), and word length (5 bins), then compares mean reading
times per bin. RMSD and correlation are computed across bins.

This is a fair comparison: E-Z Reader was designed and evaluated this way
(Reichle et al. 1998, 2003, 2006, 2009).

All models trained on GECO, evaluated on full Provo (cross-corpus).

Usage:
    python3 -u src_v2/eval_binned.py
    python3 -u src_v2/eval_binned.py --num-runs 100   # more sim runs for orig EZ
"""

import os
import sys
import csv
import math
import time
import argparse
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lm_model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'archive', 'original_ezreader'))

from diff_ezreader import DifferentiableEZReader
import model_lstm
from model_lstm import NeuralEZReader, Vocabulary
from model_bert import NeuralEZReaderBERT
from model_llama import NeuralEZReaderLLaMA
from model_llama_direct import DirectRegressionLLaMA
from data_loader import load_provo, aggregate_by_sentence
from geco_loader import load_geco, split_geco
from ez_wrapper import run_original_simulation_averaged
from utilities import time_familiarity_check, time_lexical_access

# Alias for old checkpoints
sys.modules['model_lstm_v2'] = model_lstm
sys.modules['model'] = model_lstm


# --------------------------------------------------------------------------- #
#  Logger
# --------------------------------------------------------------------------- #

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


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


def compute_real_l1_l2(tokens, predictabilities, subtlex):
    alpha1, alpha2, alpha3 = 104, 3.4, 39
    delta = 0.34
    eccentricity = 1.15
    l1_list, l2_list = [], []
    for token, pred in zip(tokens, predictabilities):
        freq = get_real_frequency(token, subtlex)
        wordlen = len(token)
        tL1 = time_familiarity_check(0, wordlen, freq, pred, eccentricity,
                                      alpha1, alpha2, alpha3)
        tL1 = max(1.0, tL1)
        tL2 = time_lexical_access(freq, pred, delta, alpha1, alpha2, alpha3)
        tL2 = max(1.0, tL2)
        l1_list.append(tL1)
        l2_list.append(tL2)
    return l1_list, l2_list


# --------------------------------------------------------------------------- #
#  Collect per-word predictions from all models
# --------------------------------------------------------------------------- #

def collect_all_predictions(sentences, subtlex, diff_ezr, lstm_model, vocab,
                            bert_model, llama_model, direct_model, device,
                            num_runs=50):
    """Run all models on sentences, return list of per-word dicts."""
    words = []
    total = len(sentences)

    for s_idx, agg in enumerate(sentences):
        if (s_idx + 1) % 25 == 0:
            print(f"  Processing sentence {s_idx+1}/{total}...", flush=True)

        tokens = agg.tokens
        preds = agg.predictabilities
        wlens = [len(t) for t in tokens]

        # Original EZ Reader (stochastic simulation)
        freqs = [get_real_frequency(t, subtlex) for t in tokens]
        orig_result = run_original_simulation_averaged(
            tokens, freqs, preds, num_runs=num_runs)

        # Formula L1/L2
        l1f, l2f = compute_real_l1_l2(tokens, preds, subtlex)

        # Differentiable EZ Reader (formula-based, no neural)
        with torch.no_grad():
            dr = diff_ezr(
                torch.tensor([l1f], dtype=torch.float32, device=device),
                torch.tensor([l2f], dtype=torch.float32, device=device),
                torch.tensor([preds], dtype=torch.float32, device=device),
                torch.tensor([wlens], dtype=torch.float32, device=device),
            )

            # LSTM
            nr = lstm_model(
                vocab.encode_sentence(tokens).unsqueeze(0).to(device),
                torch.tensor([preds], dtype=torch.float32).to(device),
                torch.tensor([wlens], dtype=torch.float32).to(device),
            )

            # BERT
            br = bert_model(
                [tokens],
                torch.tensor([preds], dtype=torch.float32).to(device),
                torch.tensor([wlens], dtype=torch.float32).to(device),
            ) if bert_model else None

            # LLaMA
            lr = llama_model(
                [tokens],
                torch.tensor([preds], dtype=torch.float32).to(device),
                torch.tensor([wlens], dtype=torch.float32).to(device),
            ) if llama_model else None

            # Direct LLaMA (no E-Z Reader)
            dr2 = direct_model(
                [tokens],
                torch.tensor([preds], dtype=torch.float32).to(device),
                torch.tensor([wlens], dtype=torch.float32).to(device),
            ) if direct_model else None

        for i in range(len(tokens)):
            freq = get_real_frequency(tokens[i], subtlex)
            w = {
                'token': tokens[i],
                'freq': freq,
                'log_freq': math.log(max(1, freq)),  # natural log (like E-Z Reader)
                'pred': preds[i],
                'wlen': wlens[i],
                # Human
                'h_ffd': agg.mean_ffd[i],
                'h_gaze': agg.mean_gaze[i],
                'h_trt': agg.mean_trt[i],
                'h_skip': agg.skip_rate[i],
                # Original EZ Reader
                'orig_ffd': orig_result['first_fixation_duration'][i],
                'orig_trt': orig_result['total_reading_time'][i],
                'orig_skip': orig_result['skip_rate'][i],
                # Diff EZ Reader (formula)
                'diff_ffd': dr['first_fixation'][0, i].item(),
                'diff_gaze': dr['gaze_duration'][0, i].item(),
                'diff_trt': dr['total_reading_time'][0, i].item(),
                'diff_skip': dr['skip_prob'][0, i].item(),
                # LSTM
                'lstm_ffd': nr['first_fixation'][0, i].cpu().item(),
                'lstm_gaze': nr['gaze_duration'][0, i].cpu().item(),
                'lstm_trt': nr['total_reading_time'][0, i].cpu().item(),
                'lstm_skip': nr['skip_prob'][0, i].cpu().item(),
                # BERT
                'bert_ffd': br['first_fixation'][0, i].cpu().item() if br else None,
                'bert_gaze': br['gaze_duration'][0, i].cpu().item() if br else None,
                'bert_trt': br['total_reading_time'][0, i].cpu().item() if br else None,
                'bert_skip': br['skip_prob'][0, i].cpu().item() if br else None,
                # LLaMA
                'llama_ffd': lr['first_fixation'][0, i].cpu().item() if lr else None,
                'llama_gaze': lr['gaze_duration'][0, i].cpu().item() if lr else None,
                'llama_trt': lr['total_reading_time'][0, i].cpu().item() if lr else None,
                'llama_skip': lr['skip_prob'][0, i].cpu().item() if lr else None,
                # Direct LLaMA
                'direct_ffd': dr2['first_fixation'][0, i].cpu().item() if dr2 else None,
                'direct_gaze': dr2['gaze_duration'][0, i].cpu().item() if dr2 else None,
                'direct_trt': dr2['total_reading_time'][0, i].cpu().item() if dr2 else None,
                'direct_skip': dr2['skip_prob'][0, i].cpu().item() if dr2 else None,
            }
            words.append(w)

    return words


# --------------------------------------------------------------------------- #
#  Binning: quintile bins (5 bins like Schilling et al.)
# --------------------------------------------------------------------------- #

def make_quintile_bins(words, key, bin_labels=None):
    """Bin words into 5 equal-sized groups by a numeric key."""
    sorted_vals = sorted(set(w[key] for w in words))
    vals = [w[key] for w in words]
    percentiles = [np.percentile(vals, p) for p in [20, 40, 60, 80]]

    bins = {i: [] for i in range(5)}
    for w in words:
        v = w[key]
        if v <= percentiles[0]:
            bins[0].append(w)
        elif v <= percentiles[1]:
            bins[1].append(w)
        elif v <= percentiles[2]:
            bins[2].append(w)
        elif v <= percentiles[3]:
            bins[3].append(w)
        else:
            bins[4].append(w)

    if bin_labels is None:
        bin_labels = [f"Bin {i+1}" for i in range(5)]

    return {bin_labels[i]: bins[i] for i in range(5)}


def bin_by_frequency(words):
    """5 frequency bins (lowest to highest)."""
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    return make_quintile_bins(words, 'log_freq', labels)


def bin_by_predictability(words):
    """5 predictability bins."""
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    return make_quintile_bins(words, 'pred', labels)


def bin_by_word_length(words):
    """Bin by word length: 1-2, 3, 4-5, 6-7, 8+."""
    bins = {
        "1-2 chars": [],
        "3 chars": [],
        "4-5 chars": [],
        "6-7 chars": [],
        "8+ chars": [],
    }
    for w in words:
        wl = w['wlen']
        if wl <= 2:    bins["1-2 chars"].append(w)
        elif wl == 3:  bins["3 chars"].append(w)
        elif wl <= 5:  bins["4-5 chars"].append(w)
        elif wl <= 7:  bins["6-7 chars"].append(w)
        else:          bins["8+ chars"].append(w)
    return bins


# --------------------------------------------------------------------------- #
#  Compute bin-level means and RMSD
# --------------------------------------------------------------------------- #

MODEL_KEYS = ['orig', 'diff', 'lstm', 'bert', 'llama', 'direct']
MODEL_NAMES = ['Orig EZ', 'Diff EZ', 'LSTM+Diff', 'BERT+Diff', 'LLaMA+Diff', 'Direct LLaMA']

MEASURES = {
    'ffd': 'First Fixation Duration',
    'trt': 'Total Reading Time',
    'skip': 'Skip Probability',
    'gaze': 'Gaze Duration',
}


def bin_mean(word_list, prefix, measure):
    """Compute mean of a measure for a bin of words."""
    key = f'{prefix}_{measure}'
    vals = [w[key] for w in word_list if w.get(key) is not None]
    return np.mean(vals) if vals else None


def compute_bin_level_metrics(bins, measure, active_models):
    """
    For each model, compute:
      - bin means (for the table)
      - RMSD across bins vs human bin means
      - Pearson r across bins vs human bin means
    """
    bin_names = list(bins.keys())
    n_bins = len(bin_names)

    # Human bin means
    h_means = []
    for bname in bin_names:
        h_means.append(bin_mean(bins[bname], 'h', measure))
    h_means = np.array(h_means, dtype=float)

    results = {}
    for mk in active_models:
        m_means = []
        for bname in bin_names:
            m_means.append(bin_mean(bins[bname], mk, measure))

        # Handle missing values
        if any(v is None for v in m_means):
            results[mk] = {'means': [None]*n_bins, 'rmsd': None, 'r': None}
            continue

        m_means = np.array(m_means, dtype=float)

        # RMSD across bins
        rmsd = np.sqrt(np.mean((h_means - m_means) ** 2))

        # Correlation across bins
        if np.std(m_means) > 0 and np.std(h_means) > 0 and n_bins >= 3:
            r = np.corrcoef(h_means, m_means)[0, 1]
        else:
            r = 0.0

        results[mk] = {'means': m_means.tolist(), 'rmsd': rmsd, 'r': r}

    results['human'] = {'means': h_means.tolist()}
    return results


# --------------------------------------------------------------------------- #
#  Word-level metrics (for comparison)
# --------------------------------------------------------------------------- #

def word_level_corr(words, model_key, measure):
    """Pearson r at word level."""
    h_key = f'h_{measure}'
    m_key = f'{model_key}_{measure}'
    h_vals = [w[h_key] for w in words if w.get(m_key) is not None]
    m_vals = [w[m_key] for w in words if w.get(m_key) is not None]
    if len(h_vals) < 3:
        return 0.0
    h_arr, m_arr = np.array(h_vals), np.array(m_vals)
    if np.std(h_arr) > 0 and np.std(m_arr) > 0:
        return np.corrcoef(h_arr, m_arr)[0, 1]
    return 0.0


# --------------------------------------------------------------------------- #
#  Print bin-level table
# --------------------------------------------------------------------------- #

def print_bin_table(title, bins, measure, measure_label, active_models, active_names):
    """Print a formatted table of bin means + RMSD + r across bins."""
    metrics = compute_bin_level_metrics(bins, measure, active_models)
    bin_names = list(bins.keys())

    print(f"\n  {title}: {measure_label}")
    print(f"  {'-' * 110}")

    # Header
    header = f"  {'Bin':<16} {'N':>6} {'Human':>10}"
    for name in active_names:
        header += f" {name:>11}"
    print(header)
    print(f"  {'-' * 110}")

    # Rows
    for i, bname in enumerate(bin_names):
        n = len(bins[bname])
        h_val = metrics['human']['means'][i]
        row = f"  {bname:<16} {n:>6} {h_val:>10.1f}" if measure != 'skip' else \
              f"  {bname:<16} {n:>6} {h_val:>10.3f}"

        for mk in active_models:
            val = metrics[mk]['means'][i]
            if val is None:
                row += f" {'N/A':>11}"
            elif measure == 'skip':
                row += f" {val:>11.3f}"
            else:
                row += f" {val:>11.1f}"
        print(row)

    # RMSD row
    print(f"  {'-' * 110}")
    rmsd_row = f"  {'RMSD':.<16} {'':>6} {'':>10}"
    for mk in active_models:
        r = metrics[mk].get('rmsd')
        if r is None:
            rmsd_row += f" {'N/A':>11}"
        elif measure == 'skip':
            rmsd_row += f" {r:>11.3f}"
        else:
            rmsd_row += f" {r:>11.1f}"
    print(rmsd_row)

    # Correlation across bins
    corr_row = f"  {'r (bins)':.<16} {'':>6} {'':>10}"
    for mk in active_models:
        r = metrics[mk].get('r')
        if r is None:
            corr_row += f" {'N/A':>11}"
        else:
            corr_row += f" {r:>11.3f}"
    print(corr_row)

    return metrics


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-runs", type=int, default=50,
                        help="Number of Monte Carlo runs for orig EZ Reader (default: 50)")
    parser.add_argument("--corpus", type=str, default="both",
                        choices=["provo", "geco", "both"],
                        help="Which corpus to evaluate on")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'v2')
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load frequency data ----
    print("Loading SUBTLEXus...")
    subtlex = load_subtlexus(os.path.join(data_dir, 'SUBTLEXus.txt'))
    print(f"  {len(subtlex):,} entries")

    # ---- Load models ----
    print("\nLoading GECO-trained models...")

    # LSTM
    lstm_ckpt = torch.load(os.path.join(ckpt_dir, "geco_lstm", "best_model_lstm.pt"),
                           map_location=device, weights_only=False)
    vocab = lstm_ckpt['vocab']
    lstm_model = NeuralEZReader(vocab_size=len(vocab)).to(device)
    lstm_model.load_state_dict(lstm_ckpt['model_state_dict'], strict=False)
    lstm_model.eval()
    print(f"  LSTM: loaded (epoch {lstm_ckpt['epoch']})")

    # BERT
    bert_model = None
    bert_path = os.path.join(ckpt_dir, "geco_bert", "best_model_bert.pt")
    if os.path.exists(bert_path):
        bert_ckpt = torch.load(bert_path, map_location=device, weights_only=False)
        bert_name = bert_ckpt.get('bert_model_name', 'bert-base-uncased')
        freeze_layers = bert_ckpt.get('freeze_bert_layers', 8)
        bert_model = NeuralEZReaderBERT(
            bert_model_name=bert_name,
            freeze_bert_layers=freeze_layers,
        ).to(device)
        bert_model.load_state_dict(bert_ckpt['model_state_dict'], strict=False)
        bert_model.eval()
        print(f"  BERT: loaded (epoch {bert_ckpt['epoch']})")

    # LLaMA
    llama_model = None
    # Try to find a llama checkpoint
    for subdir in os.listdir(ckpt_dir):
        if subdir.startswith("geco_") and "llama" in subdir.lower():
            llama_path = os.path.join(ckpt_dir, subdir, "best_model.pt")
            if os.path.exists(llama_path):
                llama_ckpt = torch.load(llama_path, map_location=device, weights_only=False)
                llama_name = llama_ckpt.get('model_name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
                freeze_layers_llama = llama_ckpt.get('freeze_layers', 16)
                llama_model = NeuralEZReaderLLaMA(
                    model_name=llama_name,
                    freeze_layers=freeze_layers_llama,
                    hidden_dim=256,
                ).to(device)
                llama_model.load_state_dict(llama_ckpt['model_state_dict'])
                llama_model.eval()
                print(f"  LLaMA: loaded from {subdir} (epoch {llama_ckpt['epoch']})")
                break

    # Direct LLaMA (no E-Z Reader)
    direct_model = None
    for subdir in os.listdir(ckpt_dir):
        if subdir.startswith("geco_direct"):
            direct_path = os.path.join(ckpt_dir, subdir, "best_model.pt")
            if os.path.exists(direct_path):
                direct_ckpt = torch.load(direct_path, map_location=device, weights_only=False)
                direct_name = direct_ckpt.get('model_name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
                freeze_layers_direct = direct_ckpt.get('freeze_layers', 16)
                direct_model = DirectRegressionLLaMA(
                    model_name=direct_name,
                    freeze_layers=freeze_layers_direct,
                    hidden_dim=256,
                ).to(device)
                direct_model.load_state_dict(direct_ckpt['model_state_dict'])
                direct_model.eval()
                print(f"  Direct: loaded from {subdir} (epoch {direct_ckpt['epoch']})")
                break

    # Diff EZ Reader (untrained formula baseline)
    diff_ezr = DifferentiableEZReader().to(device)
    diff_ezr.eval()
    print("  Diff EZ: loaded (untrained formula)")

    # ---- Determine active models ----
    active_models = ['orig', 'diff', 'lstm']
    active_names = ['Orig EZ', 'Diff EZ', 'LSTM+Diff']
    if bert_model:
        active_models.append('bert')
        active_names.append('BERT+Diff')
    if llama_model:
        active_models.append('llama')
        active_names.append('LLaMA+Diff')
    if direct_model:
        active_models.append('direct')
        active_names.append('Direct LLaMA')

    # Redirect stdout
    sys.stdout = Logger(os.path.join(results_dir, "eval_binned_results.txt"))

    def run_evaluation(sentences, corpus_name):
        """Run full bin-level + word-level evaluation on a set of sentences."""
        print(f"\n{'#' * 100}")
        print(f"#  {corpus_name}")
        print(f"#  Orig EZ Reader: {args.num_runs} Monte Carlo runs per sentence")
        print(f"{'#' * 100}")

        # Collect predictions
        t0 = time.time()
        words = collect_all_predictions(
            sentences, subtlex, diff_ezr, lstm_model, vocab,
            bert_model, llama_model, direct_model, device,
            num_runs=args.num_runs)
        elapsed = time.time() - t0
        print(f"\n  Collected {len(words)} word predictions in {elapsed:.1f}s")

        # ================================================================== #
        #  PART 1: BIN-LEVEL EVALUATION (E-Z Reader style)
        # ================================================================== #
        print(f"\n{'=' * 110}")
        print(f"  PART 1: BIN-LEVEL EVALUATION (E-Z Reader style)")
        print(f"  Bin words, compute mean per bin, measure RMSD and r across bins")
        print(f"{'=' * 110}")

        # --- Frequency bins (5 quintiles) ---
        freq_bins = bin_by_frequency(words)
        print(f"\n{'=' * 110}")
        print(f"  FREQUENCY BINS (5 quintiles by log frequency)")
        print(f"{'=' * 110}")
        for measure, label in [('ffd', 'FFD (ms)'), ('trt', 'TRT (ms)'),
                                ('gaze', 'Gaze Duration (ms)'), ('skip', 'Skip Rate')]:
            # Skip gaze for orig (not available from simulation)
            if measure == 'gaze':
                # Orig doesn't have gaze, but print anyway (shows N/A)
                pass
            print_bin_table("FREQUENCY", freq_bins, measure, label,
                           active_models, active_names)

        # --- Predictability bins (5 quintiles) ---
        pred_bins = bin_by_predictability(words)
        print(f"\n{'=' * 110}")
        print(f"  PREDICTABILITY BINS (5 quintiles by cloze predictability)")
        print(f"{'=' * 110}")
        for measure, label in [('ffd', 'FFD (ms)'), ('trt', 'TRT (ms)'),
                                ('gaze', 'Gaze Duration (ms)'), ('skip', 'Skip Rate')]:
            print_bin_table("PREDICTABILITY", pred_bins, measure, label,
                           active_models, active_names)

        # --- Word length bins ---
        len_bins = bin_by_word_length(words)
        print(f"\n{'=' * 110}")
        print(f"  WORD LENGTH BINS")
        print(f"{'=' * 110}")
        for measure, label in [('ffd', 'FFD (ms)'), ('trt', 'TRT (ms)'),
                                ('gaze', 'Gaze Duration (ms)'), ('skip', 'Skip Rate')]:
            print_bin_table("WORD LENGTH", len_bins, measure, label,
                           active_models, active_names)

        # ================================================================== #
        #  PART 2: SUMMARY — RMSD across all bin analyses
        # ================================================================== #
        print(f"\n{'=' * 110}")
        print(f"  PART 2: SUMMARY — RMSD ACROSS BINS (lower is better)")
        print(f"{'=' * 110}")

        all_binnings = [
            ("Freq bins", freq_bins),
            ("Pred bins", pred_bins),
            ("Length bins", len_bins),
        ]

        measures_for_summary = [
            ('ffd', 'FFD'),
            ('trt', 'TRT'),
            ('skip', 'Skip'),
        ]

        header = f"  {'Binning + Measure':<28}"
        for name in active_names:
            header += f" {name:>11}"
        print(header)
        print(f"  {'-' * 100}")

        # Collect all RMSDs for grand summary
        model_rmsds = {mk: [] for mk in active_models}

        for bin_label, bins in all_binnings:
            for measure, m_label in measures_for_summary:
                metrics = compute_bin_level_metrics(bins, measure, active_models)
                row = f"  {bin_label + ' / ' + m_label:<28}"
                for mk in active_models:
                    rmsd_val = metrics[mk].get('rmsd')
                    if rmsd_val is None:
                        row += f" {'N/A':>11}"
                    elif measure == 'skip':
                        row += f" {rmsd_val:>11.3f}"
                        model_rmsds[mk].append(rmsd_val)
                    else:
                        row += f" {rmsd_val:>11.1f}"
                        model_rmsds[mk].append(rmsd_val)
                print(row)

        # ================================================================== #
        #  PART 3: CORRELATION ACROSS BINS (higher is better)
        # ================================================================== #
        print(f"\n{'=' * 110}")
        print(f"  PART 3: CORRELATION ACROSS BINS (higher is better)")
        print(f"{'=' * 110}")

        header = f"  {'Binning + Measure':<28}"
        for name in active_names:
            header += f" {name:>11}"
        print(header)
        print(f"  {'-' * 100}")

        model_bin_corrs = {mk: [] for mk in active_models}

        for bin_label, bins in all_binnings:
            for measure, m_label in measures_for_summary:
                metrics = compute_bin_level_metrics(bins, measure, active_models)
                row = f"  {bin_label + ' / ' + m_label:<28}"
                for mk in active_models:
                    r_val = metrics[mk].get('r')
                    if r_val is None:
                        row += f" {'N/A':>11}"
                    else:
                        row += f" {r_val:>11.3f}"
                        model_bin_corrs[mk].append(r_val)
                print(row)

        # Mean bin correlation
        print(f"  {'-' * 100}")
        row = f"  {'MEAN':.<28}"
        for mk in active_models:
            vals = model_bin_corrs[mk]
            if vals:
                row += f" {np.mean(vals):>11.3f}"
            else:
                row += f" {'N/A':>11}"
        print(row)

        # ================================================================== #
        #  PART 4: WORD-LEVEL CORRELATIONS (for comparison)
        # ================================================================== #
        print(f"\n{'=' * 110}")
        print(f"  PART 4: WORD-LEVEL CORRELATIONS (for comparison)")
        print(f"  (This is how we evaluated before — E-Z Reader wasn't designed for this)")
        print(f"{'=' * 110}")

        header = f"  {'Measure':<28}"
        for name in active_names:
            header += f" {name:>11}"
        print(header)
        print(f"  {'-' * 100}")

        for measure, label in [('ffd', 'FFD'), ('trt', 'TRT'), ('skip', 'Skip Rate')]:
            row = f"  {label:<28}"
            for mk in active_models:
                r = word_level_corr(words, mk, measure)
                row += f" {r:>11.3f}"
            print(row)

        # ================================================================== #
        #  FINAL COMPARISON TABLE
        # ================================================================== #
        print(f"\n{'=' * 110}")
        print(f"  FINAL COMPARISON: BIN-LEVEL r (mean) vs WORD-LEVEL r")
        print(f"{'=' * 110}")
        header = f"  {'Model':<20} {'Bin r (mean)':>14} {'Word r_TRT':>14} {'Word r_FFD':>14} {'Word r_Skip':>14}"
        print(header)
        print(f"  {'-' * 80}")
        for mk, mname in zip(active_models, active_names):
            bin_r_mean = np.mean(model_bin_corrs[mk]) if model_bin_corrs[mk] else 0.0
            word_r_trt = word_level_corr(words, mk, 'trt')
            word_r_ffd = word_level_corr(words, mk, 'ffd')
            word_r_skip = word_level_corr(words, mk, 'skip')
            print(f"  {mname:<20} {bin_r_mean:>14.3f} {word_r_trt:>14.3f} {word_r_ffd:>14.3f} {word_r_skip:>14.3f}")

    # ---- Run evaluations ----
    if args.corpus in ("provo", "both"):
        print("\nLoading Provo Corpus...")
        et_path = os.path.join(data_dir, "Provo_Corpus-Eyetracking_Data.csv")
        provo_raw = load_provo(et_path)
        provo_agg = aggregate_by_sentence(provo_raw, min_participants=10)
        print(f"  Provo: {len(provo_agg)} sentences, "
              f"{sum(len(a.tokens) for a in provo_agg)} words")
        run_evaluation(provo_agg, "FULL PROVO CORPUS (cross-corpus: models trained on GECO)")

    if args.corpus in ("geco", "both"):
        print("\nLoading GECO Corpus (test set)...")
        reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
        material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
        pred_path = os.path.join(data_dir, "geco_predictability.pkl")
        geco_raw = load_geco(reading_path, material_path, pred_path)
        _, _, test_raw = split_geco(geco_raw)
        geco_agg = aggregate_by_sentence(geco_raw, min_participants=5)
        test_ids = set(sd.text_id for sd in test_raw)
        geco_test_agg = [a for a in geco_agg if a.text_id in test_ids]
        print(f"  GECO test: {len(geco_test_agg)} sentences, "
              f"{sum(len(a.tokens) for a in geco_test_agg)} words")
        run_evaluation(geco_test_agg, "GECO TEST SET (in-distribution)")

    print(f"\n\nDone! Results saved to: {os.path.join(results_dir, 'eval_binned_results.txt')}")


if __name__ == "__main__":
    main()
