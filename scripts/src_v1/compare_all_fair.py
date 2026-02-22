"""
FAIR 4-Way Comparison using REAL word features.

Models:
  1. Original EZ Reader     - REAL freq (SUBTLEXus) + cloze pred + discrete sim
  2. Differentiable EZ Reader - REAL freq + cloze pred + smooth approximation
  3. Neural EZ Reader (LSTM) - LSTM (no freq/pred needed) + smooth approximation
  4. Neural EZ Reader (BERT) - BERT (no freq/pred needed) + smooth approximation

Evaluation is done on the HELD-OUT TEST SET ONLY (same split as training,
seed=42, split by text_id) to avoid reporting inflated numbers from
evaluating on training data.
"""

import os
import sys
import csv
import math
import time
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ez_reader'))

from diff_ezreader import DifferentiableEZReader
from model_lstm import NeuralEZReader, Vocabulary
from model_bert import NeuralEZReaderBERT
from data_loader import load_provo, aggregate_by_sentence, split_aggregated
from ez_wrapper import run_original_simulation_averaged
from utilities import time_familiarity_check, time_lexical_access


# --------------------------------------------------------------------------- #
#  Load REAL word frequencies
# --------------------------------------------------------------------------- #

def load_subtlexus(path):
    """Load SUBTLEXus: word_lower -> raw_count."""
    freq = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            freq[row['Word'].lower()] = int(row['FREQcount'])
    return freq


def get_real_frequency(word, subtlex):
    """
    Get word frequency from SUBTLEXus.
    Handles contractions and missing words with sensible fallbacks.
    """
    w = word.lower().strip(".,;:!?\"'()[]{}").replace("\u2019", "'")
    if w in subtlex:
        return max(1, subtlex[w])

    # Try without apostrophe contractions
    for variant in [w.replace("'", ""), w.split("'")[0], w.split("-")[0]]:
        if variant in subtlex:
            return max(1, subtlex[variant])

    # Fallback: estimate from word length (conservative)
    length = len(w)
    if length <= 3:   return 50000
    elif length <= 5: return 10000
    elif length <= 7: return 2000
    else:             return 500


# --------------------------------------------------------------------------- #
#  Compute formula L1/L2 with REAL features
# --------------------------------------------------------------------------- #

def compute_real_l1_l2(tokens, predictabilities, subtlex):
    """
    Compute L1/L2 using original EZ Reader formulas with REAL frequencies.
    """
    alpha1, alpha2, alpha3 = 104, 3.4, 39
    delta = 0.34
    eccentricity = 1.15

    l1_list, l2_list = [], []
    for token, pred in zip(tokens, predictabilities):
        freq = get_real_frequency(token, subtlex)
        wordlen = len(token)

        # L1 (distance=0 for simplicity, same as before)
        tL1 = time_familiarity_check(0, wordlen, freq, pred, eccentricity,
                                      alpha1, alpha2, alpha3)
        tL1 = max(1.0, tL1)

        # L2
        tL2 = time_lexical_access(freq, pred, delta, alpha1, alpha2, alpha3)
        tL2 = max(1.0, tL2)

        l1_list.append(tL1)
        l2_list.append(tL2)

    return l1_list, l2_list


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def corr(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
        return np.corrcoef(a, b)[0, 1]
    return 0.0

def mae(a, b):
    return np.mean(np.abs(np.array(a) - np.array(b)))

def rmse(a, b):
    return np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2))


# --------------------------------------------------------------------------- #
#  Logger
# --------------------------------------------------------------------------- #

class Logger(object):
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
#  Run all models on a set of sentences, return collected predictions
# --------------------------------------------------------------------------- #

def run_all_models(sentences, subtlex, diff_ezr, lstm_model, vocab,
                   bert_model, device):
    """Run all 4 models on the given sentences and return per-word predictions."""
    h_trt, h_ffd, h_skip = [], [], []
    orig_trt, orig_ffd = [], []
    diff_trt, diff_ffd, diff_skip = [], [], []
    lstm_trt, lstm_ffd, lstm_skip = [], [], []
    bert_trt, bert_ffd, bert_skip = [], [], []
    lstm_l1, lstm_l2 = [], []
    bert_l1, bert_l2 = [], []
    formula_l1_all, formula_l2_all = [], []

    for agg in sentences:
        tokens = agg.tokens
        preds = agg.predictabilities
        wlens = [len(t) for t in tokens]

        h_trt.extend(agg.mean_trt)
        h_ffd.extend(agg.mean_ffd)
        h_skip.extend(agg.skip_rate)

        # Formula-based L1/L2
        l1f, l2f = compute_real_l1_l2(tokens, preds, subtlex)
        formula_l1_all.extend(l1f)
        formula_l2_all.extend(l2f)

        # Model 1: Real original EZ Reader (computes L1/L2 internally with proper distance)
        freqs = [get_real_frequency(t, subtlex) for t in tokens]
        orig_result = run_original_simulation_averaged(
            tokens, freqs, preds, num_runs=20)
        orig_trt.extend(orig_result['total_reading_time'])
        orig_ffd.extend(orig_result['first_fixation_duration'])

        # Model 2: Differentiable
        with torch.no_grad():
            dr = diff_ezr(
                torch.tensor([l1f], dtype=torch.float32),
                torch.tensor([l2f], dtype=torch.float32),
                torch.tensor([preds], dtype=torch.float32),
                torch.tensor([wlens], dtype=torch.float32),
            )
        diff_trt.extend(dr['total_reading_time'][0].tolist())
        diff_ffd.extend(dr['first_fixation'][0].tolist())
        diff_skip.extend(dr['skip_prob'][0].tolist())

        # Model 3: LSTM
        with torch.no_grad():
            nr = lstm_model(
                vocab.encode_sentence(tokens).unsqueeze(0).to(device),
                torch.tensor([preds], dtype=torch.float32).to(device),
                torch.tensor([wlens], dtype=torch.float32).to(device),
            )
        lstm_trt.extend(nr['total_reading_time'][0].cpu().tolist())
        lstm_ffd.extend(nr['first_fixation'][0].cpu().tolist())
        lstm_skip.extend(nr['skip_prob'][0].cpu().tolist())
        lstm_l1.extend(nr['L1'][0].cpu().tolist())
        lstm_l2.extend(nr['L2'][0].cpu().tolist())

        # Model 4: BERT
        if bert_model:
            with torch.no_grad():
                br = bert_model(
                    [tokens],
                    torch.tensor([preds], dtype=torch.float32).to(device),
                    torch.tensor([wlens], dtype=torch.float32).to(device),
                )
            bert_trt.extend(br['total_reading_time'][0].cpu().tolist())
            bert_ffd.extend(br['first_fixation'][0].cpu().tolist())
            bert_skip.extend(br['skip_prob'][0].cpu().tolist())
            bert_l1.extend(br['L1'][0].cpu().tolist())
            bert_l2.extend(br['L2'][0].cpu().tolist())

    return {
        'h_trt': np.array(h_trt), 'h_ffd': np.array(h_ffd), 'h_skip': np.array(h_skip),
        'orig_trt': np.array(orig_trt), 'orig_ffd': np.array(orig_ffd),
        'diff_trt': np.array(diff_trt), 'diff_ffd': np.array(diff_ffd),
        'diff_skip': np.array(diff_skip),
        'lstm_trt': np.array(lstm_trt), 'lstm_ffd': np.array(lstm_ffd),
        'lstm_skip': np.array(lstm_skip),
        'lstm_l1': np.array(lstm_l1), 'lstm_l2': np.array(lstm_l2),
        'bert_trt': np.array(bert_trt) if bert_model else np.array([]),
        'bert_ffd': np.array(bert_ffd) if bert_model else np.array([]),
        'bert_skip': np.array(bert_skip) if bert_model else np.array([]),
        'bert_l1': np.array(bert_l1) if bert_model else np.array([]),
        'bert_l2': np.array(bert_l2) if bert_model else np.array([]),
        'formula_l1': np.array(formula_l1_all), 'formula_l2': np.array(formula_l2_all),
    }


# --------------------------------------------------------------------------- #
#  Print results table for a given split
# --------------------------------------------------------------------------- #

def print_results(r, split_name, n_sentences, subtlex, bert_model,
                  diff_ezr, lstm_model, vocab, sentences_for_samples=None):
    """Print the full results table for one split."""
    W = 100
    n = len(r['h_trt'])

    print(f"\n{'=' * W}")
    print(f"   {split_name}")
    print(f"{'=' * W}")
    print(f"  Words: {n} | Sentences: {n_sentences}")
    print(f"  Frequency source: SUBTLEXus ({len(subtlex):,} words)")
    print(f"  Predictability: Provo cloze norms (OrthographicMatch)")

    # ---- Correlation ----
    has_bert = bert_model is not None and len(r['bert_trt']) > 0

    print(f"\n{'=' * W}")
    print("  CORRELATION WITH HUMAN DATA (Pearson r)")
    print(f"{'=' * W}")
    print(f"  {'Metric':<28} {'Orig EZ':>10} {'Diff EZ':>10} {'LSTM+Diff':>10} {'BERT+Diff':>10}")
    print(f"  {'-'*80}")

    bert_trt_corr = corr(r['bert_trt'], r['h_trt']) if has_bert else 0.0
    bert_ffd_corr = corr(r['bert_ffd'], r['h_ffd']) if has_bert else 0.0
    bert_skip_corr = corr(r['bert_skip'], r['h_skip']) if has_bert else 0.0

    print(f"  {'Total Reading Time (TRT)':<28} {corr(r['orig_trt'], r['h_trt']):>10.3f} {corr(r['diff_trt'], r['h_trt']):>10.3f} {corr(r['lstm_trt'], r['h_trt']):>10.3f} {bert_trt_corr:>10.3f}")
    print(f"  {'First Fixation Dur. (FFD)':<28} {corr(r['orig_ffd'], r['h_ffd']):>10.3f} {corr(r['diff_ffd'], r['h_ffd']):>10.3f} {corr(r['lstm_ffd'], r['h_ffd']):>10.3f} {bert_ffd_corr:>10.3f}")
    print(f"  {'Skip Rate':<28} {'N/A':>10} {corr(r['diff_skip'], r['h_skip']):>10.3f} {corr(r['lstm_skip'], r['h_skip']):>10.3f} {bert_skip_corr:>10.3f}")

    # ---- Error ----
    print(f"\n{'=' * W}")
    print("  ERROR METRICS")
    print(f"{'=' * W}")
    print(f"  {'Metric':<28} {'Orig EZ':>10} {'Diff EZ':>10} {'LSTM+Diff':>10} {'BERT+Diff':>10}")
    print(f"  {'-'*80}")

    bert_trt_mae = mae(r['bert_trt'], r['h_trt']) if has_bert else 0.0
    bert_ffd_mae = mae(r['bert_ffd'], r['h_ffd']) if has_bert else 0.0
    bert_trt_rmse = rmse(r['bert_trt'], r['h_trt']) if has_bert else 0.0
    bert_ffd_rmse = rmse(r['bert_ffd'], r['h_ffd']) if has_bert else 0.0

    print(f"  {'MAE TRT (ms)':<28} {mae(r['orig_trt'], r['h_trt']):>10.1f} {mae(r['diff_trt'], r['h_trt']):>10.1f} {mae(r['lstm_trt'], r['h_trt']):>10.1f} {bert_trt_mae:>10.1f}")
    print(f"  {'MAE FFD (ms)':<28} {mae(r['orig_ffd'], r['h_ffd']):>10.1f} {mae(r['diff_ffd'], r['h_ffd']):>10.1f} {mae(r['lstm_ffd'], r['h_ffd']):>10.1f} {bert_ffd_mae:>10.1f}")
    print(f"  {'RMSE TRT (ms)':<28} {rmse(r['orig_trt'], r['h_trt']):>10.1f} {rmse(r['diff_trt'], r['h_trt']):>10.1f} {rmse(r['lstm_trt'], r['h_trt']):>10.1f} {bert_trt_rmse:>10.1f}")
    print(f"  {'RMSE FFD (ms)':<28} {rmse(r['orig_ffd'], r['h_ffd']):>10.1f} {rmse(r['diff_ffd'], r['h_ffd']):>10.1f} {rmse(r['lstm_ffd'], r['h_ffd']):>10.1f} {bert_ffd_rmse:>10.1f}")

    # ---- Means ----
    print(f"\n{'=' * W}")
    print("  MEAN PREDICTIONS (ms)")
    print(f"{'=' * W}")
    print(f"  {'Metric':<28} {'Human':>10} {'Orig EZ':>10} {'Diff EZ':>10} {'LSTM+Diff':>10} {'BERT+Diff':>10}")
    print(f"  {'-'*90}")

    bert_trt_mean = np.mean(r['bert_trt']) if has_bert else 0.0
    bert_ffd_mean = np.mean(r['bert_ffd']) if has_bert else 0.0
    bert_trt_std = np.std(r['bert_trt']) if has_bert else 0.0
    bert_skip_mean = np.mean(r['bert_skip']) if has_bert else 0.0

    print(f"  {'Mean TRT':<28} {np.mean(r['h_trt']):>10.1f} {np.mean(r['orig_trt']):>10.1f} {np.mean(r['diff_trt']):>10.1f} {np.mean(r['lstm_trt']):>10.1f} {bert_trt_mean:>10.1f}")
    print(f"  {'Mean FFD':<28} {np.mean(r['h_ffd']):>10.1f} {np.mean(r['orig_ffd']):>10.1f} {np.mean(r['diff_ffd']):>10.1f} {np.mean(r['lstm_ffd']):>10.1f} {bert_ffd_mean:>10.1f}")
    print(f"  {'Std TRT':<28} {np.std(r['h_trt']):>10.1f} {np.std(r['orig_trt']):>10.1f} {np.std(r['diff_trt']):>10.1f} {np.std(r['lstm_trt']):>10.1f} {bert_trt_std:>10.1f}")
    print(f"  {'Mean Skip Rate':<28} {np.mean(r['h_skip']):>10.3f} {'N/A':>10} {np.mean(r['diff_skip']):>10.3f} {np.mean(r['lstm_skip']):>10.3f} {bert_skip_mean:>10.3f}")

    # ---- L1/L2 stats ----
    fl1, fl2 = r['formula_l1'], r['formula_l2']
    ll1, ll2 = r['lstm_l1'], r['lstm_l2']

    print(f"\n{'=' * W}")
    print("  L1/L2 STATISTICS")
    print(f"{'=' * W}")
    print(f"  {'':20} {'Formula':>12} {'LSTM':>12} {'BERT':>12}")
    print(f"  {'-'*58}")
    print(f"  {'L1 mean (ms)':<20} {np.mean(fl1):>12.0f} {np.mean(ll1):>12.0f} {np.mean(r['bert_l1']) if has_bert else 0.0:>12.0f}")
    print(f"  {'L1 std (ms)':<20} {np.std(fl1):>12.0f} {np.std(ll1):>12.0f} {np.std(r['bert_l1']) if has_bert else 0.0:>12.0f}")
    print(f"  {'L2 mean (ms)':<20} {np.mean(fl2):>12.0f} {np.mean(ll2):>12.0f} {np.mean(r['bert_l2']) if has_bert else 0.0:>12.0f}")
    print(f"  {'L2 std (ms)':<20} {np.std(fl2):>12.0f} {np.std(ll2):>12.0f} {np.std(r['bert_l2']) if has_bert else 0.0:>12.0f}")

    # ---- Sample predictions (if sentences provided) ----
    if sentences_for_samples:
        device = torch.device('cpu')
        print(f"\n{'=' * W}")
        print("  SAMPLE PER-WORD PREDICTIONS")
        print(f"{'=' * W}")

        for s_idx, s in enumerate(sentences_for_samples[:4]):
            title = ' '.join(s.tokens[:6]) + ('...' if len(s.tokens) > 6 else '')
            print(f"\n  Sentence (text {s.text_id}): \"{title}\"")
            print(f"  {'word':<14} {'freq':>8} {'pred':>5} | {'hTRT':>5} {'oTRT':>5} {'dTRT':>5} {'nTRT':>5} {'bTRT':>5} | "
                  f"{'hSkip':>5} {'nSkip':>5} {'bSkip':>5}")
            print(f"  {'-'*95}")

            l1f, l2f = compute_real_l1_l2(s.tokens, s.predictabilities, subtlex)
            freqs = [get_real_frequency(t, subtlex) for t in s.tokens]
            o = run_original_simulation_averaged(
                s.tokens, freqs, s.predictabilities, num_runs=20)
            with torch.no_grad():
                dr2 = diff_ezr(
                    torch.tensor([l1f], dtype=torch.float32),
                    torch.tensor([l2f], dtype=torch.float32),
                    torch.tensor([s.predictabilities], dtype=torch.float32),
                    torch.tensor([[len(t) for t in s.tokens]], dtype=torch.float32),
                )
                nr2 = lstm_model(
                    vocab.encode_sentence(s.tokens).unsqueeze(0).to(device),
                    torch.tensor([s.predictabilities], dtype=torch.float32).to(device),
                    torch.tensor([[len(t) for t in s.tokens]], dtype=torch.float32).to(device),
                )
                if bert_model:
                    br2 = bert_model(
                        [s.tokens],
                        torch.tensor([s.predictabilities], dtype=torch.float32).to(device),
                        torch.tensor([[len(t) for t in s.tokens]], dtype=torch.float32).to(device),
                    )

            for i in range(min(10, len(s.tokens))):
                freq = get_real_frequency(s.tokens[i], subtlex)
                pred = s.predictabilities[i]
                ht, hf, hs = s.mean_trt[i], s.mean_ffd[i], s.skip_rate[i]
                ot = o['total_reading_time'][i]
                dt = dr2['total_reading_time'][0, i].item()
                nt = nr2['total_reading_time'][0, i].item()
                ns = nr2['skip_prob'][0, i].item()

                bt = br2['total_reading_time'][0, i].item() if bert_model else 0.0
                bs = br2['skip_prob'][0, i].item() if bert_model else 0.0

                print(
                    f"  {s.tokens[i]:<14} {freq:>8,} {pred:>5.2f} | "
                    f"{ht:5.0f} {ot:5.0f} {dt:5.0f} {nt:5.0f} {bt:5.0f} | "
                    f"{hs:5.2f} {ns:5.2f} {bs:5.2f}"
                )

    # ---- Final summary ----
    print(f"\n{'=' * W}")
    print("  FINAL SUMMARY")
    print(f"{'=' * W}")
    print(f"  {'Model':<35} {'r_TRT':>8} {'r_FFD':>8} {'MAE_TRT':>10}")
    print(f"  {'-'*65}")
    print(f"  {'1. Original EZ Reader (real freq)':<35} {corr(r['orig_trt'], r['h_trt']):>8.3f} {corr(r['orig_ffd'], r['h_ffd']):>8.3f} {mae(r['orig_trt'], r['h_trt']):>9.1f}ms")
    print(f"  {'2. Diff EZ Reader (real freq)':<35} {corr(r['diff_trt'], r['h_trt']):>8.3f} {corr(r['diff_ffd'], r['h_ffd']):>8.3f} {mae(r['diff_trt'], r['h_trt']):>9.1f}ms")
    print(f"  {'3. Neural EZ Reader (LSTM)':<35} {corr(r['lstm_trt'], r['h_trt']):>8.3f} {corr(r['lstm_ffd'], r['h_ffd']):>8.3f} {mae(r['lstm_trt'], r['h_trt']):>9.1f}ms")
    if has_bert:
        print(f"  {'4. Neural EZ Reader (BERT)':<35} {bert_trt_corr:>8.3f} {bert_ffd_corr:>8.3f} {bert_trt_mae:>9.1f}ms")

    best_list = [
        ('Original EZ', corr(r['orig_trt'], r['h_trt'])),
        ('Diff EZ', corr(r['diff_trt'], r['h_trt'])),
        ('LSTM+Diff', corr(r['lstm_trt'], r['h_trt'])),
    ]
    if has_bert:
        best_list.append(('BERT+Diff', bert_trt_corr))

    best = max(best_list, key=lambda x: x[1])
    print(f"\n  WINNER: {best[0]} (r_TRT = {best[1]:.3f})")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    sys.stdout = Logger("comparison_results.txt")

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    lstm_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'checkpoints_lstm', 'best_model_lstm.pt')
    bert_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'checkpoints_bert', 'best_model_bert.pt')

    device = torch.device('cpu')

    # ---- Load data ----
    print("Loading data...")
    raw = load_provo(os.path.join(data_dir, 'Provo_Corpus-Eyetracking_Data.csv'))
    all_sentences = aggregate_by_sentence(raw, min_participants=10)
    subtlex = load_subtlexus(os.path.join(data_dir, 'SUBTLEXus.txt'))

    # ---- Split using the SAME seed as training ----
    train_agg, val_agg, test_agg = split_aggregated(all_sentences, seed=42)
    print(f"  Total: {len(all_sentences)} sentences, {sum(len(s) for s in all_sentences)} words")
    print(f"  Train: {len(train_agg)} sentences ({sum(len(s) for s in train_agg)} words)")
    print(f"  Val:   {len(val_agg)} sentences ({sum(len(s) for s in val_agg)} words)")
    print(f"  Test:  {len(test_agg)} sentences ({sum(len(s) for s in test_agg)} words)")
    print(f"  SUBTLEXus: {len(subtlex):,} entries")

    train_text_ids = set(s.text_id for s in train_agg)
    val_text_ids = set(s.text_id for s in val_agg)
    test_text_ids = set(s.text_id for s in test_agg)
    print(f"  Train text IDs: {sorted(train_text_ids)}")
    print(f"  Val text IDs:   {sorted(val_text_ids)}")
    print(f"  Test text IDs:  {sorted(test_text_ids)}")

    # ---- Load trained Neural LSTM model ----
    print("\nLoading Neural EZ Reader (LSTM)...")
    ckpt_lstm = torch.load(lstm_checkpoint, map_location=device, weights_only=False)
    vocab = ckpt_lstm['vocab']
    lstm_model = NeuralEZReader(vocab_size=len(vocab)).to(device)
    lstm_model.load_state_dict(ckpt_lstm['model_state_dict'], strict=False)
    lstm_model.eval()

    # ---- Load trained Neural BERT model ----
    print("Loading Neural EZ Reader (BERT)...")
    try:
        ckpt_bert = torch.load(bert_checkpoint, map_location=device, weights_only=False)
        bert_model = NeuralEZReaderBERT(
            bert_model_name=ckpt_bert.get('bert_model_name', 'bert-base-uncased'),
            freeze_bert_layers=ckpt_bert.get('freeze_bert_layers', 8)
        ).to(device)
        bert_model.load_state_dict(ckpt_bert['model_state_dict'])
        bert_model.eval()
    except Exception as e:
        print(f"Warning: Could not load BERT model ({e}). Skipping BERT.")
        bert_model = None

    # ---- Differentiable EZ Reader (untrained) ----
    diff_ezr = DifferentiableEZReader()
    diff_ezr.eval()

    # ================================================================
    # Run on TEST SET ONLY (the honest numbers for a paper)
    # ================================================================
    print(f"\n{'#' * 100}")
    print(f"#  EVALUATING ON HELD-OUT TEST SET ONLY")
    print(f"#  (LSTM and BERT have NEVER seen these sentences during training)")
    print(f"{'#' * 100}")

    t0 = time.time()
    test_results = run_all_models(
        test_agg, subtlex, diff_ezr, lstm_model, vocab, bert_model, device)
    elapsed = time.time() - t0
    print(f"\nDone! {len(test_results['h_trt'])} words in {elapsed:.1f}s")

    print_results(
        test_results,
        split_name="TEST SET — HELD-OUT EVALUATION (seed=42, split by text_id)",
        n_sentences=len(test_agg),
        subtlex=subtlex,
        bert_model=bert_model,
        diff_ezr=diff_ezr,
        lstm_model=lstm_model,
        vocab=vocab,
        sentences_for_samples=test_agg,
    )

    # ================================================================
    # Also run on TRAIN SET for reference (to show train vs test gap)
    # ================================================================
    print(f"\n\n{'#' * 100}")
    print(f"#  TRAIN SET (for reference only — expect inflated numbers)")
    print(f"{'#' * 100}")

    t0 = time.time()
    train_results = run_all_models(
        train_agg, subtlex, diff_ezr, lstm_model, vocab, bert_model, device)
    elapsed = time.time() - t0
    print(f"\nDone! {len(train_results['h_trt'])} words in {elapsed:.1f}s")

    print_results(
        train_results,
        split_name="TRAIN SET (reference — models were trained on this data)",
        n_sentences=len(train_agg),
        subtlex=subtlex,
        bert_model=bert_model,
        diff_ezr=diff_ezr,
        lstm_model=lstm_model,
        vocab=vocab,
    )

    # ================================================================
    # Summary: train vs test gap
    # ================================================================
    W = 100
    print(f"\n\n{'=' * W}")
    print("  GENERALIZATION GAP (train r_TRT vs test r_TRT)")
    print(f"{'=' * W}")
    print(f"  {'Model':<35} {'Train r_TRT':>12} {'Test r_TRT':>12} {'Gap':>8}")
    print(f"  {'-'*70}")

    for name, train_pred, test_pred in [
        ('Original EZ Reader', 'orig_trt', 'orig_trt'),
        ('Diff EZ Reader', 'diff_trt', 'diff_trt'),
        ('Neural EZ Reader (LSTM)', 'lstm_trt', 'lstm_trt'),
    ]:
        r_train = corr(train_results[train_pred], train_results['h_trt'])
        r_test = corr(test_results[test_pred], test_results['h_trt'])
        gap = r_train - r_test
        print(f"  {name:<35} {r_train:>12.3f} {r_test:>12.3f} {gap:>+8.3f}")

    if bert_model:
        r_train = corr(train_results['bert_trt'], train_results['h_trt'])
        r_test = corr(test_results['bert_trt'], test_results['h_trt'])
        gap = r_train - r_test
        print(f"  {'Neural EZ Reader (BERT)':<35} {r_train:>12.3f} {r_test:>12.3f} {gap:>+8.3f}")

    print(f"\n  Note: Original EZ and Diff EZ are formula-based (no training),")
    print(f"        so their gap reflects data characteristics, not overfitting.")
    print(f"        LSTM/BERT gap reflects actual overfitting.")


if __name__ == "__main__":
    main()
