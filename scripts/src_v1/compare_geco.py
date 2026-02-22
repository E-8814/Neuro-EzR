"""
GECO-trained Model Comparison + Psycholinguistic Effects Analysis.

Evaluates 4 models (all using GECO-trained checkpoints where applicable):
  1. Original EZ Reader     - formula L1/L2 + discrete simulation
  2. Differentiable EZ Reader - formula L1/L2 + smooth approximation
  3. Neural EZ Reader (LSTM) - GECO-trained LSTM + diff EZ Reader
  4. Neural EZ Reader (BERT) - GECO-trained BERT + diff EZ Reader

Evaluation on TWO datasets:
  A. GECO held-out test set   (in-distribution)
  B. Full Provo corpus        (cross-corpus generalization)

Plus psycholinguistic effects analysis (frequency, predictability,
word length, freq x pred interaction, content vs function) on both corpora.
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

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ez_reader'))

from diff_ezreader import DifferentiableEZReader
from model_lstm import NeuralEZReader, Vocabulary
from model_bert import NeuralEZReaderBERT
from data_loader import load_provo, aggregate_by_sentence, split_aggregated
from geco_loader import load_geco, split_geco
from ez_wrapper import run_original_simulation_averaged
from utilities import time_familiarity_check, time_lexical_access

# Alias so torch.load can unpickle old checkpoints
import model_lstm as _model_lstm_alias
sys.modules['model'] = _model_lstm_alias


# --------------------------------------------------------------------------- #
#  Logger (tee to file + terminal)
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
#  Metrics
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
#  Content/Function labels (Provo only)
# --------------------------------------------------------------------------- #

def load_content_function_labels(csv_path):
    raw = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                text_id = int(row['Text_ID'])
                sent_num = int(row['Sentence_Number'])
                word_pos = int(row['Word_In_Sentence_Number'])
            except (ValueError, KeyError):
                continue
            label = row.get('Word_Content_Or_Function', 'NA')
            key = (text_id, sent_num, word_pos)
            if key not in raw:
                raw[key] = label

    sentences = defaultdict(dict)
    for (text_id, sent_num, word_pos), label in raw.items():
        sentences[(text_id, sent_num)][word_pos] = label

    result = {}
    for (text_id, sent_num), pos_map in sentences.items():
        max_pos = max(pos_map.keys())
        labels = [pos_map.get(i, 'NA') for i in range(1, max_pos + 1)]
        result[(text_id, sent_num)] = labels
    return result


# --------------------------------------------------------------------------- #
#  Run all 4 models on a list of aggregated sentences
# --------------------------------------------------------------------------- #

def run_all_models(sentences, subtlex, diff_ezr, lstm_model, vocab,
                   bert_model, device):
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
                torch.tensor([l1f], dtype=torch.float32, device=device),
                torch.tensor([l2f], dtype=torch.float32, device=device),
                torch.tensor([preds], dtype=torch.float32, device=device),
                torch.tensor([wlens], dtype=torch.float32, device=device),
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
#  Collect per-word data for effects analysis
# --------------------------------------------------------------------------- #

def collect_per_word_data(sentences, subtlex, diff_ezr, lstm_model, vocab,
                          bert_model, device, cf_labels=None):
    words = []
    has_bert = bert_model is not None

    for agg in sentences:
        tokens = agg.tokens
        preds = agg.predictabilities
        wlens = [len(t) for t in tokens]
        sent_key = (agg.text_id, agg.sentence_number)

        cf = (cf_labels or {}).get(sent_key, ['NA'] * len(tokens))

        l1f, l2f = compute_real_l1_l2(tokens, preds, subtlex)
        freqs = [get_real_frequency(t, subtlex) for t in tokens]
        orig_result = run_original_simulation_averaged(
            tokens, freqs, preds, num_runs=20)

        with torch.no_grad():
            dr = diff_ezr(
                torch.tensor([l1f], dtype=torch.float32, device=device),
                torch.tensor([l2f], dtype=torch.float32, device=device),
                torch.tensor([preds], dtype=torch.float32, device=device),
                torch.tensor([wlens], dtype=torch.float32, device=device),
            )
            nr = lstm_model(
                vocab.encode_sentence(tokens).unsqueeze(0).to(device),
                torch.tensor([preds], dtype=torch.float32).to(device),
                torch.tensor([wlens], dtype=torch.float32).to(device),
            )
            if has_bert:
                br = bert_model(
                    [tokens],
                    torch.tensor([preds], dtype=torch.float32).to(device),
                    torch.tensor([wlens], dtype=torch.float32).to(device),
                )

        for i in range(len(tokens)):
            freq = get_real_frequency(tokens[i], subtlex)
            w = {
                'token': tokens[i],
                'freq': freq,
                'log_freq': math.log10(max(1, freq)),
                'pred': preds[i],
                'wlen': wlens[i],
                'cf': cf[i] if i < len(cf) else 'NA',
                'h_ffd': agg.mean_ffd[i],
                'h_gaze': agg.mean_gaze[i],
                'h_trt': agg.mean_trt[i],
                'h_skip': agg.skip_rate[i],
                'orig_ffd': orig_result['first_fixation_duration'][i],
                'orig_gaze': None,
                'orig_trt': orig_result['total_reading_time'][i],
                'orig_skip': None,
                'diff_ffd': dr['first_fixation'][0, i].item(),
                'diff_gaze': dr['gaze_duration'][0, i].item(),
                'diff_trt': dr['total_reading_time'][0, i].item(),
                'diff_skip': dr['skip_prob'][0, i].item(),
                'lstm_ffd': nr['first_fixation'][0, i].cpu().item(),
                'lstm_gaze': nr['gaze_duration'][0, i].cpu().item(),
                'lstm_trt': nr['total_reading_time'][0, i].cpu().item(),
                'lstm_skip': nr['skip_prob'][0, i].cpu().item(),
                'bert_ffd': br['first_fixation'][0, i].cpu().item() if has_bert else None,
                'bert_gaze': br['gaze_duration'][0, i].cpu().item() if has_bert else None,
                'bert_trt': br['total_reading_time'][0, i].cpu().item() if has_bert else None,
                'bert_skip': br['skip_prob'][0, i].cpu().item() if has_bert else None,
            }
            words.append(w)

    return words


# --------------------------------------------------------------------------- #
#  Print comparison results table
# --------------------------------------------------------------------------- #

def print_results(r, split_name, n_sentences, subtlex, bert_model,
                  diff_ezr, lstm_model, vocab, device, sentences_for_samples=None):
    W = 100
    n = len(r['h_trt'])
    has_bert = bert_model is not None and len(r['bert_trt']) > 0

    print(f"\n{'=' * W}")
    print(f"   {split_name}")
    print(f"{'=' * W}")
    print(f"  Words: {n} | Sentences: {n_sentences}")

    # ---- Correlation ----
    bert_trt_corr = corr(r['bert_trt'], r['h_trt']) if has_bert else 0.0
    bert_ffd_corr = corr(r['bert_ffd'], r['h_ffd']) if has_bert else 0.0
    bert_skip_corr = corr(r['bert_skip'], r['h_skip']) if has_bert else 0.0

    print(f"\n{'=' * W}")
    print("  CORRELATION WITH HUMAN DATA (Pearson r)")
    print(f"{'=' * W}")
    print(f"  {'Metric':<28} {'Orig EZ':>10} {'Diff EZ':>10} {'LSTM+Diff':>10} {'BERT+Diff':>10}")
    print(f"  {'-'*80}")
    print(f"  {'Total Reading Time (TRT)':<28} {corr(r['orig_trt'], r['h_trt']):>10.3f} {corr(r['diff_trt'], r['h_trt']):>10.3f} {corr(r['lstm_trt'], r['h_trt']):>10.3f} {bert_trt_corr:>10.3f}")
    print(f"  {'First Fixation Dur. (FFD)':<28} {corr(r['orig_ffd'], r['h_ffd']):>10.3f} {corr(r['diff_ffd'], r['h_ffd']):>10.3f} {corr(r['lstm_ffd'], r['h_ffd']):>10.3f} {bert_ffd_corr:>10.3f}")
    print(f"  {'Skip Rate':<28} {'N/A':>10} {corr(r['diff_skip'], r['h_skip']):>10.3f} {corr(r['lstm_skip'], r['h_skip']):>10.3f} {bert_skip_corr:>10.3f}")

    # ---- Error ----
    bert_trt_mae = mae(r['bert_trt'], r['h_trt']) if has_bert else 0.0
    bert_ffd_mae = mae(r['bert_ffd'], r['h_ffd']) if has_bert else 0.0
    bert_trt_rmse = rmse(r['bert_trt'], r['h_trt']) if has_bert else 0.0
    bert_ffd_rmse = rmse(r['bert_ffd'], r['h_ffd']) if has_bert else 0.0

    print(f"\n{'=' * W}")
    print("  ERROR METRICS")
    print(f"{'=' * W}")
    print(f"  {'Metric':<28} {'Orig EZ':>10} {'Diff EZ':>10} {'LSTM+Diff':>10} {'BERT+Diff':>10}")
    print(f"  {'-'*80}")
    print(f"  {'MAE TRT (ms)':<28} {mae(r['orig_trt'], r['h_trt']):>10.1f} {mae(r['diff_trt'], r['h_trt']):>10.1f} {mae(r['lstm_trt'], r['h_trt']):>10.1f} {bert_trt_mae:>10.1f}")
    print(f"  {'MAE FFD (ms)':<28} {mae(r['orig_ffd'], r['h_ffd']):>10.1f} {mae(r['diff_ffd'], r['h_ffd']):>10.1f} {mae(r['lstm_ffd'], r['h_ffd']):>10.1f} {bert_ffd_mae:>10.1f}")
    print(f"  {'RMSE TRT (ms)':<28} {rmse(r['orig_trt'], r['h_trt']):>10.1f} {rmse(r['diff_trt'], r['h_trt']):>10.1f} {rmse(r['lstm_trt'], r['h_trt']):>10.1f} {bert_trt_rmse:>10.1f}")
    print(f"  {'RMSE FFD (ms)':<28} {rmse(r['orig_ffd'], r['h_ffd']):>10.1f} {rmse(r['diff_ffd'], r['h_ffd']):>10.1f} {rmse(r['lstm_ffd'], r['h_ffd']):>10.1f} {bert_ffd_rmse:>10.1f}")

    # ---- Means ----
    bert_trt_mean = np.mean(r['bert_trt']) if has_bert else 0.0
    bert_ffd_mean = np.mean(r['bert_ffd']) if has_bert else 0.0
    bert_trt_std = np.std(r['bert_trt']) if has_bert else 0.0
    bert_skip_mean = np.mean(r['bert_skip']) if has_bert else 0.0

    print(f"\n{'=' * W}")
    print("  MEAN PREDICTIONS (ms)")
    print(f"{'=' * W}")
    print(f"  {'Metric':<28} {'Human':>10} {'Orig EZ':>10} {'Diff EZ':>10} {'LSTM+Diff':>10} {'BERT+Diff':>10}")
    print(f"  {'-'*90}")
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

    # ---- Sample predictions ----
    if sentences_for_samples:
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
                    torch.tensor([l1f], dtype=torch.float32, device=device),
                    torch.tensor([l2f], dtype=torch.float32, device=device),
                    torch.tensor([s.predictabilities], dtype=torch.float32, device=device),
                    torch.tensor([[len(t) for t in s.tokens]], dtype=torch.float32, device=device),
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
                ht = s.mean_trt[i]
                hs = s.skip_rate[i]
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
    print(f"  {'Model':<35} {'r_TRT':>8} {'r_FFD':>8} {'r_Skip':>8} {'MAE_TRT':>10}")
    print(f"  {'-'*75}")
    print(f"  {'1. Original EZ Reader (real freq)':<35} {corr(r['orig_trt'], r['h_trt']):>8.3f} {corr(r['orig_ffd'], r['h_ffd']):>8.3f} {'N/A':>8} {mae(r['orig_trt'], r['h_trt']):>9.1f}ms")
    print(f"  {'2. Diff EZ Reader (real freq)':<35} {corr(r['diff_trt'], r['h_trt']):>8.3f} {corr(r['diff_ffd'], r['h_ffd']):>8.3f} {corr(r['diff_skip'], r['h_skip']):>8.3f} {mae(r['diff_trt'], r['h_trt']):>9.1f}ms")
    print(f"  {'3. Neural EZ Reader (LSTM)':<35} {corr(r['lstm_trt'], r['h_trt']):>8.3f} {corr(r['lstm_ffd'], r['h_ffd']):>8.3f} {corr(r['lstm_skip'], r['h_skip']):>8.3f} {mae(r['lstm_trt'], r['h_trt']):>9.1f}ms")
    if has_bert:
        print(f"  {'4. Neural EZ Reader (BERT)':<35} {bert_trt_corr:>8.3f} {bert_ffd_corr:>8.3f} {bert_skip_corr:>8.3f} {mae(r['bert_trt'], r['h_trt']):>9.1f}ms")

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
#  Binning functions for effects analysis
# --------------------------------------------------------------------------- #

def compute_tertile_boundaries(values):
    s = sorted(values)
    n = len(s)
    return s[n // 3], s[2 * n // 3]

def bin_frequency(words):
    log_freqs = [w['log_freq'] for w in words]
    lo, hi = compute_tertile_boundaries(log_freqs)
    bins = {'Low freq': [], 'Med freq': [], 'High freq': []}
    for w in words:
        if w['log_freq'] <= lo:     bins['Low freq'].append(w)
        elif w['log_freq'] <= hi:   bins['Med freq'].append(w)
        else:                       bins['High freq'].append(w)
    return bins

def bin_predictability(words):
    bins = {'Zero pred': [], 'Low pred': [], 'High pred': []}
    for w in words:
        if w['pred'] == 0.0:    bins['Zero pred'].append(w)
        elif w['pred'] <= 0.3:  bins['Low pred'].append(w)
        else:                   bins['High pred'].append(w)
    return bins

def bin_word_length(words):
    bins = {'Short (1-3)': [], 'Med (4-6)': [], 'Long (7+)': []}
    for w in words:
        if w['wlen'] <= 3:   bins['Short (1-3)'].append(w)
        elif w['wlen'] <= 6: bins['Med (4-6)'].append(w)
        else:                bins['Long (7+)'].append(w)
    return bins

def bin_freq_x_pred(words):
    log_freqs = [w['log_freq'] for w in words]
    median_freq = sorted(log_freqs)[len(log_freqs) // 2]
    bins = {
        'LowFreq+LowPred': [], 'LowFreq+HighPred': [],
        'HighFreq+LowPred': [], 'HighFreq+HighPred': [],
    }
    for w in words:
        freq_hi = w['log_freq'] > median_freq
        pred_hi = w['pred'] > 0.3
        if not freq_hi and not pred_hi:   bins['LowFreq+LowPred'].append(w)
        elif not freq_hi and pred_hi:     bins['LowFreq+HighPred'].append(w)
        elif freq_hi and not pred_hi:     bins['HighFreq+LowPred'].append(w)
        else:                             bins['HighFreq+HighPred'].append(w)
    return bins

def bin_content_function(words):
    bins = {'Content': [], 'Function': []}
    for w in words:
        if w['cf'] == 'Content':    bins['Content'].append(w)
        elif w['cf'] == 'Function': bins['Function'].append(w)
    return bins


# --------------------------------------------------------------------------- #
#  Effects table printing
# --------------------------------------------------------------------------- #

MODEL_KEYS = ['h', 'orig', 'diff', 'lstm', 'bert']
MODEL_NAMES = ['Human', 'Orig EZ', 'Diff EZ', 'LSTM', 'BERT']
MEASURES = ['ffd', 'gaze', 'trt', 'skip']
MEASURE_NAMES = ['FFD (ms)', 'Gaze (ms)', 'TRT (ms)', 'Skip Rate']


def bin_mean(word_list, model, measure):
    key = f'{model}_{measure}'
    vals = [w[key] for w in word_list if w[key] is not None]
    return np.mean(vals) if vals else None


def print_effect_table(effect_name, bins, measure, measure_name,
                       expected_direction, has_bert):
    bin_names = list(bins.keys())
    W = 90

    print(f"\n  {effect_name} ON {measure_name}")
    print(f"  {'-' * W}")

    models = MODEL_NAMES[:4] if not has_bert else MODEL_NAMES
    header = f"  {'':>20}"
    for m in models:
        header += f"  {m:>10}"
    header += f"  {'N':>6}"
    print(header)
    print(f"  {'-' * W}")

    means = {mk: [] for mk in MODEL_KEYS}
    for bname in bin_names:
        row = f"  {bname:>20}"
        for mk, mname in zip(MODEL_KEYS, MODEL_NAMES):
            if not has_bert and mk == 'bert':
                continue
            val = bin_mean(bins[bname], mk, measure)
            means[mk].append(val)
            if val is None:
                row += f"  {'N/A':>10}"
            elif measure == 'skip':
                row += f"  {val:>10.3f}"
            else:
                row += f"  {val:>10.1f}"
        row += f"  {len(bins[bname]):>6}"
        print(row)

    # Effect size
    print(f"  {'':>20}", end='')
    effects = {}
    for mk, mname in zip(MODEL_KEYS, MODEL_NAMES):
        if not has_bert and mk == 'bert':
            continue
        first, last = means[mk][0], means[mk][-1]
        if first is not None and last is not None:
            eff = last - first
            effects[mk] = eff
            if measure == 'skip':
                print(f"  {eff:>+10.3f}", end='')
            else:
                print(f"  {eff:>+10.1f}", end='')
        else:
            effects[mk] = None
            print(f"  {'N/A':>10}", end='')
    print(f"  {'Eff(L-F)':>6}")

    # Direction check
    human_eff = effects.get('h')
    print(f"  {'Direction correct?':>20}", end='')
    results = {}
    for mk, mname in zip(MODEL_KEYS, MODEL_NAMES):
        if not has_bert and mk == 'bert':
            continue
        if mk == 'h':
            print(f"  {'---':>10}", end='')
            continue
        eff = effects.get(mk)
        if eff is None or human_eff is None or human_eff == 0:
            print(f"  {'N/A':>10}", end='')
            results[mk] = None
        elif (eff > 0) == (human_eff > 0):
            print(f"  {'YES':>10}", end='')
            results[mk] = True
        else:
            print(f"  {'NO':>10}", end='')
            results[mk] = False
    print()

    # Magnitude
    print(f"  {'Magnitude (% human)':>20}", end='')
    magnitudes = {}
    for mk, mname in zip(MODEL_KEYS, MODEL_NAMES):
        if not has_bert and mk == 'bert':
            continue
        if mk == 'h':
            print(f"  {'---':>10}", end='')
            continue
        eff = effects.get(mk)
        if eff is None or human_eff is None or human_eff == 0:
            print(f"  {'N/A':>10}", end='')
            magnitudes[mk] = None
        else:
            pct = abs(eff) / abs(human_eff) * 100
            print(f"  {pct:>9.0f}%", end='')
            magnitudes[mk] = pct
    print()

    reproduces = {}
    for mk in MODEL_KEYS:
        if mk == 'h':
            continue
        dir_ok = results.get(mk)
        mag = magnitudes.get(mk)
        if dir_ok is True and mag is not None and mag >= 25:
            reproduces[mk] = True
        elif dir_ok is None or mag is None:
            reproduces[mk] = None
        else:
            reproduces[mk] = False
    return reproduces


def analyze_effect(effect_name, bins, expected_directions, has_bert):
    all_results = {}
    for measure, mname in zip(MEASURES, MEASURE_NAMES):
        direction = expected_directions.get(measure)
        if direction is None:
            continue
        results = print_effect_table(
            effect_name, bins, measure, mname, direction, has_bert)
        for mk, reproduces in results.items():
            all_results[(measure, mk)] = reproduces
    return all_results


def analyze_interaction(bins, has_bert):
    W = 90
    print(f"\n  FREQ x PRED INTERACTION")
    print(f"  {'-' * W}")
    results = {}
    models = MODEL_KEYS[:4] if not has_bert else MODEL_KEYS

    for measure, mname in zip(MEASURES, MEASURE_NAMES):
        if measure == 'skip':
            continue
        print(f"\n  {mname}:")
        header = f"  {'':>25}"
        for m_name in (MODEL_NAMES[:4] if not has_bert else MODEL_NAMES):
            header += f"  {m_name:>10}"
        print(header)

        cell_means = {}
        for bname in bins:
            cell_means[bname] = {}
            for mk in models:
                cell_means[bname][mk] = bin_mean(bins[bname], mk, measure)

        for bname in bins:
            row = f"  {bname:>25}"
            for mk in models:
                val = cell_means[bname][mk]
                row += f"  {val:>10.1f}" if val is not None else f"  {'N/A':>10}"
            row += f"  (N={len(bins[bname])})"
            print(row)

        print(f"  {'Freq eff (low pred)':>25}", end='')
        freq_eff_low = {}
        for mk in models:
            lf_lp = cell_means.get('LowFreq+LowPred', {}).get(mk)
            hf_lp = cell_means.get('HighFreq+LowPred', {}).get(mk)
            if lf_lp is not None and hf_lp is not None:
                eff = lf_lp - hf_lp
                freq_eff_low[mk] = eff
                print(f"  {eff:>+10.1f}", end='')
            else:
                freq_eff_low[mk] = None
                print(f"  {'N/A':>10}", end='')
        print()

        print(f"  {'Freq eff (high pred)':>25}", end='')
        freq_eff_high = {}
        for mk in models:
            lf_hp = cell_means.get('LowFreq+HighPred', {}).get(mk)
            hf_hp = cell_means.get('HighFreq+HighPred', {}).get(mk)
            if lf_hp is not None and hf_hp is not None:
                eff = lf_hp - hf_hp
                freq_eff_high[mk] = eff
                print(f"  {eff:>+10.1f}", end='')
            else:
                freq_eff_high[mk] = None
                print(f"  {'N/A':>10}", end='')
        print()

        print(f"  {'Interaction':>25}", end='')
        for mk in models:
            fl = freq_eff_low.get(mk)
            fh = freq_eff_high.get(mk)
            if fl is not None and fh is not None:
                interaction = abs(fl) - abs(fh)
                print(f"  {interaction:>+10.1f}", end='')
            else:
                print(f"  {'N/A':>10}", end='')
        print()

        print(f"  {'Correct? (int > 0)':>25}", end='')
        for mk in models:
            if mk == 'h':
                print(f"  {'---':>10}", end='')
                continue
            fl = freq_eff_low.get(mk)
            fh = freq_eff_high.get(mk)
            if fl is not None and fh is not None:
                interaction = abs(fl) - abs(fh)
                correct = interaction > 0
                print(f"  {'YES' if correct else 'NO':>10}", end='')
                results[(measure, mk)] = correct
            else:
                print(f"  {'N/A':>10}", end='')
                results[(measure, mk)] = None
        print()

    return results


def print_effects_summary(all_effects_results, has_bert):
    W = 100
    print(f"\n\n{'=' * W}")
    print(f"  EFFECTS SUMMARY: WHICH EFFECTS DOES EACH MODEL REPRODUCE?")
    print(f"{'=' * W}")
    print(f"  (Criteria: correct direction AND >= 25% of human effect size)")
    print()

    models = ['orig', 'diff', 'lstm']
    mnames = ['Orig EZ', 'Diff EZ', 'LSTM']
    if has_bert:
        models.append('bert')
        mnames.append('BERT')

    header = f"  {'Effect / Measure':<35}"
    for mn in mnames:
        header += f"  {mn:>10}"
    print(header)
    print(f"  {'-' * (35 + 12 * len(models))}")

    pass_counts = {mk: 0 for mk in models}
    total_tests = {mk: 0 for mk in models}

    for (effect_name, measure_name), effect_results in all_effects_results:
        label = f"{effect_name} / {measure_name}"
        row = f"  {label:<35}"
        for mk in models:
            val = effect_results.get((measure_name, mk))
            if val is True:
                row += f"  {'PASS':>10}"
                pass_counts[mk] += 1
                total_tests[mk] += 1
            elif val is False:
                row += f"  {'FAIL':>10}"
                total_tests[mk] += 1
            else:
                row += f"  {'N/A':>10}"
        print(row)

    print(f"  {'-' * (35 + 12 * len(models))}")
    row = f"  {'TOTAL PASS':>35}"
    for mk in models:
        total = total_tests[mk]
        if total > 0:
            row += f"  {pass_counts[mk]:>3}/{total:<3}"
        else:
            row += f"  {'---':>10}"
    print(row)

    row = f"  {'PASS RATE':>35}"
    for mk in models:
        total = total_tests[mk]
        if total > 0:
            row += f"  {pass_counts[mk] / total * 100:>9.0f}%"
        else:
            row += f"  {'---':>10}"
    print(row)


# --------------------------------------------------------------------------- #
#  Run full effects analysis on a set of sentences
# --------------------------------------------------------------------------- #

def run_effects_analysis(corpus_name, words, has_bert, has_cf=False):
    W = 100
    all_effects = []

    print(f"\n\n{'#' * W}")
    print(f"#  PSYCHOLINGUISTIC EFFECTS ANALYSIS — {corpus_name}")
    print(f"{'#' * W}")

    print(f"\n  Human data summary:")
    print(f"    Words: {len(words)}")
    print(f"    Mean FFD  = {np.mean([w['h_ffd'] for w in words]):.1f} ms")
    print(f"    Mean Gaze = {np.mean([w['h_gaze'] for w in words]):.1f} ms")
    print(f"    Mean TRT  = {np.mean([w['h_trt'] for w in words]):.1f} ms")
    print(f"    Mean Skip = {np.mean([w['h_skip'] for w in words]):.3f}")

    # Effect 1: Frequency
    print(f"\n\n{'=' * W}")
    print(f"  EFFECT 1: WORD FREQUENCY")
    print(f"{'=' * W}")
    freq_bins = bin_frequency(words)
    for bname, bwords in freq_bins.items():
        print(f"  {bname}: N={len(bwords)}")
    freq_results = analyze_effect(
        "FREQUENCY", freq_bins,
        {'ffd': 'decrease', 'gaze': 'decrease', 'trt': 'decrease', 'skip': 'increase'},
        has_bert)
    for measure in MEASURES:
        all_effects.append((("Frequency", measure), freq_results))

    # Effect 2: Predictability
    print(f"\n\n{'=' * W}")
    print(f"  EFFECT 2: PREDICTABILITY")
    print(f"{'=' * W}")
    pred_bins = bin_predictability(words)
    for bname, bwords in pred_bins.items():
        print(f"  {bname}: N={len(bwords)}")
    pred_results = analyze_effect(
        "PREDICTABILITY", pred_bins,
        {'ffd': 'decrease', 'gaze': 'decrease', 'trt': 'decrease', 'skip': 'increase'},
        has_bert)
    for measure in MEASURES:
        all_effects.append((("Predictability", measure), pred_results))

    # Effect 3: Word Length
    print(f"\n\n{'=' * W}")
    print(f"  EFFECT 3: WORD LENGTH")
    print(f"{'=' * W}")
    wlen_bins = bin_word_length(words)
    for bname, bwords in wlen_bins.items():
        print(f"  {bname}: N={len(bwords)}")
    wlen_results = analyze_effect(
        "WORD LENGTH", wlen_bins,
        {'ffd': 'increase', 'gaze': 'increase', 'trt': 'increase', 'skip': 'decrease'},
        has_bert)
    for measure in MEASURES:
        all_effects.append((("Word Length", measure), wlen_results))

    # Effect 4: Freq x Pred Interaction
    print(f"\n\n{'=' * W}")
    print(f"  EFFECT 4: FREQUENCY x PREDICTABILITY INTERACTION")
    print(f"{'=' * W}")
    fxp_bins = bin_freq_x_pred(words)
    for bname, bwords in fxp_bins.items():
        print(f"  {bname}: N={len(bwords)}")
    interaction_results = analyze_interaction(fxp_bins, has_bert)
    for measure in ['ffd', 'gaze', 'trt']:
        all_effects.append((("Freq x Pred", measure), interaction_results))

    # Effect 5: Content vs Function (only if labels available)
    if has_cf:
        cf_content = sum(1 for w in words if w['cf'] == 'Content')
        cf_func = sum(1 for w in words if w['cf'] == 'Function')
        if cf_content > 0 and cf_func > 0:
            print(f"\n\n{'=' * W}")
            print(f"  EFFECT 5: CONTENT vs FUNCTION WORDS")
            print(f"{'=' * W}")
            cf_bins = bin_content_function(words)
            for bname, bwords in cf_bins.items():
                print(f"  {bname}: N={len(bwords)}")
            cf_results = analyze_effect(
                "CONTENT vs FUNCTION", cf_bins,
                {'ffd': 'decrease', 'gaze': 'decrease', 'trt': 'decrease', 'skip': 'increase'},
                has_bert)
            for measure in MEASURES:
                all_effects.append((("Content/Function", measure), cf_results))

    # Summary
    print_effects_summary(all_effects, has_bert)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    args = parser.parse_args()

    output_path = os.path.join(os.path.dirname(__file__), 'comparison_geco_results.txt')
    sys.stdout = Logger(output_path)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    geco_lstm_ckpt = os.path.join(os.path.dirname(__file__), '..', 'checkpoints_v1/geco_lstm', 'best_model_lstm.pt')
    geco_bert_ckpt = os.path.join(os.path.dirname(__file__), '..', 'checkpoints_v1/geco_bert', 'best_model_bert.pt')

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    W = 100

    print(f"{'=' * W}")
    print(f"  GECO-TRAINED MODEL COMPARISON + EFFECTS ANALYSIS")
    print(f"  Models trained on GECO, evaluated on GECO test + full Provo")
    print(f"{'=' * W}")
    print(f"  Device: {device}")

    # ================================================================
    #  Load data
    # ================================================================
    print("\nLoading GECO Corpus...")
    reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
    material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
    pred_path = os.path.join(data_dir, "geco_predictability.pkl")

    geco_raw = load_geco(reading_path, material_path, pred_path)
    train_raw, val_raw, test_raw = split_geco(geco_raw)

    geco_aggregated = aggregate_by_sentence(geco_raw, min_participants=5)
    train_text_ids = set(sd.text_id for sd in train_raw)
    val_text_ids = set(sd.text_id for sd in val_raw)
    geco_test_agg = [a for a in geco_aggregated
                     if a.text_id not in train_text_ids and a.text_id not in val_text_ids]
    print(f"  GECO test set: {len(geco_test_agg)} sentences, "
          f"{sum(len(s) for s in geco_test_agg)} words")

    print("\nLoading Provo Corpus...")
    et_csv = os.path.join(data_dir, 'Provo_Corpus-Eyetracking_Data.csv')
    provo_raw = load_provo(et_csv)
    provo_all = aggregate_by_sentence(provo_raw, min_participants=10)
    print(f"  Provo: {len(provo_all)} sentences, "
          f"{sum(len(s) for s in provo_all)} words")

    print("\nLoading SUBTLEXus...")
    subtlex = load_subtlexus(os.path.join(data_dir, 'SUBTLEXus.txt'))
    print(f"  {len(subtlex):,} entries")

    # Content/function labels (Provo only)
    cf_labels = load_content_function_labels(et_csv)
    print(f"  Content/Function labels: {len(cf_labels)} sentences (Provo)")

    # ================================================================
    #  Load models (GECO-trained)
    # ================================================================
    print("\nLoading GECO-trained models...")

    # LSTM
    ckpt_lstm = torch.load(geco_lstm_ckpt, map_location=device, weights_only=False)
    vocab = ckpt_lstm['vocab']
    lstm_model = NeuralEZReader(vocab_size=len(vocab)).to(device)
    lstm_model.load_state_dict(ckpt_lstm['model_state_dict'], strict=False)
    lstm_model.eval()
    print(f"  LSTM: loaded (epoch {ckpt_lstm.get('epoch', '?')})")

    # BERT
    bert_model = None
    try:
        ckpt_bert = torch.load(geco_bert_ckpt, map_location=device, weights_only=False)
        bert_model = NeuralEZReaderBERT(
            bert_model_name=ckpt_bert.get('bert_model_name', 'bert-base-uncased'),
            freeze_bert_layers=ckpt_bert.get('freeze_bert_layers', 8)
        ).to(device)
        bert_model.load_state_dict(ckpt_bert['model_state_dict'], strict=False)
        bert_model.eval()
        print(f"  BERT: loaded (epoch {ckpt_bert.get('epoch', '?')})")
    except Exception as e:
        print(f"  BERT: SKIPPED ({e})")

    # Differentiable EZ Reader (untrained formula)
    diff_ezr = DifferentiableEZReader().to(device)
    diff_ezr.eval()
    print("  Diff EZ: loaded (untrained formula)")

    has_bert = bert_model is not None

    # ================================================================
    #  PART 1: GECO TEST SET (in-distribution)
    # ================================================================
    print(f"\n\n{'#' * W}")
    print(f"#  PART 1: GECO HELD-OUT TEST SET")
    print(f"#  (LSTM and BERT trained on GECO train split — never saw these sentences)")
    print(f"{'#' * W}")

    t0 = time.time()
    geco_test_results = run_all_models(
        geco_test_agg, subtlex, diff_ezr, lstm_model, vocab, bert_model, device)
    print(f"\n  Done: {len(geco_test_results['h_trt'])} words in {time.time()-t0:.1f}s")

    print_results(
        geco_test_results,
        split_name="GECO TEST SET — IN-DISTRIBUTION EVALUATION",
        n_sentences=len(geco_test_agg),
        subtlex=subtlex,
        bert_model=bert_model,
        diff_ezr=diff_ezr,
        lstm_model=lstm_model,
        vocab=vocab,
        device=device,
        sentences_for_samples=geco_test_agg,
    )

    # ================================================================
    #  PART 2: FULL PROVO CORPUS (cross-corpus generalization)
    # ================================================================
    print(f"\n\n{'#' * W}")
    print(f"#  PART 2: FULL PROVO CORPUS (CROSS-CORPUS GENERALIZATION)")
    print(f"#  (Models trained on GECO, evaluated on entirely different corpus)")
    print(f"{'#' * W}")

    t0 = time.time()
    provo_results = run_all_models(
        provo_all, subtlex, diff_ezr, lstm_model, vocab, bert_model, device)
    print(f"\n  Done: {len(provo_results['h_trt'])} words in {time.time()-t0:.1f}s")

    print_results(
        provo_results,
        split_name="FULL PROVO CORPUS — CROSS-CORPUS EVALUATION",
        n_sentences=len(provo_all),
        subtlex=subtlex,
        bert_model=bert_model,
        diff_ezr=diff_ezr,
        lstm_model=lstm_model,
        vocab=vocab,
        device=device,
        sentences_for_samples=provo_all,
    )

    # ================================================================
    #  PART 3: PSYCHOLINGUISTIC EFFECTS — GECO TEST
    # ================================================================
    print(f"\n\nCollecting per-word data for GECO effects analysis...")
    t0 = time.time()
    geco_words = collect_per_word_data(
        geco_test_agg, subtlex, diff_ezr, lstm_model, vocab,
        bert_model, device, cf_labels=None)
    print(f"  Done: {len(geco_words)} words in {time.time()-t0:.1f}s")

    run_effects_analysis("GECO TEST SET", geco_words, has_bert, has_cf=False)

    # ================================================================
    #  PART 4: PSYCHOLINGUISTIC EFFECTS — FULL PROVO
    # ================================================================
    print(f"\n\nCollecting per-word data for Provo effects analysis...")
    t0 = time.time()
    provo_words = collect_per_word_data(
        provo_all, subtlex, diff_ezr, lstm_model, vocab,
        bert_model, device, cf_labels=cf_labels)
    print(f"  Done: {len(provo_words)} words in {time.time()-t0:.1f}s")

    run_effects_analysis("FULL PROVO CORPUS", provo_words, has_bert, has_cf=True)

    # ================================================================
    #  PART 5: CROSS-CORPUS SUMMARY
    # ================================================================
    print(f"\n\n{'=' * W}")
    print("  CROSS-CORPUS SUMMARY")
    print(f"{'=' * W}")
    print(f"  {'Model':<35} {'GECO r_TRT':>12} {'Provo r_TRT':>12} {'Drop':>8}")
    print(f"  {'-'*70}")

    for name, key in [
        ('Original EZ Reader', 'orig_trt'),
        ('Diff EZ Reader', 'diff_trt'),
        ('Neural EZ Reader (LSTM)', 'lstm_trt'),
    ]:
        r_geco = corr(geco_test_results[key], geco_test_results['h_trt'])
        r_provo = corr(provo_results[key], provo_results['h_trt'])
        drop = r_geco - r_provo
        print(f"  {name:<35} {r_geco:>12.3f} {r_provo:>12.3f} {drop:>+8.3f}")

    if has_bert:
        r_geco = corr(geco_test_results['bert_trt'], geco_test_results['h_trt'])
        r_provo = corr(provo_results['bert_trt'], provo_results['h_trt'])
        drop = r_geco - r_provo
        print(f"  {'Neural EZ Reader (BERT)':<35} {r_geco:>12.3f} {r_provo:>12.3f} {drop:>+8.3f}")

    print(f"\n  Note: Original EZ and Diff EZ are formula-based (no training),")
    print(f"        so their difference reflects corpus characteristics only.")
    print(f"        LSTM/BERT drop reflects generalization ability.")

    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
