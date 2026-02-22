"""
Psycholinguistic Effect Analysis.

Tests whether each model (Original EZ, Differentiable EZ, LSTM, BERT)
reproduces known experimental effects from the reading literature:

  1. Frequency effect       - high-freq words read faster, skipped more
  2. Predictability effect  - predictable words read faster, skipped more
  3. Word length effect     - longer words read slower, skipped less
  4. Freq x Pred interaction - freq effect larger for unpredictable words
  5. Content vs Function    - function words: shorter fixations, more skipping

Evaluation uses ALL 151 sentences (no train/test split) because we are
testing qualitative patterns, not predictive generalization. Formula models
are untrained anyway, and more data gives more stable bin means.
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

# The LSTM checkpoint was pickled when the module was named 'model'.
# Register an alias so torch.load can unpickle the Vocabulary class.
import model_lstm as _model_lstm_alias
sys.modules['model'] = _model_lstm_alias
from data_loader import load_provo, aggregate_by_sentence
from ez_wrapper import run_original_simulation_averaged
from compare_all_fair import (
    load_subtlexus, get_real_frequency, compute_real_l1_l2, Logger
)


# --------------------------------------------------------------------------- #
#  Load Content/Function labels from the CSV
# --------------------------------------------------------------------------- #

def load_content_function_labels(csv_path):
    """
    Read Word_Content_Or_Function from the eye-tracking CSV.

    Returns a dict: (text_id, sentence_number) -> list of labels per word position.
    Uses the first participant's row for each word position.
    """
    # Collect: (text_id, sent_num, word_in_sent) -> label
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

    # Reorganize into per-sentence lists
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
#  Run all models and collect per-word data
# --------------------------------------------------------------------------- #

def collect_all_data(sentences, subtlex, diff_ezr, lstm_model, vocab,
                     bert_model, device, cf_labels):
    """
    Run all 4 models on all sentences and return a list of per-word dicts.

    Each dict contains human data, model predictions, and word properties
    needed for binning (frequency, predictability, word length, content/function).
    """
    words = []

    for agg in sentences:
        tokens = agg.tokens
        preds = agg.predictabilities
        wlens = [len(t) for t in tokens]
        sent_key = (agg.text_id, agg.sentence_number)

        # Get content/function labels for this sentence
        cf = cf_labels.get(sent_key, ['NA'] * len(tokens))

        # Formula-based L1/L2
        l1f, l2f = compute_real_l1_l2(tokens, preds, subtlex)
        freqs = [get_real_frequency(t, subtlex) for t in tokens]

        # Model 1: Real original EZ Reader (computes L1/L2 internally with proper distance)
        orig_result = run_original_simulation_averaged(
            tokens, freqs, preds, num_runs=20)

        # Model 2: Differentiable
        with torch.no_grad():
            dr = diff_ezr(
                torch.tensor([l1f], dtype=torch.float32),
                torch.tensor([l2f], dtype=torch.float32),
                torch.tensor([preds], dtype=torch.float32),
                torch.tensor([wlens], dtype=torch.float32),
            )

        # Model 3: LSTM
        with torch.no_grad():
            nr = lstm_model(
                vocab.encode_sentence(tokens).unsqueeze(0).to(device),
                torch.tensor([preds], dtype=torch.float32).to(device),
                torch.tensor([wlens], dtype=torch.float32).to(device),
            )

        # Model 4: BERT
        has_bert = bert_model is not None
        if has_bert:
            with torch.no_grad():
                br = bert_model(
                    [tokens],
                    torch.tensor([preds], dtype=torch.float32).to(device),
                    torch.tensor([wlens], dtype=torch.float32).to(device),
                )

        for i in range(len(tokens)):
            freq = get_real_frequency(tokens[i], subtlex)
            log_freq = math.log10(max(1, freq))

            cf_label = cf[i] if i < len(cf) else 'NA'

            w = {
                'token': tokens[i],
                'freq': freq,
                'log_freq': log_freq,
                'pred': preds[i],
                'wlen': wlens[i],
                'cf': cf_label,
                # Human data
                'h_ffd': agg.mean_ffd[i],
                'h_gaze': agg.mean_gaze[i],
                'h_trt': agg.mean_trt[i],
                'h_skip': agg.skip_rate[i],
                # Orig EZ (no gaze, no skip)
                'orig_ffd': orig_result['first_fixation_duration'][i],
                'orig_gaze': None,  # original sim doesn't output gaze
                'orig_trt': orig_result['total_reading_time'][i],
                'orig_skip': None,  # original sim doesn't output skip
                # Diff EZ
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
                'bert_ffd': br['first_fixation'][0, i].cpu().item() if has_bert else None,
                'bert_gaze': br['gaze_duration'][0, i].cpu().item() if has_bert else None,
                'bert_trt': br['total_reading_time'][0, i].cpu().item() if has_bert else None,
                'bert_skip': br['skip_prob'][0, i].cpu().item() if has_bert else None,
            }
            words.append(w)

    return words


# --------------------------------------------------------------------------- #
#  Binning functions
# --------------------------------------------------------------------------- #

def compute_tertile_boundaries(values):
    """Return (low_upper, med_upper) for splitting into 3 equal-sized bins."""
    s = sorted(values)
    n = len(s)
    return s[n // 3], s[2 * n // 3]


def bin_frequency(words):
    """Bin words into LOW / MED / HIGH frequency by log10(freq) tertiles."""
    log_freqs = [w['log_freq'] for w in words]
    lo, hi = compute_tertile_boundaries(log_freqs)
    bins = {'Low freq': [], 'Med freq': [], 'High freq': []}
    for w in words:
        if w['log_freq'] <= lo:
            bins['Low freq'].append(w)
        elif w['log_freq'] <= hi:
            bins['Med freq'].append(w)
        else:
            bins['High freq'].append(w)
    return bins


def bin_predictability(words):
    """Bin words into ZERO / LOW / HIGH predictability."""
    bins = {'Zero pred': [], 'Low pred': [], 'High pred': []}
    for w in words:
        if w['pred'] == 0.0:
            bins['Zero pred'].append(w)
        elif w['pred'] <= 0.3:
            bins['Low pred'].append(w)
        else:
            bins['High pred'].append(w)
    return bins


def bin_word_length(words):
    """Bin words into SHORT (1-3) / MED (4-6) / LONG (7+)."""
    bins = {'Short (1-3)': [], 'Med (4-6)': [], 'Long (7+)': []}
    for w in words:
        if w['wlen'] <= 3:
            bins['Short (1-3)'].append(w)
        elif w['wlen'] <= 6:
            bins['Med (4-6)'].append(w)
        else:
            bins['Long (7+)'].append(w)
    return bins


def bin_freq_x_pred(words):
    """2x2 cross: LOW/HIGH freq x LOW/HIGH pred using median splits."""
    log_freqs = [w['log_freq'] for w in words]
    median_freq = sorted(log_freqs)[len(log_freqs) // 2]

    bins = {
        'LowFreq+LowPred': [],
        'LowFreq+HighPred': [],
        'HighFreq+LowPred': [],
        'HighFreq+HighPred': [],
    }
    for w in words:
        freq_hi = w['log_freq'] > median_freq
        pred_hi = w['pred'] > 0.3
        if not freq_hi and not pred_hi:
            bins['LowFreq+LowPred'].append(w)
        elif not freq_hi and pred_hi:
            bins['LowFreq+HighPred'].append(w)
        elif freq_hi and not pred_hi:
            bins['HighFreq+LowPred'].append(w)
        else:
            bins['HighFreq+HighPred'].append(w)
    return bins


def bin_content_function(words):
    """Bin into Content / Function (skip NA labels)."""
    bins = {'Content': [], 'Function': []}
    for w in words:
        if w['cf'] == 'Content':
            bins['Content'].append(w)
        elif w['cf'] == 'Function':
            bins['Function'].append(w)
    return bins


# --------------------------------------------------------------------------- #
#  Compute bin means
# --------------------------------------------------------------------------- #

MODEL_KEYS = ['h', 'orig', 'diff', 'lstm', 'bert']
MODEL_NAMES = ['Human', 'Orig EZ', 'Diff EZ', 'LSTM', 'BERT']
MEASURES = ['ffd', 'gaze', 'trt', 'skip']
MEASURE_NAMES = ['FFD (ms)', 'Gaze (ms)', 'TRT (ms)', 'Skip Rate']


def bin_mean(word_list, model, measure):
    """Compute mean of model_measure for a list of word dicts. Returns None if N/A."""
    key = f'{model}_{measure}'
    vals = [w[key] for w in word_list if w[key] is not None]
    if not vals:
        return None
    return np.mean(vals)


# --------------------------------------------------------------------------- #
#  Print one effect table
# --------------------------------------------------------------------------- #

def print_effect_table(effect_name, bins, measure, measure_name,
                       expected_direction, has_bert):
    """
    Print a table showing bin means for all models on one measure.

    expected_direction: 'decrease' or 'increase' — what should happen
    from first bin to last bin.
    """
    bin_names = list(bins.keys())
    W = 90

    print(f"\n  {effect_name} ON {measure_name}")
    print(f"  {'-' * W}")

    # Header
    models = MODEL_NAMES[:4] if not has_bert else MODEL_NAMES
    header = f"  {'':>20}"
    for m in models:
        header += f"  {m:>10}"
    header += f"  {'N':>6}"
    print(header)
    print(f"  {'-' * W}")

    # Bin means
    means = {}  # model_key -> list of bin means
    for mk in MODEL_KEYS:
        means[mk] = []

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

    # Effect size (last bin minus first bin)
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

    # Magnitude as % of human
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

    # Return whether each model reproduces this effect
    reproduces = {}
    for mk in MODEL_KEYS:
        if mk == 'h':
            continue
        dir_ok = results.get(mk)
        mag = magnitudes.get(mk)
        if dir_ok is True and mag is not None and mag >= 25:
            reproduces[mk] = True
        elif dir_ok is None or mag is None:
            reproduces[mk] = None  # N/A
        else:
            reproduces[mk] = False
    return reproduces


# --------------------------------------------------------------------------- #
#  Analyze one effect across all measures
# --------------------------------------------------------------------------- #

def analyze_effect(effect_name, bins, expected_directions, has_bert):
    """
    Analyze one effect across all 4 measures.

    expected_directions: dict mapping measure -> 'decrease' or 'increase'
        (describes what happens from first to last bin)

    Returns: dict of (measure, model) -> True/False/None
    """
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


# --------------------------------------------------------------------------- #
#  Interaction effect analysis
# --------------------------------------------------------------------------- #

def analyze_interaction(bins, has_bert):
    """
    Analyze the Freq x Pred interaction.

    The key test: frequency effect should be LARGER for low-predictability words.
    freq_effect_low_pred = LowFreq+LowPred - HighFreq+LowPred
    freq_effect_high_pred = LowFreq+HighPred - HighFreq+HighPred
    Interaction = freq_effect_low_pred - freq_effect_high_pred > 0
    """
    W = 90
    print(f"\n  FREQ x PRED INTERACTION")
    print(f"  {'-' * W}")

    results = {}

    for measure, mname in zip(MEASURES, MEASURE_NAMES):
        if measure == 'skip':
            continue  # skip rate interaction is less clear-cut

        print(f"\n  {mname}:")
        models = MODEL_KEYS[:4] if not has_bert else MODEL_KEYS
        header = f"  {'':>25}"
        for m_name in (MODEL_NAMES[:4] if not has_bert else MODEL_NAMES):
            header += f"  {m_name:>10}"
        print(header)

        # Compute means for each cell
        cell_means = {}
        for bname in bins:
            cell_means[bname] = {}
            for mk in models:
                cell_means[bname][mk] = bin_mean(bins[bname], mk, measure)

        # Print 2x2 table
        for bname in bins:
            row = f"  {bname:>25}"
            for mk in models:
                val = cell_means[bname][mk]
                if val is None:
                    row += f"  {'N/A':>10}"
                else:
                    row += f"  {val:>10.1f}"
            row += f"  (N={len(bins[bname])})"
            print(row)

        # Freq effect for low pred
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

        # Freq effect for high pred
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

        # Interaction: is freq effect larger for low-pred?
        print(f"  {'Interaction':>25}", end='')
        human_interaction = None
        for mk in models:
            fl = freq_eff_low.get(mk)
            fh = freq_eff_high.get(mk)
            if fl is not None and fh is not None:
                interaction = abs(fl) - abs(fh)
                if mk == 'h':
                    human_interaction = interaction
                print(f"  {interaction:>+10.1f}", end='')
            else:
                print(f"  {'N/A':>10}", end='')
        print()

        # Check: interaction positive means freq effect larger for unpredictable
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


# --------------------------------------------------------------------------- #
#  Summary table
# --------------------------------------------------------------------------- #

def print_summary(all_effects_results, has_bert):
    """Print a final summary showing which effects each model reproduces."""
    W = 100
    print(f"\n\n{'=' * W}")
    print(f"  SUMMARY: WHICH EFFECTS DOES EACH MODEL REPRODUCE?")
    print(f"{'=' * W}")
    print(f"  (Criteria: correct direction AND >= 25% of human effect size)")
    print(f"  For interaction: freq effect must be larger for unpredictable words")
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

    # Count passes per model
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
            rate = pass_counts[mk] / total * 100
            row += f"  {rate:>9.0f}%"
        else:
            row += f"  {'---':>10}"
    print(row)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    output_path = os.path.join(os.path.dirname(__file__), 'effects_results.txt')
    sys.stdout = Logger(output_path)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    et_csv = os.path.join(data_dir, 'Provo_Corpus-Eyetracking_Data.csv')
    lstm_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'checkpoints_v1/provo_lstm', 'best_model_lstm.pt')
    bert_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'checkpoints_v1/provo_bert', 'best_model_bert.pt')

    device = torch.device('cpu')

    W = 100

    print(f"{'=' * W}")
    print(f"  PSYCHOLINGUISTIC EFFECT ANALYSIS")
    print(f"  Testing whether models reproduce known experimental effects")
    print(f"{'=' * W}")

    # ---- Load data (ALL sentences) ----
    print("\nLoading data...")
    raw = load_provo(et_csv)
    all_sentences = aggregate_by_sentence(raw, min_participants=10)
    subtlex = load_subtlexus(os.path.join(data_dir, 'SUBTLEXus.txt'))
    cf_labels = load_content_function_labels(et_csv)
    print(f"  Sentences: {len(all_sentences)}")
    print(f"  Words: {sum(len(s) for s in all_sentences)}")
    print(f"  SUBTLEXus: {len(subtlex):,} entries")
    print(f"  Content/Function labels: {len(cf_labels)} sentences")

    # ---- Load models ----
    print("\nLoading models...")

    # LSTM
    ckpt_lstm = torch.load(lstm_checkpoint, map_location=device, weights_only=False)
    vocab = ckpt_lstm['vocab']
    lstm_model = NeuralEZReader(vocab_size=len(vocab)).to(device)
    lstm_model.load_state_dict(ckpt_lstm['model_state_dict'], strict=False)
    lstm_model.eval()
    print("  LSTM: loaded")

    # BERT
    try:
        ckpt_bert = torch.load(bert_checkpoint, map_location=device, weights_only=False)
        bert_model = NeuralEZReaderBERT(
            bert_model_name=ckpt_bert.get('bert_model_name', 'bert-base-uncased'),
            freeze_bert_layers=ckpt_bert.get('freeze_bert_layers', 8)
        ).to(device)
        bert_model.load_state_dict(ckpt_bert['model_state_dict'])
        bert_model.eval()
        print("  BERT: loaded")
    except Exception as e:
        print(f"  BERT: SKIPPED ({e})")
        bert_model = None

    # Differentiable EZ
    diff_ezr = DifferentiableEZReader()
    diff_ezr.eval()
    print("  Diff EZ: loaded (untrained formula)")

    has_bert = bert_model is not None

    # ---- Run all models ----
    print(f"\nRunning all models on {len(all_sentences)} sentences...")
    t0 = time.time()
    words = collect_all_data(
        all_sentences, subtlex, diff_ezr, lstm_model, vocab,
        bert_model, device, cf_labels)
    elapsed = time.time() - t0
    print(f"  Done: {len(words)} words in {elapsed:.1f}s")

    # ---- Print data summary ----
    log_freqs = [w['log_freq'] for w in words]
    lo, hi = compute_tertile_boundaries(log_freqs)
    print(f"\n  Frequency tertile boundaries: log10(freq) = {lo:.2f}, {hi:.2f}")
    print(f"    (freq = {10**lo:.0f}, {10**hi:.0f})")

    pred_zero = sum(1 for w in words if w['pred'] == 0.0)
    pred_low = sum(1 for w in words if 0 < w['pred'] <= 0.3)
    pred_high = sum(1 for w in words if w['pred'] > 0.3)
    print(f"  Predictability bins: Zero={pred_zero}, Low={pred_low}, High={pred_high}")

    cf_content = sum(1 for w in words if w['cf'] == 'Content')
    cf_func = sum(1 for w in words if w['cf'] == 'Function')
    cf_na = sum(1 for w in words if w['cf'] not in ('Content', 'Function'))
    print(f"  Content/Function: Content={cf_content}, Function={cf_func}, NA={cf_na}")

    # ---- Sanity check: human data means ----
    print(f"\n  Human data sanity check:")
    print(f"    Mean FFD = {np.mean([w['h_ffd'] for w in words]):.1f} ms")
    print(f"    Mean Gaze = {np.mean([w['h_gaze'] for w in words]):.1f} ms")
    print(f"    Mean TRT = {np.mean([w['h_trt'] for w in words]):.1f} ms")
    print(f"    Mean Skip = {np.mean([w['h_skip'] for w in words]):.3f}")

    # ================================================================
    #  EFFECT 1: FREQUENCY
    # ================================================================
    all_effects = []

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
        all_effects.append(
            (("Frequency", measure), freq_results))

    # ================================================================
    #  EFFECT 2: PREDICTABILITY
    # ================================================================
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
        all_effects.append(
            (("Predictability", measure), pred_results))

    # ================================================================
    #  EFFECT 3: WORD LENGTH
    # ================================================================
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
        all_effects.append(
            (("Word Length", measure), wlen_results))

    # ================================================================
    #  EFFECT 4: FREQ x PRED INTERACTION
    # ================================================================
    print(f"\n\n{'=' * W}")
    print(f"  EFFECT 4: FREQUENCY x PREDICTABILITY INTERACTION")
    print(f"{'=' * W}")

    fxp_bins = bin_freq_x_pred(words)
    for bname, bwords in fxp_bins.items():
        print(f"  {bname}: N={len(bwords)}")

    interaction_results = analyze_interaction(fxp_bins, has_bert)

    for measure in ['ffd', 'gaze', 'trt']:
        all_effects.append(
            (("Freq x Pred", measure), interaction_results))

    # ================================================================
    #  EFFECT 5: CONTENT vs FUNCTION
    # ================================================================
    print(f"\n\n{'=' * W}")
    print(f"  EFFECT 5: CONTENT vs FUNCTION WORDS")
    print(f"{'=' * W}")

    cf_bins = bin_content_function(words)
    for bname, bwords in cf_bins.items():
        print(f"  {bname}: N={len(bwords)}")

    # Function words should have shorter fixations and more skipping
    # Bins are ordered Content, Function — so direction from Content to Function:
    # FFD, gaze, TRT should decrease; skip should increase
    cf_results = analyze_effect(
        "CONTENT vs FUNCTION", cf_bins,
        {'ffd': 'decrease', 'gaze': 'decrease', 'trt': 'decrease', 'skip': 'increase'},
        has_bert)

    for measure in MEASURES:
        all_effects.append(
            (("Content/Function", measure), cf_results))

    # ================================================================
    #  SUMMARY
    # ================================================================
    print_summary(all_effects, has_bert)

    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
