"""
Ablation Study Evaluation Pipeline.

Loads the full LLaMA+Diff model and all 5 ablation variants, evaluates them
on both GECO (test set) and Provo (cross-corpus) using:
  1. Word-level correlations (r for TRT, FFD, Gaze, Skip)
  2. Bin-level evaluation (frequency, predictability, length bins)
  3. Psycholinguistic effects (16 tests: direction + magnitude)

The goal is to determine which E-Z Reader components are necessary for
capturing human reading patterns.

Ablations:
  - no_two_stage:    Single processing time (no L1/L2 decomposition)
  - no_eccentricity: Remove word-length scaling of L1
  - no_regressions:  Remove regression mechanism from TRT
  - skip_from_l1:    Derive skip from L1 instead of separate head
  - ffd_l1_only:     FFD = L1 only (no L2 contribution)

Usage:
    python3 -u src_v2/eval_ablations.py
    python3 -u src_v2/eval_ablations.py --corpus provo
    python3 -u src_v2/eval_ablations.py --corpus geco
    python3 -u src_v2/eval_ablations.py --corpus both
"""

import os
import sys
import csv
import math
import time
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lm_model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'archive', 'original_ezreader'))

from model_llama import NeuralEZReaderLLaMA
from data_loader import load_provo, aggregate_by_sentence
from geco_loader import load_geco, split_geco


# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #

ABLATION_NAMES = [
    'no_two_stage',
    'no_eccentricity',
    'no_regressions',
    'skip_from_l1',
    'ffd_l1_only',
]

ABLATION_LABELS = {
    None:               'Full Model',
    'no_two_stage':     'No Two-Stage',
    'no_eccentricity':  'No Eccentricity',
    'no_regressions':   'No Regressions',
    'skip_from_l1':     'Skip from L1',
    'ffd_l1_only':      'FFD L1 Only',
}

# All model keys: full + 5 ablations
ALL_KEYS = ['full'] + ABLATION_NAMES
ALL_LABELS = [ABLATION_LABELS[None]] + [ABLATION_LABELS[a] for a in ABLATION_NAMES]

MEASURES = ['ffd', 'gaze', 'trt', 'skip']
MEASURE_NAMES = ['FFD (ms)', 'Gaze (ms)', 'TRT (ms)', 'Skip Rate']


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
#  Load content/function labels (Provo only)
# --------------------------------------------------------------------------- #

def load_content_function_labels(et_csv):
    """Load content/function word labels from Provo eye-tracking CSV."""
    cf = {}
    with open(et_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                text_id = int(row['Text_ID'])
                sent_num = int(row['Sentence_Number'])
                word_num = int(row['Word_Number'])
            except (ValueError, TypeError):
                continue
            label = row.get('Content_Or_Function_Word', 'NA')
            key = (text_id, sent_num)
            if key not in cf:
                cf[key] = {}
            cf[key][word_num] = label
    return cf


# --------------------------------------------------------------------------- #
#  Model loading
# --------------------------------------------------------------------------- #

def load_model(checkpoint_path, ablation, device):
    """Load a LLaMA+Diff model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get('model_name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    freeze_layers = ckpt.get('freeze_layers', 16)

    model = NeuralEZReaderLLaMA(
        model_name=model_name,
        freeze_layers=freeze_layers,
        hidden_dim=256,
        ablation=ablation,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model, ckpt.get('epoch', '?')


def load_all_models(ckpt_dir, device):
    """Load full model + all available ablation models."""
    models = {}
    model_short = "TinyLlama_TinyLlama-1.1B-Chat-v1.0"

    # Full model (no ablation)
    full_path = os.path.join(ckpt_dir, f"geco_{model_short}", "best_model.pt")
    try:
        model, epoch = load_model(full_path, ablation=None, device=device)
        models['full'] = model
        print(f"  Full Model: loaded (epoch {epoch})")
    except Exception as e:
        print(f"  Full Model: SKIPPED ({e})")

    # Ablation models
    for abl in ABLATION_NAMES:
        abl_path = os.path.join(ckpt_dir, f"geco_{model_short}_ablation_{abl}", "best_model.pt")
        try:
            model, epoch = load_model(abl_path, ablation=abl, device=device)
            models[abl] = model
            print(f"  {ABLATION_LABELS[abl]}: loaded (epoch {epoch})")
        except Exception as e:
            print(f"  {ABLATION_LABELS[abl]}: SKIPPED ({e})")

    return models


# --------------------------------------------------------------------------- #
#  Collect per-word predictions
# --------------------------------------------------------------------------- #

def collect_predictions(sentences, subtlex, models, device, cf_labels=None):
    """Run all models on sentences, return list of per-word dicts."""
    words = []
    total = len(sentences)
    active_keys = [k for k in ALL_KEYS if k in models]

    for s_idx, agg in enumerate(sentences):
        if (s_idx + 1) % 50 == 0:
            print(f"  Processing sentence {s_idx+1}/{total}...", flush=True)

        tokens = agg.tokens
        preds = agg.predictabilities
        wlens = [len(t) for t in tokens]

        # Run all models
        model_results = {}
        with torch.no_grad():
            for mk in active_keys:
                model_results[mk] = models[mk](
                    [tokens],
                    torch.tensor([preds], dtype=torch.float32).to(device),
                    torch.tensor([wlens], dtype=torch.float32).to(device),
                )

        # Content/Function labels
        cf_key = (agg.text_id, agg.sentence_number) if cf_labels else None
        cf_words = cf_labels.get(cf_key, {}) if cf_labels else {}

        for i in range(len(tokens)):
            freq = get_real_frequency(tokens[i], subtlex)
            w = {
                'token': tokens[i],
                'freq': freq,
                'log_freq': math.log10(max(1, freq)),
                'pred': preds[i],
                'wlen': wlens[i],
                'cf': cf_words.get(i + 1, 'NA'),
                # Human
                'h_ffd': agg.mean_ffd[i],
                'h_gaze': agg.mean_gaze[i],
                'h_trt': agg.mean_trt[i],
                'h_skip': agg.skip_rate[i],
            }
            # Model predictions
            for mk in active_keys:
                r = model_results[mk]
                w[f'{mk}_ffd'] = r['first_fixation'][0, i].cpu().item()
                w[f'{mk}_gaze'] = r['gaze_duration'][0, i].cpu().item()
                w[f'{mk}_trt'] = r['total_reading_time'][0, i].cpu().item()
                w[f'{mk}_skip'] = r['skip_prob'][0, i].cpu().item()
                w[f'{mk}_l1'] = r['L1'][0, i].cpu().item()
                w[f'{mk}_l2'] = r['L2'][0, i].cpu().item()

            words.append(w)

    return words, active_keys


# --------------------------------------------------------------------------- #
#  Word-level correlations
# --------------------------------------------------------------------------- #

def pearson_r(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return np.corrcoef(a, b)[0, 1]


def word_level_corr(words, model_key, measure):
    h_key = f'h_{measure}'
    m_key = f'{model_key}_{measure}'
    h_vals = [w[h_key] for w in words if w.get(m_key) is not None]
    m_vals = [w[m_key] for w in words if w.get(m_key) is not None]
    return pearson_r(h_vals, m_vals)


def print_word_level_table(words, active_keys):
    """Print word-level correlation table."""
    active_labels = [ABLATION_LABELS[k if k != 'full' else None] for k in active_keys]

    print(f"\n{'=' * 120}")
    print(f"  WORD-LEVEL CORRELATIONS")
    print(f"{'=' * 120}")

    header = f"  {'Measure':<16}"
    for label in active_labels:
        header += f" {label:>15}"
    print(header)
    print(f"  {'-' * (16 + 16 * len(active_keys))}")

    for measure, mname in zip(MEASURES, MEASURE_NAMES):
        row = f"  {mname:<16}"
        for mk in active_keys:
            r = word_level_corr(words, mk, measure)
            row += f" {r:>15.3f}"
        print(row)


# --------------------------------------------------------------------------- #
#  Bin-level evaluation
# --------------------------------------------------------------------------- #

def make_tertile_bins(words, key, labels=None):
    """Bin words into 3 equal-sized groups."""
    vals = [w[key] for w in words]
    p33, p66 = np.percentile(vals, [33.3, 66.7])
    bins = {0: [], 1: [], 2: []}
    for w in words:
        v = w[key]
        if v <= p33:
            bins[0].append(w)
        elif v <= p66:
            bins[1].append(w)
        else:
            bins[2].append(w)
    if labels is None:
        labels = ['Low', 'Med', 'High']
    return {labels[i]: bins[i] for i in range(3)}


def bin_frequency(words):
    return make_tertile_bins(words, 'log_freq', ['Low Freq', 'Med Freq', 'High Freq'])


def bin_predictability(words):
    return make_tertile_bins(words, 'pred', ['Low Pred', 'Med Pred', 'High Pred'])


def bin_word_length(words):
    bins = {'Short (1-3)': [], 'Med (4-6)': [], 'Long (7+)': []}
    for w in words:
        wl = w['wlen']
        if wl <= 3:    bins['Short (1-3)'].append(w)
        elif wl <= 6:  bins['Med (4-6)'].append(w)
        else:          bins['Long (7+)'].append(w)
    return bins


def bin_mean(word_list, prefix, measure):
    key = f'{prefix}_{measure}'
    vals = [w[key] for w in word_list if w.get(key) is not None]
    return np.mean(vals) if vals else None


def compute_bin_metrics(bins, measure, active_keys):
    """Compute bin means, RMSD, and r across bins for each model."""
    bin_names = list(bins.keys())
    h_means = np.array([bin_mean(bins[b], 'h', measure) for b in bin_names], dtype=float)

    results = {}
    for mk in active_keys:
        m_means = [bin_mean(bins[b], mk, measure) for b in bin_names]
        if any(v is None for v in m_means):
            results[mk] = {'means': [None] * len(bin_names), 'rmsd': None, 'r': None}
            continue
        m_means = np.array(m_means, dtype=float)
        rmsd = np.sqrt(np.mean((h_means - m_means) ** 2))
        if np.std(m_means) > 0 and np.std(h_means) > 0:
            r = np.corrcoef(h_means, m_means)[0, 1]
        else:
            r = 0.0
        results[mk] = {'means': m_means.tolist(), 'rmsd': rmsd, 'r': r}

    results['human'] = {'means': h_means.tolist()}
    return results


def print_bin_summary(words, active_keys):
    """Print bin-level RMSD and correlation summary."""
    active_labels = [ABLATION_LABELS[k if k != 'full' else None] for k in active_keys]

    all_binnings = [
        ("Freq", bin_frequency(words)),
        ("Pred", bin_predictability(words)),
        ("Length", bin_word_length(words)),
    ]

    measures_summary = [('ffd', 'FFD'), ('trt', 'TRT'), ('gaze', 'Gaze'), ('skip', 'Skip')]

    # RMSD table
    print(f"\n{'=' * 120}")
    print(f"  BIN-LEVEL RMSD (lower is better)")
    print(f"{'=' * 120}")

    header = f"  {'Binning / Measure':<24}"
    for label in active_labels:
        header += f" {label:>15}"
    print(header)
    print(f"  {'-' * (24 + 16 * len(active_keys))}")

    for bin_label, bins in all_binnings:
        for measure, m_label in measures_summary:
            metrics = compute_bin_metrics(bins, measure, active_keys)
            row = f"  {bin_label + ' / ' + m_label:<24}"
            for mk in active_keys:
                rmsd = metrics[mk].get('rmsd')
                if rmsd is None:
                    row += f" {'N/A':>15}"
                elif measure == 'skip':
                    row += f" {rmsd:>15.3f}"
                else:
                    row += f" {rmsd:>15.1f}"
            print(row)

    # Correlation table
    print(f"\n{'=' * 120}")
    print(f"  BIN-LEVEL CORRELATION (higher is better)")
    print(f"{'=' * 120}")

    header = f"  {'Binning / Measure':<24}"
    for label in active_labels:
        header += f" {label:>15}"
    print(header)
    print(f"  {'-' * (24 + 16 * len(active_keys))}")

    model_corrs = {mk: [] for mk in active_keys}

    for bin_label, bins in all_binnings:
        for measure, m_label in measures_summary:
            metrics = compute_bin_metrics(bins, measure, active_keys)
            row = f"  {bin_label + ' / ' + m_label:<24}"
            for mk in active_keys:
                r = metrics[mk].get('r')
                if r is None:
                    row += f" {'N/A':>15}"
                else:
                    row += f" {r:>15.3f}"
                    model_corrs[mk].append(r)
            print(row)

    print(f"  {'-' * (24 + 16 * len(active_keys))}")
    row = f"  {'MEAN':<24}"
    for mk in active_keys:
        vals = model_corrs[mk]
        row += f" {np.mean(vals):>15.3f}" if vals else f" {'N/A':>15}"
    print(row)


# --------------------------------------------------------------------------- #
#  Psycholinguistic effects
# --------------------------------------------------------------------------- #

def compute_effect(bins, model_key, measure):
    """Compute effect size: last bin mean - first bin mean."""
    bin_names = list(bins.keys())
    first = bin_mean(bins[bin_names[0]], model_key, measure)
    last = bin_mean(bins[bin_names[-1]], model_key, measure)
    if first is None or last is None:
        return None
    return last - first


def analyze_effect(human_effect, model_effect):
    """
    Evaluate if model captures the effect.
    Returns (pass/fail/na, magnitude_pct).
    Pass = correct direction AND >= 25% of human effect size.
    """
    if human_effect is None or model_effect is None:
        return 'N/A', None
    if abs(human_effect) < 1e-6:
        return 'N/A', None

    direction_correct = (model_effect > 0) == (human_effect > 0)
    magnitude_pct = abs(model_effect) / abs(human_effect) * 100

    if direction_correct and magnitude_pct >= 25:
        return 'PASS', magnitude_pct
    else:
        return 'FAIL', magnitude_pct


def evaluate_interaction(words, model_key, measure):
    """
    Frequency x Predictability interaction.
    Expected: frequency effect larger for low-predictability words.
    """
    vals_freq = [w['log_freq'] for w in words]
    vals_pred = [w['pred'] for w in words]
    med_freq = np.median(vals_freq)
    med_pred = np.median(vals_pred)

    cells = {
        'lf_lp': [], 'lf_hp': [],
        'hf_lp': [], 'hf_hp': [],
    }
    for w in words:
        f_low = w['log_freq'] <= med_freq
        p_low = w['pred'] <= med_pred
        if f_low and p_low:     cells['lf_lp'].append(w)
        elif f_low and not p_low: cells['lf_hp'].append(w)
        elif not f_low and p_low: cells['hf_lp'].append(w)
        else:                    cells['hf_hp'].append(w)

    mk = model_key
    m_key = f'{mk}_{measure}'
    h_key = f'h_{measure}'

    def cell_mean(cell, key):
        vals = [w[key] for w in cell if w.get(key) is not None]
        return np.mean(vals) if vals else None

    # Frequency effect at low pred
    hf_lp_h = cell_mean(cells['hf_lp'], h_key)
    lf_lp_h = cell_mean(cells['lf_lp'], h_key)
    hf_lp_m = cell_mean(cells['hf_lp'], m_key)
    lf_lp_m = cell_mean(cells['lf_lp'], m_key)

    # Frequency effect at high pred
    hf_hp_h = cell_mean(cells['hf_hp'], h_key)
    lf_hp_h = cell_mean(cells['lf_hp'], h_key)
    hf_hp_m = cell_mean(cells['hf_hp'], m_key)
    lf_hp_m = cell_mean(cells['lf_hp'], m_key)

    if any(v is None for v in [hf_lp_h, lf_lp_h, hf_hp_h, lf_hp_h,
                                hf_lp_m, lf_lp_m, hf_hp_m, lf_hp_m]):
        return 'N/A'

    # Human interaction
    h_freq_eff_low_pred = lf_lp_h - hf_lp_h  # effect at low pred
    h_freq_eff_high_pred = lf_hp_h - hf_hp_h  # effect at high pred
    h_interaction = abs(h_freq_eff_low_pred) - abs(h_freq_eff_high_pred)

    # Model interaction
    m_freq_eff_low_pred = lf_lp_m - hf_lp_m
    m_freq_eff_high_pred = lf_hp_m - hf_hp_m
    m_interaction = abs(m_freq_eff_low_pred) - abs(m_freq_eff_high_pred)

    # Pass if interaction > 0 (larger freq effect at low pred)
    if h_interaction > 0 and m_interaction > 0:
        return 'PASS'
    elif h_interaction <= 0:
        return 'N/A'
    else:
        return 'FAIL'


def run_effects_analysis(words, active_keys, has_cf=False):
    """Run all psycholinguistic effect tests and print results."""
    active_labels = [ABLATION_LABELS[k if k != 'full' else None] for k in active_keys]

    print(f"\n{'=' * 120}")
    print(f"  PSYCHOLINGUISTIC EFFECTS (direction + magnitude >= 25%)")
    print(f"{'=' * 120}")

    # Build effects list
    freq_bins = bin_frequency(words)
    pred_bins = bin_predictability(words)
    len_bins = bin_word_length(words)

    effects = []

    # Frequency effects (higher freq -> lower reading time, higher skip)
    for measure, mname in [('ffd', 'FFD'), ('gaze', 'Gaze'), ('trt', 'TRT')]:
        effects.append(('Freq -> ' + mname, freq_bins, measure, 'decrease'))
    effects.append(('Freq -> Skip', freq_bins, 'skip', 'increase'))

    # Predictability effects
    for measure, mname in [('ffd', 'FFD'), ('gaze', 'Gaze'), ('trt', 'TRT')]:
        effects.append(('Pred -> ' + mname, pred_bins, measure, 'decrease'))
    effects.append(('Pred -> Skip', pred_bins, 'skip', 'increase'))

    # Word length effects (longer -> slower, lower skip)
    for measure, mname in [('ffd', 'FFD'), ('gaze', 'Gaze'), ('trt', 'TRT')]:
        effects.append(('Length -> ' + mname, len_bins, measure, 'increase'))
    effects.append(('Length -> Skip', len_bins, 'skip', 'decrease'))

    # Content/Function (Provo only)
    if has_cf:
        content_words = [w for w in words if w.get('cf') == 'Content']
        function_words = [w for w in words if w.get('cf') == 'Function']
        if content_words and function_words:
            cf_bins = {'Function': function_words, 'Content': content_words}
            for measure, mname in [('ffd', 'FFD'), ('gaze', 'Gaze'), ('trt', 'TRT')]:
                effects.append(('C/F -> ' + mname, cf_bins, measure, 'decrease'))
            effects.append(('C/F -> Skip', cf_bins, 'skip', 'increase'))

    # Print effect table
    header = f"  {'Effect':<20} {'Human':>8}"
    for label in active_labels:
        header += f" {label:>15}"
    print(header)
    print(f"  {'-' * (28 + 16 * len(active_keys))}")

    # Track pass/fail per model
    model_pass = {mk: 0 for mk in active_keys}
    model_fail = {mk: 0 for mk in active_keys}
    model_na = {mk: 0 for mk in active_keys}

    for effect_name, bins, measure, expected_dir in effects:
        h_effect = compute_effect(bins, 'h', measure)

        row = f"  {effect_name:<20}"
        if h_effect is not None:
            row += f" {h_effect:>+8.1f}" if measure != 'skip' else f" {h_effect:>+8.3f}"
        else:
            row += f" {'N/A':>8}"

        for mk in active_keys:
            m_effect = compute_effect(bins, mk, measure)
            result, mag = analyze_effect(h_effect, m_effect)

            if result == 'PASS':
                model_pass[mk] += 1
                if m_effect is not None:
                    val_str = f"{m_effect:+.1f}" if measure != 'skip' else f"{m_effect:+.3f}"
                    row += f" {val_str:>12} OK"
                else:
                    row += f" {'PASS':>15}"
            elif result == 'FAIL':
                model_fail[mk] += 1
                if m_effect is not None:
                    val_str = f"{m_effect:+.1f}" if measure != 'skip' else f"{m_effect:+.3f}"
                    row += f" {val_str:>11} !!!"
                else:
                    row += f" {'FAIL':>15}"
            else:
                model_na[mk] += 1
                row += f" {'N/A':>15}"

        print(row)

    # Interaction effects
    print(f"\n  {'Interaction':<20} {'':>8}", end="")
    for _ in active_keys:
        print(f" {'':>15}", end="")
    print()

    interaction_measures = [('ffd', 'FFD'), ('gaze', 'Gaze'), ('trt', 'TRT')]
    for measure, mname in interaction_measures:
        row = f"  {'FxP -> ' + mname:<20} {'':>8}"
        for mk in active_keys:
            result = evaluate_interaction(words, mk, measure)
            if result == 'PASS':
                model_pass[mk] += 1
                row += f" {'PASS':>15}"
            elif result == 'FAIL':
                model_fail[mk] += 1
                row += f" {'FAIL':>15}"
            else:
                model_na[mk] += 1
                row += f" {'N/A':>15}"
        print(row)

    # Summary
    print(f"\n  {'-' * (28 + 16 * len(active_keys))}")
    print(f"  SUMMARY")
    row_pass = f"  {'PASS':<20} {'':>8}"
    row_fail = f"  {'FAIL':<20} {'':>8}"
    row_total = f"  {'Pass Rate':<20} {'':>8}"

    for mk in active_keys:
        total_tests = model_pass[mk] + model_fail[mk]
        pct = model_pass[mk] / total_tests * 100 if total_tests > 0 else 0
        row_pass += f" {model_pass[mk]:>15d}"
        row_fail += f" {model_fail[mk]:>15d}"
        row_total += f" {pct:>14.0f}%"

    print(row_pass)
    print(row_fail)
    print(row_total)

    return model_pass, model_fail


# --------------------------------------------------------------------------- #
#  Learned EZ Reader parameters
# --------------------------------------------------------------------------- #

def print_ezreader_params_table(models, active_keys):
    """Print a comparison table of learned EZ Reader parameters."""
    active_labels = [ABLATION_LABELS[k if k != 'full' else None] for k in active_keys]

    print(f"\n{'=' * 120}")
    print(f"  LEARNED EZ READER PARAMETERS")
    print(f"{'=' * 120}")

    params = [
        ('saccade_time', 'Saccade time (ms)', 150.0),
        ('attention_shift', 'Attn shift (ms)', 25.0),
        ('skip_sharpness', 'Skip sharpness', 8.0),
        ('eccentricity', 'Eccentricity', 0.1),
        ('l2_contribution', 'L2 contribution', 0.3),
        ('regression_threshold', 'Regr threshold', 50.0),
        ('regression_sharpness', 'Regr sharpness', 0.1),
        ('regression_cost_scale', 'Regr cost scale', 1.0),
    ]

    header = f"  {'Parameter':<22} {'Init':>8}"
    for label in active_labels:
        header += f" {label:>15}"
    print(header)
    print(f"  {'-' * (30 + 16 * len(active_keys))}")

    for param_name, label, init_val in params:
        row = f"  {label:<22} {init_val:>8.1f}"
        for mk in active_keys:
            if mk not in models:
                row += f" {'N/A':>15}"
                continue
            ezr = models[mk].ezreader
            param = getattr(ezr, param_name, None)
            if param is not None:
                row += f" {param.item():>15.3f}"
            else:
                row += f" {'N/A':>15}"
        print(row)

    # Also print L1/L2 scales from the model itself
    row = f"  {'L1 scale':<22} {'50.0':>8}"
    for mk in active_keys:
        if mk not in models:
            row += f" {'N/A':>15}"
        else:
            row += f" {models[mk].l1_scale.item():>15.3f}"
    print(row)

    row = f"  {'L2 scale':<22} {'30.0':>8}"
    for mk in active_keys:
        if mk not in models:
            row += f" {'N/A':>15}"
        else:
            row += f" {models[mk].l2_scale.item():>15.3f}"
    print(row)


# --------------------------------------------------------------------------- #
#  DETAILED ANALYSIS: L1/L2 distributions and theoretical consistency
# --------------------------------------------------------------------------- #

def print_l1_l2_analysis(words, active_keys):
    """Analyze L1/L2 distributions, ratios, and theoretical consistency."""
    active_labels = [ABLATION_LABELS[k if k != 'full' else None] for k in active_keys]

    print(f"\n{'=' * 120}")
    print(f"  L1/L2 DISTRIBUTION ANALYSIS")
    print(f"  (E-Z Reader theory: L2 = delta * L1, delta ≈ 0.34)")
    print(f"{'=' * 120}")

    header = f"  {'Statistic':<24}"
    for label in active_labels:
        header += f" {label:>15}"
    print(header)
    print(f"  {'-' * (24 + 16 * len(active_keys))}")

    stats = [
        ('Mean L1 (ms)', lambda w, mk: w[f'{mk}_l1']),
        ('Std L1', lambda w, mk: w[f'{mk}_l1']),
        ('Mean L2 (ms)', lambda w, mk: w[f'{mk}_l2']),
        ('Std L2', lambda w, mk: w[f'{mk}_l2']),
    ]

    for mk_label, stat_name, fn in [
        ('mean', 'Mean L1 (ms)', lambda w, mk: w[f'{mk}_l1']),
        ('std',  'Std L1', lambda w, mk: w[f'{mk}_l1']),
        ('mean', 'Mean L2 (ms)', lambda w, mk: w[f'{mk}_l2']),
        ('std',  'Std L2', lambda w, mk: w[f'{mk}_l2']),
    ]:
        row = f"  {stat_name:<24}"
        for mk in active_keys:
            vals = [fn(w, mk) for w in words]
            if mk_label == 'mean':
                row += f" {np.mean(vals):>15.1f}"
            else:
                row += f" {np.std(vals):>15.1f}"
        print(row)

    # L2/L1 ratio (should be ~0.34 per theory)
    row = f"  {'Mean L2/L1 ratio':<24}"
    for mk in active_keys:
        ratios = [w[f'{mk}_l2'] / max(w[f'{mk}_l1'], 0.1) for w in words]
        row += f" {np.mean(ratios):>15.3f}"
    print(row)

    row = f"  {'Median L2/L1 ratio':<24}"
    for mk in active_keys:
        ratios = [w[f'{mk}_l2'] / max(w[f'{mk}_l1'], 0.1) for w in words]
        row += f" {np.median(ratios):>15.3f}"
    print(row)

    # L1 range
    row = f"  {'L1 min':<24}"
    for mk in active_keys:
        row += f" {min(w[f'{mk}_l1'] for w in words):>15.1f}"
    print(row)
    row = f"  {'L1 max':<24}"
    for mk in active_keys:
        row += f" {max(w[f'{mk}_l1'] for w in words):>15.1f}"
    print(row)

    # L2 range
    row = f"  {'L2 min':<24}"
    for mk in active_keys:
        row += f" {min(w[f'{mk}_l2'] for w in words):>15.1f}"
    print(row)
    row = f"  {'L2 max':<24}"
    for mk in active_keys:
        row += f" {max(w[f'{mk}_l2'] for w in words):>15.1f}"
    print(row)

    # L1 variance / L2 variance (measures whether two-stage provides differentiation)
    row = f"  {'L1 CV (std/mean)':<24}"
    for mk in active_keys:
        vals = [w[f'{mk}_l1'] for w in words]
        cv = np.std(vals) / max(np.mean(vals), 0.1)
        row += f" {cv:>15.3f}"
    print(row)

    row = f"  {'L2 CV (std/mean)':<24}"
    for mk in active_keys:
        vals = [w[f'{mk}_l2'] for w in words]
        cv = np.std(vals) / max(np.mean(vals), 0.1)
        row += f" {cv:>15.3f}"
    print(row)

    # L1-L2 correlation (should be moderate, not 1.0 — independent processes)
    row = f"  {'r(L1, L2)':<24}"
    for mk in active_keys:
        l1 = [w[f'{mk}_l1'] for w in words]
        l2 = [w[f'{mk}_l2'] for w in words]
        row += f" {pearson_r(l1, l2):>15.3f}"
    print(row)


def print_component_contributions(words, active_keys):
    """Analyze how much each EZ Reader component contributes to predictions."""
    active_labels = [ABLATION_LABELS[k if k != 'full' else None] for k in active_keys]

    print(f"\n{'=' * 120}")
    print(f"  COMPONENT CONTRIBUTION ANALYSIS")
    print(f"  (Derived measures showing what each component adds)")
    print(f"{'=' * 120}")

    header = f"  {'Measure':<28}"
    for label in active_labels:
        header += f" {label:>15}"
    print(header)
    print(f"  {'-' * (28 + 16 * len(active_keys))}")

    # Gaze - FFD gap = L2 contribution to gaze
    # In the model: Gaze = L1_scaled + L2, FFD = L1_scaled + softplus(l2_contrib) * L2
    # So Gaze - FFD measures how much L2 adds beyond FFD
    row = f"  {'Mean Gaze - FFD (ms)':<28}"
    for mk in active_keys:
        gaps = [w[f'{mk}_gaze'] - w[f'{mk}_ffd'] for w in words]
        row += f" {np.mean(gaps):>15.1f}"
    print(row)

    row = f"  {'Human Gaze - FFD (ms)':<28}"
    gaps = [w['h_gaze'] - w['h_ffd'] for w in words]
    print(f"  {'Human Gaze - FFD (ms)':<28} {np.mean(gaps):>15.1f}")

    # TRT - Gaze gap = regression + skip contribution
    # TRT = (1-skip) * (Gaze + overhead + regression_penalty)
    row = f"  {'Mean TRT - Gaze (ms)':<28}"
    for mk in active_keys:
        gaps = [w[f'{mk}_trt'] - w[f'{mk}_gaze'] for w in words]
        row += f" {np.mean(gaps):>15.1f}"
    print(row)

    row = f"  {'Human TRT - Gaze (ms)':<28}"
    gaps = [w['h_trt'] - w['h_gaze'] for w in words]
    print(f"  {'Human TRT - Gaze (ms)':<28} {np.mean(gaps):>15.1f}")

    # Mean skip probability
    row = f"  {'Mean skip prob':<28}"
    for mk in active_keys:
        row += f" {np.mean([w[f'{mk}_skip'] for w in words]):>15.3f}"
    print(row)
    print(f"  {'Human skip rate':<28} {np.mean([w['h_skip'] for w in words]):>15.3f}")

    # FFD/L1 ratio — how much of FFD comes from L1?
    row = f"  {'Mean FFD/L1 ratio':<28}"
    for mk in active_keys:
        ratios = [w[f'{mk}_ffd'] / max(w[f'{mk}_l1'], 0.1) for w in words]
        row += f" {np.mean(ratios):>15.3f}"
    print(row)

    # Gaze/(L1+L2) ratio — should be ~1.0 (gaze = L1_scaled + L2)
    row = f"  {'Mean Gaze/(L1+L2) ratio':<28}"
    for mk in active_keys:
        ratios = [w[f'{mk}_gaze'] / max(w[f'{mk}_l1'] + w[f'{mk}_l2'], 0.1) for w in words]
        row += f" {np.mean(ratios):>15.3f}"
    print(row)


def print_l1_l2_correlations_with_variables(words, active_keys):
    """Check if L1/L2 correlate with the right psycholinguistic variables."""
    active_labels = [ABLATION_LABELS[k if k != 'full' else None] for k in active_keys]

    print(f"\n{'=' * 120}")
    print(f"  L1/L2 CORRELATIONS WITH PSYCHOLINGUISTIC VARIABLES")
    print(f"  (Theory: L1 driven by frequency+length, L2 driven by frequency+predictability)")
    print(f"{'=' * 120}")

    header = f"  {'Correlation':<28}"
    for label in active_labels:
        header += f" {label:>15}"
    print(header)
    print(f"  {'-' * (28 + 16 * len(active_keys))}")

    variables = [
        ('r(L1, log_freq)', 'l1', 'log_freq', 'Negative expected'),
        ('r(L1, pred)', 'l1', 'pred', 'Negative expected'),
        ('r(L1, wlen)', 'l1', 'wlen', 'Positive expected (eccentricity)'),
        ('r(L2, log_freq)', 'l2', 'log_freq', 'Negative expected'),
        ('r(L2, pred)', 'l2', 'pred', 'Negative expected'),
        ('r(L2, wlen)', 'l2', 'wlen', 'Weak/none expected'),
    ]

    for label, l_key, var_key, note in variables:
        row = f"  {label:<28}"
        for mk in active_keys:
            l_vals = [w[f'{mk}_{l_key}'] for w in words]
            v_vals = [w[var_key] for w in words]
            row += f" {pearson_r(l_vals, v_vals):>15.3f}"
        print(row + f"  ({note})")


def print_word_length_stratified(words, active_keys):
    """Stratified analysis: performance on short vs long words."""
    active_labels = [ABLATION_LABELS[k if k != 'full' else None] for k in active_keys]

    print(f"\n{'=' * 120}")
    print(f"  WORD LENGTH STRATIFIED ANALYSIS")
    print(f"  (Tests which ablations hurt long words vs short words specifically)")
    print(f"{'=' * 120}")

    length_groups = [
        ('Short (1-3)', [w for w in words if w['wlen'] <= 3]),
        ('Medium (4-6)', [w for w in words if 4 <= w['wlen'] <= 6]),
        ('Long (7+)', [w for w in words if w['wlen'] >= 7]),
    ]

    for measure, mname in [('ffd', 'FFD'), ('trt', 'TRT'), ('gaze', 'Gaze'), ('skip', 'Skip')]:
        print(f"\n  {mname} correlation by word length:")
        header = f"  {'Length group':<20} {'N':>6}"
        for label in active_labels:
            header += f" {label:>15}"
        print(header)
        print(f"  {'-' * (26 + 16 * len(active_keys))}")

        for group_name, group_words in length_groups:
            row = f"  {group_name:<20} {len(group_words):>6}"
            for mk in active_keys:
                r = word_level_corr(group_words, mk, measure)
                row += f" {r:>15.3f}"
            print(row)


def print_frequency_stratified(words, active_keys):
    """Stratified analysis: performance on low vs high frequency words."""
    active_labels = [ABLATION_LABELS[k if k != 'full' else None] for k in active_keys]

    print(f"\n{'=' * 120}")
    print(f"  FREQUENCY STRATIFIED ANALYSIS")
    print(f"  (Tests which ablations hurt low-frequency words specifically)")
    print(f"{'=' * 120}")

    med_freq = np.median([w['log_freq'] for w in words])
    freq_groups = [
        ('Low frequency', [w for w in words if w['log_freq'] <= med_freq]),
        ('High frequency', [w for w in words if w['log_freq'] > med_freq]),
    ]

    for measure, mname in [('ffd', 'FFD'), ('trt', 'TRT'), ('gaze', 'Gaze'), ('skip', 'Skip')]:
        print(f"\n  {mname} correlation by frequency:")
        header = f"  {'Freq group':<20} {'N':>6}"
        for label in active_labels:
            header += f" {label:>15}"
        print(header)
        print(f"  {'-' * (26 + 16 * len(active_keys))}")

        for group_name, group_words in freq_groups:
            row = f"  {group_name:<20} {len(group_words):>6}"
            for mk in active_keys:
                r = word_level_corr(group_words, mk, measure)
                row += f" {r:>15.3f}"
            print(row)


def print_effect_magnitude_table(words, active_keys, has_cf=False):
    """Effect magnitude as % of human effect (not just pass/fail)."""
    active_labels = [ABLATION_LABELS[k if k != 'full' else None] for k in active_keys]

    print(f"\n{'=' * 120}")
    print(f"  EFFECT MAGNITUDE (% of human effect size)")
    print(f"  100% = matches human exactly. >100% = overshoots. <100% = undershoots.")
    print(f"{'=' * 120}")

    freq_bins = bin_frequency(words)
    pred_bins = bin_predictability(words)
    len_bins = bin_word_length(words)

    effects = [
        ('Freq -> FFD', freq_bins, 'ffd'),
        ('Freq -> Gaze', freq_bins, 'gaze'),
        ('Freq -> TRT', freq_bins, 'trt'),
        ('Freq -> Skip', freq_bins, 'skip'),
        ('Pred -> FFD', pred_bins, 'ffd'),
        ('Pred -> Gaze', pred_bins, 'gaze'),
        ('Pred -> TRT', pred_bins, 'trt'),
        ('Pred -> Skip', pred_bins, 'skip'),
        ('Length -> FFD', len_bins, 'ffd'),
        ('Length -> Gaze', len_bins, 'gaze'),
        ('Length -> TRT', len_bins, 'trt'),
        ('Length -> Skip', len_bins, 'skip'),
    ]

    if has_cf:
        content_words = [w for w in words if w.get('cf') == 'Content']
        function_words = [w for w in words if w.get('cf') == 'Function']
        if content_words and function_words:
            cf_bins = {'Function': function_words, 'Content': content_words}
            effects += [
                ('C/F -> FFD', cf_bins, 'ffd'),
                ('C/F -> Gaze', cf_bins, 'gaze'),
                ('C/F -> TRT', cf_bins, 'trt'),
                ('C/F -> Skip', cf_bins, 'skip'),
            ]

    header = f"  {'Effect':<20} {'Human':>10}"
    for label in active_labels:
        header += f" {label:>15}"
    print(header)
    print(f"  {'-' * (30 + 16 * len(active_keys))}")

    for effect_name, bins, measure in effects:
        h_effect = compute_effect(bins, 'h', measure)
        if h_effect is None or abs(h_effect) < 1e-6:
            continue
        fmt = '.3f' if measure == 'skip' else '.1f'
        row = f"  {effect_name:<20} {h_effect:>+10{fmt}}"
        for mk in active_keys:
            m_effect = compute_effect(bins, mk, measure)
            if m_effect is None:
                row += f" {'N/A':>15}"
            else:
                pct = m_effect / h_effect * 100
                row += f" {pct:>14.0f}%"
        print(row)

    # Mean magnitude
    print(f"  {'-' * (30 + 16 * len(active_keys))}")
    row = f"  {'MEAN MAGNITUDE':<20} {'':>10}"
    for mk in active_keys:
        pcts = []
        for _, bins, measure in effects:
            h_eff = compute_effect(bins, 'h', measure)
            m_eff = compute_effect(bins, mk, measure)
            if h_eff and m_eff and abs(h_eff) > 1e-6:
                pcts.append(m_eff / h_eff * 100)
        row += f" {np.mean(pcts):>14.0f}%" if pcts else f" {'N/A':>15}"
    print(row)


def print_ablation_specific_diagnostics(words, active_keys):
    """Targeted diagnostics for each specific ablation."""
    active_labels = [ABLATION_LABELS[k if k != 'full' else None] for k in active_keys]

    print(f"\n{'=' * 120}")
    print(f"  ABLATION-SPECIFIC DIAGNOSTICS")
    print(f"{'=' * 120}")

    # --- no_two_stage: Does removing separate L1/L2 hurt L1-L2 differentiation? ---
    if 'no_two_stage' in active_keys and 'full' in active_keys:
        print(f"\n  [no_two_stage] L1/L2 Differentiation Check:")
        print(f"  (With no_two_stage: L1=0.6*total, L2=0.4*total → L2/L1 fixed at 0.667)")

        for mk, label in [('full', 'Full'), ('no_two_stage', 'No Two-Stage')]:
            l1 = [w[f'{mk}_l1'] for w in words]
            l2 = [w[f'{mk}_l2'] for w in words]
            ratios = [l2[i] / max(l1[i], 0.1) for i in range(len(l1))]
            print(f"    {label:20s}: L2/L1 = {np.mean(ratios):.3f} ± {np.std(ratios):.3f}  "
                  f"(L1={np.mean(l1):.1f}±{np.std(l1):.1f}, L2={np.mean(l2):.1f}±{np.std(l2):.1f})")

        # Check if L1 and L2 are more correlated in no_two_stage (they should be r=1.0)
        l1_nts = [w['no_two_stage_l1'] for w in words]
        l2_nts = [w['no_two_stage_l2'] for w in words]
        l1_full = [w['full_l1'] for w in words]
        l2_full = [w['full_l2'] for w in words]
        print(f"    Full r(L1,L2)         = {pearson_r(l1_full, l2_full):.3f} (should be moderate)")
        print(f"    No Two-Stage r(L1,L2) = {pearson_r(l1_nts, l2_nts):.3f} (should be 1.000)")

    # --- no_eccentricity: Does length effect on FFD degrade? ---
    if 'no_eccentricity' in active_keys and 'full' in active_keys:
        print(f"\n  [no_eccentricity] Word Length Effect on FFD:")
        print(f"  (Eccentricity scales L1 by word length. Removing it should weaken length->FFD)")

        short = [w for w in words if w['wlen'] <= 3]
        long = [w for w in words if w['wlen'] >= 7]

        for mk, label in [('full', 'Full'), ('no_eccentricity', 'No Eccentricity')]:
            ffd_short = np.mean([w[f'{mk}_ffd'] for w in short])
            ffd_long = np.mean([w[f'{mk}_ffd'] for w in long])
            h_ffd_short = np.mean([w['h_ffd'] for w in short])
            h_ffd_long = np.mean([w['h_ffd'] for w in long])
            length_eff = ffd_long - ffd_short
            h_eff = h_ffd_long - h_ffd_short
            print(f"    {label:20s}: FFD(short)={ffd_short:.1f}, FFD(long)={ffd_long:.1f}, "
                  f"effect={length_eff:+.1f}ms (human={h_eff:+.1f}ms, {length_eff/h_eff*100:.0f}%)")

        # Also check L1 correlation with word length
        for mk, label in [('full', 'Full'), ('no_eccentricity', 'No Eccentricity')]:
            l1 = [w[f'{mk}_l1'] for w in words]
            wl = [w['wlen'] for w in words]
            print(f"    {label:20s}: r(L1, wlen) = {pearson_r(l1, wl):.3f}")

    # --- no_regressions: Does TRT lose regression-driven variance? ---
    if 'no_regressions' in active_keys and 'full' in active_keys:
        print(f"\n  [no_regressions] Regression Contribution Check:")
        print(f"  (TRT = fixate_prob * (Gaze + overhead + regression). Without regressions, TRT-Gaze shrinks)")

        for mk, label in [('full', 'Full'), ('no_regressions', 'No Regressions')]:
            trt_gaze_gaps = [w[f'{mk}_trt'] - w[f'{mk}_gaze'] for w in words]
            print(f"    {label:20s}: Mean TRT-Gaze = {np.mean(trt_gaze_gaps):.1f}ms "
                  f"(std={np.std(trt_gaze_gaps):.1f})")

        h_gaps = [w['h_trt'] - w['h_gaze'] for w in words]
        print(f"    {'Human':20s}: Mean TRT-Gaze = {np.mean(h_gaps):.1f}ms "
              f"(std={np.std(h_gaps):.1f})")

        # Does removing regressions hurt TRT for difficult (low-freq, low-pred) words?
        hard_words = [w for w in words if w['log_freq'] <= np.percentile([w2['log_freq'] for w2 in words], 25)
                      and w['pred'] <= np.percentile([w2['pred'] for w2 in words], 25)]
        if len(hard_words) > 10:
            print(f"\n    Difficult words (low freq + low pred, N={len(hard_words)}):")
            for mk, label in [('full', 'Full'), ('no_regressions', 'No Regressions')]:
                r_trt = word_level_corr(hard_words, mk, 'trt')
                print(f"      {label:20s}: r_TRT = {r_trt:.3f}")

    # --- skip_from_l1: Does skip lose independent predictability sensitivity? ---
    if 'skip_from_l1' in active_keys and 'full' in active_keys:
        print(f"\n  [skip_from_l1] Skip Probability Analysis:")
        print(f"  (Skip derived from L1 via sigmoid, instead of separate head)")

        for mk, label in [('full', 'Full'), ('skip_from_l1', 'Skip from L1')]:
            skip_vals = [w[f'{mk}_skip'] for w in words]
            print(f"    {label:20s}: Mean skip = {np.mean(skip_vals):.3f} "
                  f"(std={np.std(skip_vals):.3f}, min={min(skip_vals):.3f}, max={max(skip_vals):.3f})")

        print(f"    {'Human':20s}: Mean skip = {np.mean([w['h_skip'] for w in words]):.3f}")

        # Skip correlation with predictability (separate head should capture pred better)
        for mk, label in [('full', 'Full'), ('skip_from_l1', 'Skip from L1')]:
            skip = [w[f'{mk}_skip'] for w in words]
            pred = [w['pred'] for w in words]
            freq = [w['log_freq'] for w in words]
            wlen = [w['wlen'] for w in words]
            print(f"    {label:20s}: r(skip,pred)={pearson_r(skip,pred):.3f}  "
                  f"r(skip,freq)={pearson_r(skip,freq):.3f}  "
                  f"r(skip,wlen)={pearson_r(skip,wlen):.3f}")

    # --- ffd_l1_only: Does FFD lose sensitivity to L2-related processing? ---
    if 'ffd_l1_only' in active_keys and 'full' in active_keys:
        print(f"\n  [ffd_l1_only] FFD Sensitivity Check:")
        print(f"  (FFD = L1 only, no L2 contribution. Should lose predictability sensitivity in FFD)")

        for mk, label in [('full', 'Full'), ('ffd_l1_only', 'FFD L1 Only')]:
            ffd = [w[f'{mk}_ffd'] for w in words]
            pred = [w['pred'] for w in words]
            freq = [w['log_freq'] for w in words]
            print(f"    {label:20s}: r(FFD,pred)={pearson_r(ffd,pred):.3f}  "
                  f"r(FFD,freq)={pearson_r(ffd,freq):.3f}")

        # Check Gaze-FFD gap (with ffd_l1_only, FFD=L1 so gap = L2)
        for mk, label in [('full', 'Full'), ('ffd_l1_only', 'FFD L1 Only')]:
            gaps = [w[f'{mk}_gaze'] - w[f'{mk}_ffd'] for w in words]
            print(f"    {label:20s}: Mean Gaze-FFD = {np.mean(gaps):.1f}ms "
                  f"(this IS the L2 contribution to FFD in full model)")


def print_error_analysis(words, active_keys):
    """Where does each ablation make the WORST predictions vs full model?"""
    if 'full' not in active_keys:
        return

    active_labels = [ABLATION_LABELS[k if k != 'full' else None] for k in active_keys]

    print(f"\n{'=' * 120}")
    print(f"  ERROR ANALYSIS vs FULL MODEL")
    print(f"  (Mean absolute difference between ablation and full model predictions)")
    print(f"{'=' * 120}")

    header = f"  {'Measure':<20}"
    for label in active_labels:
        header += f" {label:>15}"
    print(header)
    print(f"  {'-' * (20 + 16 * len(active_keys))}")

    for measure, mname in [('ffd', 'FFD (ms)'), ('gaze', 'Gaze (ms)'),
                            ('trt', 'TRT (ms)'), ('skip', 'Skip'),
                            ('l1', 'L1 (ms)'), ('l2', 'L2 (ms)')]:
        row = f"  {mname:<20}"
        for mk in active_keys:
            if mk == 'full':
                row += f" {'---':>15}"
            else:
                diffs = [abs(w[f'{mk}_{measure}'] - w[f'full_{measure}']) for w in words]
                row += f" {np.mean(diffs):>15.1f}" if measure != 'skip' else f" {np.mean(diffs):>15.3f}"
        print(row)

    # Also show: which word types show the biggest divergence per ablation?
    abl_keys = [k for k in active_keys if k != 'full']
    for abl in abl_keys:
        label = ABLATION_LABELS[abl]
        # Find the measure with biggest divergence
        max_div_measure = None
        max_div = 0
        for measure in ['ffd', 'gaze', 'trt']:
            div = np.mean([abs(w[f'{abl}_{measure}'] - w[f'full_{measure}']) for w in words])
            if div > max_div:
                max_div = div
                max_div_measure = measure

        if max_div_measure:
            # Stratify by word length
            short = [w for w in words if w['wlen'] <= 3]
            long = [w for w in words if w['wlen'] >= 7]
            div_short = np.mean([abs(w[f'{abl}_{max_div_measure}'] - w[f'full_{max_div_measure}']) for w in short])
            div_long = np.mean([abs(w[f'{abl}_{max_div_measure}'] - w[f'full_{max_div_measure}']) for w in long])
            print(f"\n  {label}: Biggest divergence in {max_div_measure.upper()} — "
                  f"short words: {div_short:.1f}ms, long words: {div_long:.1f}ms")


# --------------------------------------------------------------------------- #
#  Grand summary table
# --------------------------------------------------------------------------- #

def print_grand_summary(words, active_keys, model_pass, model_fail):
    """Print a single summary table comparing all models."""
    active_labels = [ABLATION_LABELS[k if k != 'full' else None] for k in active_keys]

    print(f"\n{'=' * 120}")
    print(f"  GRAND SUMMARY")
    print(f"{'=' * 120}")

    header = f"  {'Metric':<24}"
    for label in active_labels:
        header += f" {label:>15}"
    print(header)
    print(f"  {'-' * (24 + 16 * len(active_keys))}")

    # Word-level correlations
    for measure, mname in zip(MEASURES, MEASURE_NAMES):
        row = f"  {'Word r_' + mname.split()[0]:<24}"
        for mk in active_keys:
            r = word_level_corr(words, mk, measure)
            row += f" {r:>15.3f}"
        print(row)

    print(f"  {'-' * (24 + 16 * len(active_keys))}")

    # Bin-level mean correlation
    all_binnings = [
        bin_frequency(words),
        bin_predictability(words),
        bin_word_length(words),
    ]
    row = f"  {'Bin r (mean)':24}"
    for mk in active_keys:
        corrs = []
        for bins in all_binnings:
            for measure in MEASURES:
                metrics = compute_bin_metrics(bins, measure, [mk])
                r = metrics[mk].get('r')
                if r is not None:
                    corrs.append(r)
        row += f" {np.mean(corrs):>15.3f}" if corrs else f" {'N/A':>15}"
    print(row)

    print(f"  {'-' * (24 + 16 * len(active_keys))}")

    # Effects pass rate
    row = f"  {'Effects pass rate':<24}"
    for mk in active_keys:
        total = model_pass[mk] + model_fail[mk]
        pct = model_pass[mk] / total * 100 if total > 0 else 0
        row += f" {f'{model_pass[mk]}/{total} ({pct:.0f}%)':>15}"
    print(row)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Ablation study evaluation")
    parser.add_argument("--corpus", type=str, default="both",
                        choices=["provo", "geco", "both"])
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
    print("\nLoading models...")
    models = load_all_models(ckpt_dir, device)

    if not models:
        print("ERROR: No models found. Exiting.")
        return

    active_keys = [k for k in ALL_KEYS if k in models]
    active_labels = [ABLATION_LABELS[k if k != 'full' else None] for k in active_keys]

    print(f"\n  Active models: {', '.join(active_labels)}")
    if len(active_keys) < 2:
        print("  WARNING: Need at least full model + 1 ablation for comparison.")

    # ---- Redirect stdout ----
    sys.stdout = Logger(os.path.join(results_dir, "eval_ablations_results.txt"))

    print(f"\n{'#' * 120}")
    print(f"#  ABLATION STUDY EVALUATION")
    print(f"#  Models: {', '.join(active_labels)}")
    print(f"#  Device: {device}")
    print(f"{'#' * 120}")

    # ---- Print learned parameters ----
    print_ezreader_params_table(models, active_keys)

    # ---- Evaluate on each corpus ----
    def run_on_corpus(sentences, corpus_name, cf_labels=None):
        print(f"\n{'#' * 120}")
        print(f"#  {corpus_name}")
        print(f"{'#' * 120}")

        t0 = time.time()
        words, act_keys = collect_predictions(sentences, subtlex, models, device,
                                              cf_labels=cf_labels)
        elapsed = time.time() - t0
        print(f"\n  Collected {len(words)} word predictions in {elapsed:.1f}s")

        has_cf = cf_labels is not None
        print_word_level_table(words, act_keys)
        print_bin_summary(words, act_keys)
        m_pass, m_fail = run_effects_analysis(words, act_keys, has_cf=has_cf)
        print_effect_magnitude_table(words, act_keys, has_cf=has_cf)
        print_l1_l2_analysis(words, act_keys)
        print_component_contributions(words, act_keys)
        print_l1_l2_correlations_with_variables(words, act_keys)
        print_word_length_stratified(words, act_keys)
        print_frequency_stratified(words, act_keys)
        print_ablation_specific_diagnostics(words, act_keys)
        print_error_analysis(words, act_keys)
        print_grand_summary(words, act_keys, m_pass, m_fail)

    if args.corpus in ("provo", "both"):
        print("\nLoading Provo Corpus...")
        et_path = os.path.join(data_dir, "Provo_Corpus-Eyetracking_Data.csv")
        provo_raw = load_provo(et_path)
        provo_agg = aggregate_by_sentence(provo_raw, min_participants=10)
        cf_labels = load_content_function_labels(et_path)
        print(f"  Provo: {len(provo_agg)} sentences, "
              f"{sum(len(a.tokens) for a in provo_agg)} words")
        run_on_corpus(provo_agg, "PROVO (cross-corpus)", cf_labels=cf_labels)

    if args.corpus in ("geco", "both"):
        print("\nLoading GECO Corpus (test set)...")
        reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
        material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
        pred_path = os.path.join(data_dir, "geco_predictability.pkl")
        geco_raw = load_geco(reading_path, material_path, pred_path)
        train_raw, val_raw, _ = split_geco(geco_raw)
        geco_agg = aggregate_by_sentence(geco_raw, min_participants=5)
        train_ids = set(sd.text_id for sd in train_raw)
        val_ids = set(sd.text_id for sd in val_raw)
        geco_test = [a for a in geco_agg if a.text_id not in train_ids
                     and a.text_id not in val_ids]
        print(f"  GECO test: {len(geco_test)} sentences, "
              f"{sum(len(a.tokens) for a in geco_test)} words")
        run_on_corpus(geco_test, "GECO TEST SET (in-distribution)")

    print(f"\n\nDone! Results saved to: {os.path.join(results_dir, 'eval_ablations_results.txt')}")


if __name__ == "__main__":
    main()
