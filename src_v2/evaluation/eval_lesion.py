"""
Lesion Study: Test what the trained full model actually relies on.

Unlike the ablation study (which retrained from scratch for each variant),
this loads the SINGLE trained full model and surgically disables components
at inference time. If a lesion hurts performance, that component was
actively contributing in the trained model.

Lesions:
  Representation lesions (modify neural network outputs):
    - const_L1:    Replace L1 with per-sentence mean (destroy word-specific L1)
    - const_L2:    Replace L2 with per-sentence mean (destroy word-specific L2)
    - const_skip:  Replace skip with per-sentence mean (destroy word-specific skip)
    - swap_L1_L2:  Swap L1 and L2 signals (test if decomposition is meaningful)
    - shuffle_L1:  Randomly permute L1 within sentence (destroy word-order info)
    - shuffle_L2:  Randomly permute L2 within sentence

  Parameter lesions (modify EZ Reader parameters in-place):
    - zero_ecc:    Set eccentricity parameter to 0
    - no_regr:     Disable regression mechanism
    - no_l2_ffd:   Set L2 contribution to FFD to 0

Usage:
    python3 -u src_v2/eval_lesion.py
    python3 -u src_v2/eval_lesion.py --corpus provo
    python3 -u src_v2/eval_lesion.py --corpus geco
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
#  Lesion definitions
# --------------------------------------------------------------------------- #

REPRESENTATION_LESIONS = [
    'const_L1',
    'const_L2',
    'const_skip',
    'swap_L1_L2',
    'shuffle_L1',
    'shuffle_L2',
]

PARAMETER_LESIONS = [
    'zero_ecc',
    'no_regr',
    'no_l2_ffd',
]

ALL_LESIONS = REPRESENTATION_LESIONS + PARAMETER_LESIONS

LESION_LABELS = {
    'full':        'Full Model',
    'const_L1':    'Const L1',
    'const_L2':    'Const L2',
    'const_skip':  'Const Skip',
    'swap_L1_L2':  'Swap L1/L2',
    'shuffle_L1':  'Shuffle L1',
    'shuffle_L2':  'Shuffle L2',
    'zero_ecc':    'Zero Ecc',
    'no_regr':     'No Regr',
    'no_l2_ffd':   'No L2→FFD',
}

ALL_KEYS = ['full'] + ALL_LESIONS
ALL_LABELS = [LESION_LABELS[k] for k in ALL_KEYS]

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


# --------------------------------------------------------------------------- #
#  Apply lesion to L1/L2/skip, re-run EZ Reader
# --------------------------------------------------------------------------- #

def apply_lesion(lesion, L1, L2, skip_prob, word_lengths, ezreader):
    """Apply a representation lesion and return modified EZ Reader output.

    For parameter lesions, we temporarily modify EZ Reader params, run forward,
    then restore them.
    """
    # Clone to avoid modifying originals
    L1_mod = L1.clone()
    L2_mod = L2.clone()
    skip_mod = skip_prob.clone()

    # --- Representation lesions ---
    if lesion == 'const_L1':
        # Replace each word's L1 with the sentence mean
        for b in range(L1_mod.size(0)):
            L1_mod[b] = L1_mod[b].mean()

    elif lesion == 'const_L2':
        for b in range(L2_mod.size(0)):
            L2_mod[b] = L2_mod[b].mean()

    elif lesion == 'const_skip':
        for b in range(skip_mod.size(0)):
            skip_mod[b] = skip_mod[b].mean()

    elif lesion == 'swap_L1_L2':
        L1_mod, L2_mod = L2.clone(), L1.clone()

    elif lesion == 'shuffle_L1':
        for b in range(L1_mod.size(0)):
            perm = torch.randperm(L1_mod.size(1))
            L1_mod[b] = L1_mod[b, perm]

    elif lesion == 'shuffle_L2':
        for b in range(L2_mod.size(0)):
            perm = torch.randperm(L2_mod.size(1))
            L2_mod[b] = L2_mod[b, perm]

    # --- Parameter lesions: save, modify, run, restore ---
    elif lesion == 'zero_ecc':
        saved = ezreader.eccentricity.data.clone()
        ezreader.eccentricity.data.fill_(0.0)
        result = ezreader(L1_mod, L2_mod, skip_mod, word_lengths, input_is_prob=True)
        ezreader.eccentricity.data.copy_(saved)
        result['L1'] = L1
        result['L2'] = L2
        return result

    elif lesion == 'no_regr':
        saved = ezreader.ablation
        ezreader.ablation = 'no_regressions'
        result = ezreader(L1_mod, L2_mod, skip_mod, word_lengths, input_is_prob=True)
        ezreader.ablation = saved
        result['L1'] = L1
        result['L2'] = L2
        return result

    elif lesion == 'no_l2_ffd':
        saved = ezreader.ablation
        ezreader.ablation = 'ffd_l1_only'
        result = ezreader(L1_mod, L2_mod, skip_mod, word_lengths, input_is_prob=True)
        ezreader.ablation = saved
        result['L1'] = L1
        result['L2'] = L2
        return result

    # Run EZ Reader with modified representations
    result = ezreader(L1_mod, L2_mod, skip_mod, word_lengths, input_is_prob=True)
    result['L1'] = L1_mod
    result['L2'] = L2_mod
    return result


# --------------------------------------------------------------------------- #
#  Model loading
# --------------------------------------------------------------------------- #

def load_full_model(ckpt_dir, device):
    model_short = "TinyLlama_TinyLlama-1.1B-Chat-v1.0"
    ckpt_path = os.path.join(ckpt_dir, f"geco_{model_short}", "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = ckpt.get('model_name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    freeze_layers = ckpt.get('freeze_layers', 16)

    model = NeuralEZReaderLLaMA(
        model_name=model_name,
        freeze_layers=freeze_layers,
        hidden_dim=256,
        ablation=None,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    print(f"  Full model loaded (epoch {epoch})")

    # Print learned EZ Reader parameters
    ez = model.ezreader
    print(f"\n  Learned EZ Reader parameters:")
    print(f"    saccade_time:       {ez.saccade_time.item():.1f} ms")
    print(f"    attention_shift:    {ez.attention_shift.item():.1f} ms")
    print(f"    eccentricity:       {ez.eccentricity.item():.4f}")
    print(f"    skip_sharpness:     {ez.skip_sharpness.item():.2f}")
    print(f"    l2_contribution:    {ez.l2_contribution.item():.4f} (softplus: {torch.nn.functional.softplus(ez.l2_contribution).item():.4f})")
    print(f"    regression_thresh:  {ez.regression_threshold.item():.1f} ms")
    print(f"    regression_sharp:   {ez.regression_sharpness.item():.4f}")
    print(f"    regression_cost:    {ez.regression_cost_scale.item():.4f} (softplus: {torch.nn.functional.softplus(ez.regression_cost_scale).item():.4f})")
    print(f"    l1_scale:           {model.l1_scale.item():.2f}")
    print(f"    l2_scale:           {model.l2_scale.item():.2f}")

    return model


# --------------------------------------------------------------------------- #
#  Collect predictions with all lesions
# --------------------------------------------------------------------------- #

def collect_predictions(sentences, subtlex, model, device):
    """Run full model once per sentence, then apply each lesion to the same
    L1/L2/skip outputs and re-run just the EZ Reader."""
    words = []
    total = len(sentences)

    # Fix random seed for shuffle reproducibility
    torch.manual_seed(42)

    for s_idx, agg in enumerate(sentences):
        if (s_idx + 1) % 50 == 0:
            print(f"  Processing sentence {s_idx+1}/{total}...", flush=True)

        tokens = agg.tokens
        preds = agg.predictabilities
        wlens = [len(t) for t in tokens]

        pred_tensor = torch.tensor([preds], dtype=torch.float32).to(device)
        wlen_tensor = torch.tensor([wlens], dtype=torch.float32).to(device)

        with torch.no_grad():
            # --- Run full model once to get L1, L2, skip ---
            base_result = model([tokens], pred_tensor, wlen_tensor)
            L1 = base_result['L1']
            L2 = base_result['L2']
            skip_prob = base_result['skip_prob']

            # --- Apply each lesion ---
            lesion_results = {}
            for lesion in ALL_LESIONS:
                lesion_results[lesion] = apply_lesion(
                    lesion, L1, L2, skip_prob, wlen_tensor, model.ezreader
                )

        for i in range(len(tokens)):
            freq = get_real_frequency(tokens[i], subtlex)
            w = {
                'token': tokens[i],
                'freq': freq,
                'log_freq': math.log10(max(1, freq)),
                'pred': preds[i],
                'wlen': wlens[i],
                'h_ffd': agg.mean_ffd[i],
                'h_gaze': agg.mean_gaze[i],
                'h_trt': agg.mean_trt[i],
                'h_skip': agg.skip_rate[i],
            }

            # Full model predictions
            w['full_ffd'] = base_result['first_fixation'][0, i].cpu().item()
            w['full_gaze'] = base_result['gaze_duration'][0, i].cpu().item()
            w['full_trt'] = base_result['total_reading_time'][0, i].cpu().item()
            w['full_skip'] = base_result['skip_prob'][0, i].cpu().item()
            w['full_l1'] = L1[0, i].cpu().item()
            w['full_l2'] = L2[0, i].cpu().item()

            # Lesion predictions
            for lesion in ALL_LESIONS:
                r = lesion_results[lesion]
                w[f'{lesion}_ffd'] = r['first_fixation'][0, i].cpu().item()
                w[f'{lesion}_gaze'] = r['gaze_duration'][0, i].cpu().item()
                w[f'{lesion}_trt'] = r['total_reading_time'][0, i].cpu().item()
                w[f'{lesion}_skip'] = r['skip_prob'][0, i].cpu().item()
                w[f'{lesion}_l1'] = r['L1'][0, i].cpu().item()
                w[f'{lesion}_l2'] = r['L2'][0, i].cpu().item()

            words.append(w)

    return words


# --------------------------------------------------------------------------- #
#  Metrics
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


def word_level_mae(words, model_key, measure):
    h_key = f'h_{measure}'
    m_key = f'{model_key}_{measure}'
    diffs = [abs(w[h_key] - w[m_key]) for w in words if w.get(m_key) is not None]
    return np.mean(diffs) if diffs else 0.0


# --------------------------------------------------------------------------- #
#  Printing
# --------------------------------------------------------------------------- #

def print_section(title):
    print(f"\n{'=' * 130}")
    print(f"  {title}")
    print(f"{'=' * 130}")


def print_correlation_table(words, title="WORD-LEVEL CORRELATIONS (Pearson r with human data)"):
    print_section(title)

    header = f"  {'Measure':<12}"
    for k in ALL_KEYS:
        header += f" {LESION_LABELS[k]:>12}"
    print(header)
    print(f"  {'-' * (12 + 13 * len(ALL_KEYS))}")

    for measure, mname in zip(MEASURES, MEASURE_NAMES):
        row = f"  {mname:<12}"
        full_r = word_level_corr(words, 'full', measure)
        row += f" {full_r:>12.3f}"
        for k in ALL_LESIONS:
            r = word_level_corr(words, k, measure)
            delta = r - full_r
            row += f" {r:>6.3f}{delta:>+6.3f}"
        print(row)


def print_delta_table(words):
    """Show the drop from full model for each lesion — the core result."""
    print_section("PERFORMANCE DROP FROM FULL MODEL (negative = lesion hurts)")

    header = f"  {'Measure':<12} {'Full r':>8}"
    for k in ALL_LESIONS:
        header += f" {LESION_LABELS[k]:>12}"
    print(header)
    print(f"  {'-' * (22 + 13 * len(ALL_LESIONS))}")

    for measure, mname in zip(MEASURES, MEASURE_NAMES):
        full_r = word_level_corr(words, 'full', measure)
        row = f"  {mname:<12} {full_r:>8.3f}"
        for k in ALL_LESIONS:
            r = word_level_corr(words, k, measure)
            delta = r - full_r
            row += f" {delta:>+12.3f}"
        print(row)


def print_prediction_divergence(words):
    """Mean absolute difference between lesioned and full model predictions."""
    print_section("PREDICTION DIVERGENCE FROM FULL MODEL (mean abs difference)")

    header = f"  {'Measure':<12}"
    for k in ALL_LESIONS:
        header += f" {LESION_LABELS[k]:>12}"
    print(header)
    print(f"  {'-' * (12 + 13 * len(ALL_LESIONS))}")

    for measure in ['ffd', 'gaze', 'trt', 'skip']:
        mname = {'ffd': 'FFD (ms)', 'gaze': 'Gaze (ms)', 'trt': 'TRT (ms)', 'skip': 'Skip'}[measure]
        row = f"  {mname:<12}"
        for k in ALL_LESIONS:
            diffs = [abs(w[f'full_{measure}'] - w[f'{k}_{measure}']) for w in words]
            row += f" {np.mean(diffs):>12.1f}" if measure != 'skip' else f" {np.mean(diffs):>12.3f}"
        print(row)


def print_l1_l2_analysis(words):
    """Check what happens to L1/L2 distributions under lesions."""
    print_section("L1/L2 STATISTICS UNDER LESIONS")

    header = f"  {'Statistic':<20}"
    for k in ['full'] + REPRESENTATION_LESIONS:
        header += f" {LESION_LABELS[k]:>12}"
    print(header)
    print(f"  {'-' * (20 + 13 * (1 + len(REPRESENTATION_LESIONS)))}")

    for stat_name, key_prefix, func in [
        ('Mean L1 (ms)', 'l1', np.mean),
        ('Std L1', 'l1', np.std),
        ('Mean L2 (ms)', 'l2', np.mean),
        ('Std L2', 'l2', np.std),
    ]:
        row = f"  {stat_name:<20}"
        for k in ['full'] + REPRESENTATION_LESIONS:
            vals = [w[f'{k}_{key_prefix}'] for w in words]
            row += f" {func(vals):>12.1f}"
        print(row)

    # r(L1, L2) for each condition
    row = f"  {'r(L1, L2)':<20}"
    for k in ['full'] + REPRESENTATION_LESIONS:
        l1_vals = [w[f'{k}_l1'] for w in words]
        l2_vals = [w[f'{k}_l2'] for w in words]
        row += f" {pearson_r(l1_vals, l2_vals):>12.3f}"
    print(row)


def print_grand_summary(words):
    """Compact summary table."""
    print_section("GRAND SUMMARY")

    header = f"  {'':>16}"
    for k in ALL_KEYS:
        header += f" {LESION_LABELS[k]:>12}"
    print(header)
    print(f"  {'-' * (16 + 13 * len(ALL_KEYS))}")

    for measure, mname in zip(MEASURES, MEASURE_NAMES):
        row = f"  {'r_' + mname:<16}"
        for k in ALL_KEYS:
            r = word_level_corr(words, k, measure)
            row += f" {r:>12.3f}"
        print(row)

    # Highlight biggest drops
    print(f"\n  BIGGEST DROPS (which lesion hurts most for each measure):")
    for measure, mname in zip(MEASURES, MEASURE_NAMES):
        full_r = word_level_corr(words, 'full', measure)
        worst_k = None
        worst_drop = 0
        for k in ALL_LESIONS:
            delta = word_level_corr(words, k, measure) - full_r
            if delta < worst_drop:
                worst_drop = delta
                worst_k = k
        if worst_k:
            print(f"    {mname:<12}: {LESION_LABELS[worst_k]:<15} (Δr = {worst_drop:+.3f})")
        else:
            print(f"    {mname:<12}: No lesion hurts")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', choices=['geco', 'provo', 'both'], default='both')
    args = parser.parse_args()

    results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'eval_lesion_results.txt')
    sys.stdout = Logger(results_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'v2')
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    # Load frequency data
    subtlex = load_subtlexus(os.path.join(data_dir, 'SUBTLEXus.txt'))
    print(f"SUBTLEXus: {len(subtlex)} entries")

    # Load model
    print(f"\nLoading full model...")
    model = load_full_model(ckpt_dir, device)

    # --- GECO test set ---
    if args.corpus in ('geco', 'both'):
        print(f"\n{'#' * 130}")
        print(f"#  GECO TEST SET (in-distribution)")
        print(f"{'#' * 130}")

        geco_mat = os.path.join(data_dir, 'Geco_EnglishMaterial.csv')
        geco_rd = os.path.join(data_dir, 'Geco_MonolingualReadingData.csv')
        geco_pred = os.path.join(data_dir, 'geco_predictability.pkl')
        geco_raw = load_geco(geco_rd, geco_mat, geco_pred)
        train_raw, val_raw, _ = split_geco(geco_raw)
        geco_agg = aggregate_by_sentence(geco_raw, min_participants=5)
        train_ids = set(sd.text_id for sd in train_raw)
        val_ids = set(sd.text_id for sd in val_raw)
        geco_test = [a for a in geco_agg if a.text_id not in train_ids
                     and a.text_id not in val_ids]
        print(f"  GECO test: {len(geco_test)} sentences, "
              f"{sum(len(a.tokens) for a in geco_test)} words")

        t0 = time.time()
        geco_words = collect_predictions(geco_test, subtlex, model, device)
        print(f"\n  Collected {len(geco_words)} word predictions in {time.time()-t0:.1f}s")

        print_correlation_table(geco_words)
        print_delta_table(geco_words)
        print_prediction_divergence(geco_words)
        print_l1_l2_analysis(geco_words)
        print_grand_summary(geco_words)

    # --- Provo (cross-corpus) ---
    if args.corpus in ('provo', 'both'):
        print(f"\n{'#' * 130}")
        print(f"#  PROVO CORPUS (cross-corpus generalization)")
        print(f"{'#' * 130}")

        provo_et = os.path.join(data_dir, 'Provo_Corpus-Eyetracking_Data.csv')
        raw = load_provo(provo_et)
        provo_sents = aggregate_by_sentence(raw, min_participants=10)
        print(f"  Provo: {len(provo_sents)} sentences")

        t0 = time.time()
        provo_words = collect_predictions(provo_sents, subtlex, model, device)
        print(f"\n  Collected {len(provo_words)} word predictions in {time.time()-t0:.1f}s")

        print_correlation_table(provo_words)
        print_delta_table(provo_words)
        print_prediction_divergence(provo_words)
        print_l1_l2_analysis(provo_words)
        print_grand_summary(provo_words)

    print(f"\n\nDone! Results saved to: {os.path.abspath(results_path)}")


if __name__ == '__main__':
    main()
