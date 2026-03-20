"""
Model Comparison: LLaMA+EZR v2 vs LLaMA+EZR v3.

Evaluates both models on GECO test (in-distribution) and Provo (cross-corpus)
with comprehensive metrics:
  - Pearson correlation (pattern quality)
  - MAE (absolute accuracy, CMCL 2021 standard)
  - RMSE (penalizes large errors)
  - Mean bias (systematic over/under-prediction)
  - R-squared (variance explained)
  - Psycholinguistic effects (frequency, predictability, word length)
  - Learned EZR parameter comparison
  - Per-word sample predictions

Usage:
  python3 -u src_v2/eval_model_comparison.py
  python3 -u src_v2/eval_model_comparison.py --v2_ckpt path/to/v2.pt --v3_ckpt path/to/v3.pt
"""

import os
import sys
import math
import argparse
import csv
from collections import defaultdict

import torch
import numpy as np
from scipy import stats as sp_stats
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lm_model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'archive', 'original_ezreader'))

from model_llama import NeuralEZReaderLLaMA as NeuralEZReaderLLaMAv2
from model_llama_v3 import NeuralEZReaderLLaMAv3
from data_loader import load_provo, aggregate_by_sentence, split_aggregated
from geco_loader import load_geco, split_geco


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
#  Collation
# --------------------------------------------------------------------------- #

def collate_aggregated(batch, device):
    word_lists = [a.tokens for a in batch]
    pred_vals = pad_sequence(
        [torch.tensor(a.predictabilities, dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    wlens = pad_sequence(
        [torch.tensor([len(t) for t in a.tokens], dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    h_trt = pad_sequence(
        [torch.tensor(a.mean_trt, dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    h_ffd = pad_sequence(
        [torch.tensor(a.mean_ffd, dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    h_gaze = pad_sequence(
        [torch.tensor(a.mean_gaze, dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    h_skip = pad_sequence(
        [torch.tensor(a.skip_rate, dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    return word_lists, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip


# --------------------------------------------------------------------------- #
#  Collect per-word predictions from a model
# --------------------------------------------------------------------------- #

def collect_predictions(model, agg_data, device, batch_size=8):
    """Run model on data, return per-word predictions and human values."""
    model.eval()
    results = {
        'pred_trt': [], 'human_trt': [],
        'pred_ffd': [], 'human_ffd': [],
        'pred_gaze': [], 'human_gaze': [],
        'pred_skip': [], 'human_skip': [],
        'pred_l1': [], 'pred_l2': [],
        'words': [], 'word_lengths': [], 'predictabilities': [],
    }

    with torch.no_grad():
        for i in range(0, len(agg_data), batch_size):
            batch = agg_data[i:i + batch_size]
            word_lists, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip = collate_aggregated(batch, device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(word_lists, pred_vals, wlens)

            for b in range(len(batch)):
                seq_len = len(batch[b].tokens)
                results['pred_trt'].extend(pred['total_reading_time'][b, :seq_len].cpu().tolist())
                results['human_trt'].extend(batch[b].mean_trt)
                results['pred_ffd'].extend(pred['first_fixation'][b, :seq_len].cpu().tolist())
                results['human_ffd'].extend(batch[b].mean_ffd)
                results['pred_gaze'].extend(pred['gaze_duration'][b, :seq_len].cpu().tolist())
                results['human_gaze'].extend(batch[b].mean_gaze)
                results['pred_skip'].extend(pred['skip_prob'][b, :seq_len].cpu().tolist())
                results['human_skip'].extend(batch[b].skip_rate)
                results['pred_l1'].extend(pred['L1'][b, :seq_len].cpu().tolist())
                results['pred_l2'].extend(pred['L2'][b, :seq_len].cpu().tolist())
                results['words'].extend(batch[b].tokens)
                results['word_lengths'].extend([len(t) for t in batch[b].tokens])
                results['predictabilities'].extend(batch[b].predictabilities)

    # Convert to numpy
    for key in results:
        if key != 'words':
            results[key] = np.array(results[key])

    return results


def correct_for_skip(results):
    """
    Correct v2-style predictions for fair comparison against human data.

    The human aggregated data averages reading times ONLY over participants who
    fixated the word (skips excluded). But v2 computes:
        pred_TRT = (1-skip) * reading_time_if_fixated

    So we recover the conditional reading time:
        corrected_TRT = pred_TRT / (1 - pred_skip)

    This also applies to Gaze (which is used inside TRT via the same formula).
    FFD is NOT multiplied by skip in the model, so no correction needed.
    """
    skip = results['pred_skip']
    fixate_prob = np.clip(1.0 - skip, 0.05, 1.0)  # clamp to avoid division by ~0

    corrected = dict(results)  # shallow copy
    corrected['pred_trt'] = results['pred_trt'] / fixate_prob
    # Gaze in model output is NOT multiplied by skip (it's raw L1+L2),
    # but TRT = (1-skip) * (gaze + overhead + regression), so only TRT needs correction
    return corrected


# --------------------------------------------------------------------------- #
#  Compute all metrics
# --------------------------------------------------------------------------- #

def compute_metrics(pred, human, metric_name=""):
    """Compute correlation, MAE, RMSE, bias, R-squared for a single measure."""
    pred = np.array(pred)
    human = np.array(human)

    # Pearson r + p-value
    if len(pred) > 2 and np.std(pred) > 0 and np.std(human) > 0:
        r, p_val = sp_stats.pearsonr(pred, human)
    else:
        r, p_val = 0.0, 1.0

    mae = np.mean(np.abs(pred - human))
    rmse = np.sqrt(np.mean((pred - human) ** 2))
    bias = np.mean(pred) - np.mean(human)

    # R-squared
    ss_res = np.sum((human - pred) ** 2)
    ss_tot = np.sum((human - np.mean(human)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'r': r,
        'p_value': p_val,
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'r_squared': r_squared,
        'mean_pred': np.mean(pred),
        'mean_human': np.mean(human),
        'std_pred': np.std(pred),
        'std_human': np.std(human),
        'median_abs_err': np.median(np.abs(pred - human)),
        'pct_within_50ms': np.mean(np.abs(pred - human) < 50) * 100,
        'pct_within_100ms': np.mean(np.abs(pred - human) < 100) * 100,
    }


def print_metrics_table(metrics_dict, label):
    """Print a formatted table of metrics for one model on one dataset."""
    measures = ['TRT', 'FFD', 'Gaze', 'Skip']
    keys = ['trt', 'ffd', 'gaze', 'skip']

    print(f"\n{'─' * 95}")
    print(f"  {label}")
    print(f"{'─' * 95}")
    print(f"  {'Metric':<20s} {'TRT':>12s} {'FFD':>12s} {'Gaze':>12s} {'Skip':>12s}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")

    # Pearson r
    vals = [f"{metrics_dict[k]['r']:.3f}" for k in keys]
    print(f"  {'Pearson r':<20s} {vals[0]:>12s} {vals[1]:>12s} {vals[2]:>12s} {vals[3]:>12s}")

    # p-value significance
    sigs = []
    for k in keys:
        p = metrics_dict[k]['p_value']
        if p < 0.001:
            sigs.append("***")
        elif p < 0.01:
            sigs.append("**")
        elif p < 0.05:
            sigs.append("*")
        else:
            sigs.append("n.s.")
    print(f"  {'  significance':<20s} {sigs[0]:>12s} {sigs[1]:>12s} {sigs[2]:>12s} {sigs[3]:>12s}")

    # MAE
    units = ['ms', 'ms', 'ms', '']
    for row_name, metric_key in [('MAE', 'mae'), ('RMSE', 'rmse'), ('Bias (pred-human)', 'bias')]:
        vals = []
        for i, k in enumerate(keys):
            v = metrics_dict[k][metric_key]
            if k == 'skip':
                vals.append(f"{v:.4f}")
            else:
                vals.append(f"{v:.1f}ms")
        print(f"  {row_name:<20s} {vals[0]:>12s} {vals[1]:>12s} {vals[2]:>12s} {vals[3]:>12s}")

    # R-squared
    vals = [f"{metrics_dict[k]['r_squared']:.3f}" for k in keys]
    print(f"  {'R-squared':<20s} {vals[0]:>12s} {vals[1]:>12s} {vals[2]:>12s} {vals[3]:>12s}")

    # Median absolute error
    vals = []
    for k in keys:
        v = metrics_dict[k]['median_abs_err']
        vals.append(f"{v:.1f}ms" if k != 'skip' else f"{v:.4f}")
    print(f"  {'Median |error|':<20s} {vals[0]:>12s} {vals[1]:>12s} {vals[2]:>12s} {vals[3]:>12s}")

    # % within thresholds (only for time measures)
    vals = []
    for k in keys:
        if k == 'skip':
            vals.append("--")
        else:
            vals.append(f"{metrics_dict[k]['pct_within_50ms']:.1f}%")
    print(f"  {'% within 50ms':<20s} {vals[0]:>12s} {vals[1]:>12s} {vals[2]:>12s} {vals[3]:>12s}")

    vals = []
    for k in keys:
        if k == 'skip':
            vals.append("--")
        else:
            vals.append(f"{metrics_dict[k]['pct_within_100ms']:.1f}%")
    print(f"  {'% within 100ms':<20s} {vals[0]:>12s} {vals[1]:>12s} {vals[2]:>12s} {vals[3]:>12s}")

    # Mean predicted vs human
    print(f"\n  {'Distribution':<20s} {'TRT':>12s} {'FFD':>12s} {'Gaze':>12s} {'Skip':>12s}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    for row_name, metric_key in [('Mean (predicted)', 'mean_pred'), ('Mean (human)', 'mean_human'),
                                  ('Std (predicted)', 'std_pred'), ('Std (human)', 'std_human')]:
        vals = []
        for k in keys:
            v = metrics_dict[k][metric_key]
            vals.append(f"{v:.1f}ms" if k != 'skip' else f"{v:.3f}")
        print(f"  {row_name:<20s} {vals[0]:>12s} {vals[1]:>12s} {vals[2]:>12s} {vals[3]:>12s}")


# --------------------------------------------------------------------------- #
#  Psycholinguistic effects
# --------------------------------------------------------------------------- #

def check_effects(results, label):
    """Check frequency, predictability, and word length effects."""
    print(f"\n{'─' * 95}")
    print(f"  Psycholinguistic Effects — {label}")
    print(f"{'─' * 95}")

    words = results['words']
    pred_trt = results['pred_trt']
    pred_ffd = results['pred_ffd']
    pred_skip = results['pred_skip']
    human_trt = results['human_trt']
    human_ffd = results['human_ffd']
    human_skip = results['human_skip']
    wlens = results['word_lengths']
    preds = results['predictabilities']

    # --- Word length effect ---
    short_mask = wlens <= 4
    long_mask = wlens >= 7

    if short_mask.sum() > 10 and long_mask.sum() > 10:
        print(f"\n  Word Length Effect (short<=4 vs long>=7):")
        print(f"  {'Measure':<12s} {'Short(pred)':>12s} {'Long(pred)':>12s} {'Direction':>12s} "
              f"{'Short(human)':>12s} {'Long(human)':>12s} {'Direction':>12s} {'Match':>8s}")
        print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*8}")

        for name, p_arr, h_arr in [('TRT', pred_trt, human_trt), ('FFD', pred_ffd, human_ffd),
                                     ('Skip', pred_skip, human_skip)]:
            p_short, p_long = p_arr[short_mask].mean(), p_arr[long_mask].mean()
            h_short, h_long = h_arr[short_mask].mean(), h_arr[long_mask].mean()
            if name == 'Skip':
                p_dir = "short>long" if p_short > p_long else "short<long"
                h_dir = "short>long" if h_short > h_long else "short<long"
            else:
                p_dir = "short<long" if p_short < p_long else "short>long"
                h_dir = "short<long" if h_short < h_long else "short>long"
            match = "PASS" if p_dir == h_dir else "FAIL"
            fmt = ".3f" if name == 'Skip' else ".1f"
            print(f"  {name:<12s} {p_short:>12{fmt}} {p_long:>12{fmt}} {p_dir:>12s} "
                  f"{h_short:>12{fmt}} {h_long:>12{fmt}} {h_dir:>12s} {match:>8s}")

    # --- Predictability effect ---
    pred_thirds = np.percentile(preds, [33, 67])
    low_pred = preds <= pred_thirds[0]
    high_pred = preds >= pred_thirds[1]

    if low_pred.sum() > 10 and high_pred.sum() > 10:
        print(f"\n  Predictability Effect (low vs high, tertile split):")
        print(f"  {'Measure':<12s} {'Low(pred)':>12s} {'High(pred)':>12s} {'Direction':>12s} "
              f"{'Low(human)':>12s} {'High(human)':>12s} {'Direction':>12s} {'Match':>8s}")
        print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*8}")

        for name, p_arr, h_arr in [('TRT', pred_trt, human_trt), ('FFD', pred_ffd, human_ffd),
                                     ('Skip', pred_skip, human_skip)]:
            p_low, p_high = p_arr[low_pred].mean(), p_arr[high_pred].mean()
            h_low, h_high = h_arr[low_pred].mean(), h_arr[high_pred].mean()
            if name == 'Skip':
                p_dir = "low<high" if p_low < p_high else "low>high"
                h_dir = "low<high" if h_low < h_high else "low>high"
            else:
                p_dir = "low>high" if p_low > p_high else "low<high"
                h_dir = "low>high" if h_low > h_high else "low<high"
            match = "PASS" if p_dir == h_dir else "FAIL"
            fmt = ".3f" if name == 'Skip' else ".1f"
            print(f"  {name:<12s} {p_low:>12{fmt}} {p_high:>12{fmt}} {p_dir:>12s} "
                  f"{h_low:>12{fmt}} {h_high:>12{fmt}} {h_dir:>12s} {match:>8s}")

    # --- Frequency effect (using word length as proxy: short words tend to be more frequent) ---
    # We don't have direct frequency data in the aggregated format, but word length
    # serves as a reasonable proxy (r ~ 0.6 with log frequency)
    print(f"\n  Note: Frequency effect uses word length as proxy (corr ~0.6 with log freq)")


# --------------------------------------------------------------------------- #
#  EZR parameter comparison
# --------------------------------------------------------------------------- #

def print_ezr_params(model, label):
    """Print learned EZR parameters."""
    ezr = model.ezreader
    print(f"\n  {label} — Learned EZ Reader parameters:")
    print(f"    saccade_time          = {ezr.saccade_time.item():.1f}ms")
    print(f"    attention_shift       = {ezr.attention_shift.item():.1f}ms")
    print(f"    skip_sharpness        = {ezr.skip_sharpness.item():.2f}")
    print(f"    eccentricity          = {ezr.eccentricity.item():.4f}")
    print(f"    l2_contribution       = {ezr.l2_contribution.item():.4f}")
    print(f"    regression_threshold  = {ezr.regression_threshold.item():.1f}ms")
    print(f"    regression_sharpness  = {ezr.regression_sharpness.item():.4f}")
    print(f"    regression_cost_scale = {ezr.regression_cost_scale.item():.4f}")
    print(f"    l1_scale              = {model.l1_scale.item():.1f}")
    if hasattr(model, 'l2_scale'):
        print(f"    l2_scale              = {model.l2_scale.item():.1f}")
    if hasattr(model, '_delta_raw'):
        print(f"    delta (L2/L1)         = {model.delta.item():.4f}")


# --------------------------------------------------------------------------- #
#  L1/L2 distribution comparison
# --------------------------------------------------------------------------- #

def print_l1_l2_stats(results, label):
    """Print L1/L2 distribution statistics."""
    l1 = results['pred_l1']
    l2 = results['pred_l2']
    print(f"\n  {label} — L1/L2 distributions:")
    print(f"    L1: mean={l1.mean():.1f}ms  std={l1.std():.1f}ms  "
          f"min={l1.min():.1f}ms  max={l1.max():.1f}ms  median={np.median(l1):.1f}ms")
    print(f"    L2: mean={l2.mean():.1f}ms  std={l2.std():.1f}ms  "
          f"min={l2.min():.1f}ms  max={l2.max():.1f}ms  median={np.median(l2):.1f}ms")
    if len(l1) > 2:
        r_l1_l2 = np.corrcoef(l1, l2)[0, 1]
        ratio = l2.mean() / l1.mean() if l1.mean() > 0 else 0.0
        print(f"    L2/L1 ratio (mean): {ratio:.3f}  |  corr(L1, L2): {r_l1_l2:.3f}")


# --------------------------------------------------------------------------- #
#  Sample predictions
# --------------------------------------------------------------------------- #

def print_sample_predictions(model, agg_data, device, label, n_sentences=3, n_words=10):
    """Print per-word predictions for a few example sentences."""
    print(f"\n  {label} — Sample predictions:")
    model.eval()
    with torch.no_grad():
        for s_idx in range(min(n_sentences, len(agg_data))):
            s = agg_data[s_idx]
            word_list = [s.tokens]
            pv = torch.tensor(s.predictabilities, dtype=torch.float32).unsqueeze(0).to(device)
            wl = torch.tensor([len(t) for t in s.tokens], dtype=torch.float32).unsqueeze(0).to(device)
            p = model(word_list, pv, wl)

            title = ' '.join(s.tokens[:6]) + ('...' if len(s.tokens) > 6 else '')
            print(f"\n    \"{title}\"")
            print(f"    {'word':<14s} {'pTRT':>6s} {'hTRT':>6s} {'err':>6s} | "
                  f"{'pFFD':>6s} {'hFFD':>6s} | {'pSkip':>5s} {'hSkip':>5s}")
            print(f"    {'-'*72}")

            for i in range(min(n_words, len(s.tokens))):
                pt = p['total_reading_time'][0, i].item()
                pf = p['first_fixation'][0, i].item()
                ps = p['skip_prob'][0, i].item()
                ht = s.mean_trt[i]
                hf = s.mean_ffd[i]
                hs = s.skip_rate[i]
                err = pt - ht
                print(f"    {s.tokens[i]:<14s} {pt:6.0f} {ht:6.0f} {err:+6.0f} | "
                      f"{pf:6.0f} {hf:6.0f} | {ps:5.2f} {hs:5.2f}")


# --------------------------------------------------------------------------- #
#  Head-to-head comparison table
# --------------------------------------------------------------------------- #

def print_head_to_head(v2_metrics, v3_metrics, dataset_name):
    """Print a side-by-side comparison of v2 vs v3."""
    print(f"\n{'=' * 95}")
    print(f"  HEAD-TO-HEAD: v2 vs v3 on {dataset_name}")
    print(f"{'=' * 95}")
    measures = ['TRT', 'FFD', 'Gaze', 'Skip']
    keys = ['trt', 'ffd', 'gaze', 'skip']

    print(f"\n  {'Metric':<20s} {'v2':>10s} {'v3':>10s} {'Winner':>10s} {'Delta':>10s}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for name, k in zip(measures, keys):
        # Correlation (higher is better)
        v2_r = v2_metrics[k]['r']
        v3_r = v3_metrics[k]['r']
        winner = "v2" if v2_r > v3_r else "v3"
        delta = abs(v2_r - v3_r)
        print(f"  {'r_' + name:<20s} {v2_r:>10.3f} {v3_r:>10.3f} {winner:>10s} {delta:>10.3f}")

    print()
    for name, k in zip(measures, keys):
        # MAE (lower is better)
        v2_mae = v2_metrics[k]['mae']
        v3_mae = v3_metrics[k]['mae']
        winner = "v2" if v2_mae < v3_mae else "v3"
        delta = abs(v2_mae - v3_mae)
        if k == 'skip':
            print(f"  {'MAE_' + name:<20s} {v2_mae:>10.4f} {v3_mae:>10.4f} {winner:>10s} {delta:>10.4f}")
        else:
            print(f"  {'MAE_' + name:<20s} {v2_mae:>9.1f}ms {v3_mae:>9.1f}ms {winner:>10s} {delta:>9.1f}ms")

    print()
    for name, k in zip(measures, keys):
        # RMSE (lower is better)
        v2_rmse = v2_metrics[k]['rmse']
        v3_rmse = v3_metrics[k]['rmse']
        winner = "v2" if v2_rmse < v3_rmse else "v3"
        delta = abs(v2_rmse - v3_rmse)
        if k == 'skip':
            print(f"  {'RMSE_' + name:<20s} {v2_rmse:>10.4f} {v3_rmse:>10.4f} {winner:>10s} {delta:>10.4f}")
        else:
            print(f"  {'RMSE_' + name:<20s} {v2_rmse:>9.1f}ms {v3_rmse:>9.1f}ms {winner:>10s} {delta:>9.1f}ms")

    print()
    for name, k in zip(measures, keys):
        # Bias (closer to 0 is better)
        v2_bias = v2_metrics[k]['bias']
        v3_bias = v3_metrics[k]['bias']
        winner = "v2" if abs(v2_bias) < abs(v3_bias) else "v3"
        if k == 'skip':
            print(f"  {'Bias_' + name:<20s} {v2_bias:>+10.4f} {v3_bias:>+10.4f} {winner:>10s}")
        else:
            print(f"  {'Bias_' + name:<20s} {v2_bias:>+9.1f}ms {v3_bias:>+9.1f}ms {winner:>10s}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2_ckpt", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "..",
                                             "checkpoints", "v2", "geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0", "best_model.pt"))
    parser.add_argument("--v3_ckpt", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "..",
                                             "checkpoints", "v3", "geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0", "best_model.pt"))
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "results/eval_model_comparison_results.txt"))
    args = parser.parse_args()

    # Setup logging
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sys.stdout = Logger(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"v2 checkpoint: {args.v2_ckpt}")
    print(f"v3 checkpoint: {args.v3_ckpt}")

    # ---- Load data ----
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    print("\nLoading GECO corpus...")
    geco_raw = load_geco(
        os.path.join(data_dir, "Geco_MonolingualReadingData.csv"),
        os.path.join(data_dir, "Geco_EnglishMaterial.csv"),
        os.path.join(data_dir, "geco_predictability.pkl"),
    )
    train_raw, val_raw, test_raw = split_geco(geco_raw)
    aggregated = aggregate_by_sentence(geco_raw, min_participants=5)
    train_text_ids = set(sd.text_id for sd in train_raw)
    val_text_ids = set(sd.text_id for sd in val_raw)
    geco_test = [a for a in aggregated if a.text_id not in train_text_ids and a.text_id not in val_text_ids]
    print(f"  GECO test: {len(geco_test)} aggregated sentences")

    print("Loading Provo corpus...")
    provo_raw = load_provo(os.path.join(data_dir, "Provo_Corpus-Eyetracking_Data.csv"))
    provo_agg = aggregate_by_sentence(provo_raw, min_participants=5)
    print(f"  Provo: {len(provo_agg)} aggregated sentences (cross-corpus)")

    # ---- Load v2 model ----
    print(f"\nLoading v2 model...")
    v2_ckpt = torch.load(args.v2_ckpt, map_location=device, weights_only=False)
    v2_model_name = v2_ckpt.get('model_name', args.model)
    v2_freeze = v2_ckpt.get('freeze_layers', 12)
    v2_ablation = v2_ckpt.get('ablation', None)

    model_v2 = NeuralEZReaderLLaMAv2(
        model_name=v2_model_name, freeze_layers=v2_freeze, hidden_dim=256, ablation=v2_ablation
    ).to(device)
    model_v2.load_state_dict(v2_ckpt['model_state_dict'])
    model_v2.eval()
    print(f"  Loaded: {v2_model_name}, freeze={v2_freeze}, ablation={v2_ablation}")
    if 'val_metrics' in v2_ckpt:
        print(f"  Training val r_TRT: {v2_ckpt['val_metrics'].get('r_trt', 'N/A')}")

    # ---- Load v3 model ----
    print(f"\nLoading v3 model...")
    v3_ckpt = torch.load(args.v3_ckpt, map_location=device, weights_only=False)
    v3_model_name = v3_ckpt.get('model_name', args.model)
    v3_freeze = v3_ckpt.get('freeze_layers', 12)
    v3_ablation = v3_ckpt.get('ablation', None)

    model_v3 = NeuralEZReaderLLaMAv3(
        model_name=v3_model_name, freeze_layers=v3_freeze, hidden_dim=256, ablation=v3_ablation
    ).to(device)
    model_v3.load_state_dict(v3_ckpt['model_state_dict'])
    model_v3.eval()
    print(f"  Loaded: {v3_model_name}, freeze={v3_freeze}, ablation={v3_ablation}")
    if 'val_metrics' in v3_ckpt:
        print(f"  Training val r_TRT: {v3_ckpt['val_metrics'].get('r_trt', 'N/A')}")
    if 'delta' in v3_ckpt:
        print(f"  Learned delta: {v3_ckpt['delta']:.4f}")

    # ---- Evaluate on GECO test ----
    print("\n" + "=" * 95)
    print("  GECO TEST SET (in-distribution)")
    print("=" * 95)

    v2_geco = collect_predictions(model_v2, geco_test, device)
    v3_geco = collect_predictions(model_v3, geco_test, device)
    n_words_geco = len(v2_geco['pred_trt'])
    print(f"  Total words evaluated: {n_words_geco:,}")

    v2_geco_metrics = {
        'trt': compute_metrics(v2_geco['pred_trt'], v2_geco['human_trt']),
        'ffd': compute_metrics(v2_geco['pred_ffd'], v2_geco['human_ffd']),
        'gaze': compute_metrics(v2_geco['pred_gaze'], v2_geco['human_gaze']),
        'skip': compute_metrics(v2_geco['pred_skip'], v2_geco['human_skip']),
    }
    v3_geco_metrics = {
        'trt': compute_metrics(v3_geco['pred_trt'], v3_geco['human_trt']),
        'ffd': compute_metrics(v3_geco['pred_ffd'], v3_geco['human_ffd']),
        'gaze': compute_metrics(v3_geco['pred_gaze'], v3_geco['human_gaze']),
        'skip': compute_metrics(v3_geco['pred_skip'], v3_geco['human_skip']),
    }

    print_metrics_table(v2_geco_metrics, "LLaMA + DiffEZR v2 (independent L1/L2) — GECO test")
    print_metrics_table(v3_geco_metrics, "LLaMA + DiffEZR v3 (L2=delta*L1, conditional TRT) — GECO test")
    print_head_to_head(v2_geco_metrics, v3_geco_metrics, "GECO test")

    # ---- Corrected v2 metrics (undo skip multiplication for fair comparison) ----
    v2_geco_corrected = correct_for_skip(v2_geco)
    v2_geco_corrected_metrics = {
        'trt': compute_metrics(v2_geco_corrected['pred_trt'], v2_geco_corrected['human_trt']),
        'ffd': compute_metrics(v2_geco_corrected['pred_ffd'], v2_geco_corrected['human_ffd']),
        'gaze': compute_metrics(v2_geco_corrected['pred_gaze'], v2_geco_corrected['human_gaze']),
        'skip': compute_metrics(v2_geco_corrected['pred_skip'], v2_geco_corrected['human_skip']),
    }
    print_metrics_table(v2_geco_corrected_metrics,
                        "LLaMA + DiffEZR v2 CORRECTED (TRT / (1-skip)) — GECO test")

    # ---- Evaluate on Provo (cross-corpus) ----
    print("\n" + "=" * 95)
    print("  PROVO CORPUS (cross-corpus generalization)")
    print("=" * 95)

    v2_provo = collect_predictions(model_v2, provo_agg, device)
    v3_provo = collect_predictions(model_v3, provo_agg, device)
    n_words_provo = len(v2_provo['pred_trt'])
    print(f"  Total words evaluated: {n_words_provo:,}")

    v2_provo_metrics = {
        'trt': compute_metrics(v2_provo['pred_trt'], v2_provo['human_trt']),
        'ffd': compute_metrics(v2_provo['pred_ffd'], v2_provo['human_ffd']),
        'gaze': compute_metrics(v2_provo['pred_gaze'], v2_provo['human_gaze']),
        'skip': compute_metrics(v2_provo['pred_skip'], v2_provo['human_skip']),
    }
    v3_provo_metrics = {
        'trt': compute_metrics(v3_provo['pred_trt'], v3_provo['human_trt']),
        'ffd': compute_metrics(v3_provo['pred_ffd'], v3_provo['human_ffd']),
        'gaze': compute_metrics(v3_provo['pred_gaze'], v3_provo['human_gaze']),
        'skip': compute_metrics(v3_provo['pred_skip'], v3_provo['human_skip']),
    }

    print_metrics_table(v2_provo_metrics, "LLaMA + DiffEZR v2 — Provo (cross-corpus)")
    print_metrics_table(v3_provo_metrics, "LLaMA + DiffEZR v3 — Provo (cross-corpus)")
    print_head_to_head(v2_provo_metrics, v3_provo_metrics, "Provo (cross-corpus)")

    # ---- Corrected v2 metrics on Provo ----
    v2_provo_corrected = correct_for_skip(v2_provo)
    v2_provo_corrected_metrics = {
        'trt': compute_metrics(v2_provo_corrected['pred_trt'], v2_provo_corrected['human_trt']),
        'ffd': compute_metrics(v2_provo_corrected['pred_ffd'], v2_provo_corrected['human_ffd']),
        'gaze': compute_metrics(v2_provo_corrected['pred_gaze'], v2_provo_corrected['human_gaze']),
        'skip': compute_metrics(v2_provo_corrected['pred_skip'], v2_provo_corrected['human_skip']),
    }
    print_metrics_table(v2_provo_corrected_metrics,
                        "LLaMA + DiffEZR v2 CORRECTED (TRT / (1-skip)) — Provo (cross-corpus)")

    # ---- Psycholinguistic effects ----
    check_effects(v2_provo, "v2 — Provo")
    check_effects(v3_provo, "v3 — Provo")

    # ---- L1/L2 distributions ----
    print(f"\n{'=' * 95}")
    print("  INTERNAL REPRESENTATIONS")
    print(f"{'=' * 95}")
    print_l1_l2_stats(v2_geco, "v2 — GECO test")
    print_l1_l2_stats(v3_geco, "v3 — GECO test")
    print_l1_l2_stats(v2_provo, "v2 — Provo")
    print_l1_l2_stats(v3_provo, "v3 — Provo")

    # ---- EZR parameters ----
    print(f"\n{'=' * 95}")
    print("  LEARNED EZ READER PARAMETERS")
    print(f"{'=' * 95}")
    print_ezr_params(model_v2, "v2 (DiffEZR v2, independent L1/L2)")
    print_ezr_params(model_v3, "v3 (DiffEZR v3, L2=delta*L1)")

    # ---- Sample predictions ----
    print(f"\n{'=' * 95}")
    print("  SAMPLE PREDICTIONS (Provo)")
    print(f"{'=' * 95}")
    print_sample_predictions(model_v2, provo_agg, device, "v2", n_sentences=3, n_words=10)
    print_sample_predictions(model_v3, provo_agg, device, "v3", n_sentences=3, n_words=10)

    # ---- Final summary ----
    print(f"\n{'=' * 95}")
    print("  SUMMARY")
    print(f"{'=' * 95}")
    print(f"\n  GECO test ({n_words_geco:,} words):")
    print(f"    v2 raw:       r_TRT={v2_geco_metrics['trt']['r']:.3f}  MAE_TRT={v2_geco_metrics['trt']['mae']:.1f}ms  "
          f"Bias={v2_geco_metrics['trt']['bias']:+.1f}ms  r_Skip={v2_geco_metrics['skip']['r']:.3f}")
    print(f"    v2 corrected: r_TRT={v2_geco_corrected_metrics['trt']['r']:.3f}  MAE_TRT={v2_geco_corrected_metrics['trt']['mae']:.1f}ms  "
          f"Bias={v2_geco_corrected_metrics['trt']['bias']:+.1f}ms")
    print(f"    v3:           r_TRT={v3_geco_metrics['trt']['r']:.3f}  MAE_TRT={v3_geco_metrics['trt']['mae']:.1f}ms  "
          f"Bias={v3_geco_metrics['trt']['bias']:+.1f}ms  r_Skip={v3_geco_metrics['skip']['r']:.3f}")

    print(f"\n  Provo cross-corpus ({n_words_provo:,} words):")
    print(f"    v2 raw:       r_TRT={v2_provo_metrics['trt']['r']:.3f}  MAE_TRT={v2_provo_metrics['trt']['mae']:.1f}ms  "
          f"Bias={v2_provo_metrics['trt']['bias']:+.1f}ms  r_Skip={v2_provo_metrics['skip']['r']:.3f}")
    print(f"    v2 corrected: r_TRT={v2_provo_corrected_metrics['trt']['r']:.3f}  MAE_TRT={v2_provo_corrected_metrics['trt']['mae']:.1f}ms  "
          f"Bias={v2_provo_corrected_metrics['trt']['bias']:+.1f}ms")
    print(f"    v3:           r_TRT={v3_provo_metrics['trt']['r']:.3f}  MAE_TRT={v3_provo_metrics['trt']['mae']:.1f}ms  "
          f"Bias={v3_provo_metrics['trt']['bias']:+.1f}ms  r_Skip={v3_provo_metrics['skip']['r']:.3f}")

    print(f"\n  Key architectural differences:")
    print(f"    v2: Independent L1/L2 heads, TRT = (1-skip) * reading_time")
    print(f"    v3: L2 = delta*L1, TRT conditional on fixation (no skip multiplication)")
    if hasattr(model_v3, '_delta_raw'):
        print(f"    v3 learned delta = {model_v3.delta.item():.4f} (literature: 0.20-0.85)")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
