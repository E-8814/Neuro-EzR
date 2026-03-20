"""
Comprehensive comparison of ALL models on GECO test + Provo.
v2: Adds prediction std table to reveal model discrimination ability.

Models compared:
  1. LLaMA + EZR v2          (ours, cognitive)
  2. Direct LLaMA            (ours, black-box)
  3. Ohio State RoBERTa      (CMCL 2021, 6th place)
  4. Toronto CL RoBERTa      (CMCL 2021, 3rd place)
  5. BERT direct regression   (standard NLP baseline)
  6. GPT-2 surprisal + linear (psycholinguistic baseline)
  7. Original EZ Reader       (cognitive, Monte Carlo simulation)
  8. Differentiable EZ Reader (cognitive, formula-based, no neural net)

Metrics: Pearson r, MAE, RMSE, Bias, Prediction Std for TRT, FFD, Gaze, Skip.
Also reports corrected TRT for v2 models (TRT / (1-skip)).

Usage:
  python3 -u src_v2/eval_all_models_v2.py
  CUDA_VISIBLE_DEVICES=1 python3 -u src_v2/eval_all_models_v2.py
"""

import os
import sys
import csv
import math
import argparse
import time
import numpy as np
import torch
from scipy import stats as sp_stats
from torch.nn.utils.rnn import pad_sequence

# --- Path setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')
BASELINES_DIR = os.path.join(ROOT_DIR, 'archive', 'baselines')
EZR_DIR = os.path.join(ROOT_DIR, 'archive', 'original_ezreader')
OHIO_DIR = os.path.join(BASELINES_DIR, 'cmcl21_st')
TORONTO_DIR = os.path.join(BASELINES_DIR, 'cmcl21_torontocl')

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, EZR_DIR)
sys.path.insert(0, OHIO_DIR)

from data_loader import load_provo, aggregate_by_sentence
from geco_loader import load_geco, split_geco
from utilities import time_familiarity_check, time_lexical_access
from ez_wrapper import run_original_simulation_averaged


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
#  Shared helpers
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
    return word_lists, pred_vals, wlens


def compute_metrics(pred, human):
    pred, human = np.array(pred), np.array(human)
    if len(pred) > 2 and np.std(pred) > 0 and np.std(human) > 0:
        r, p = sp_stats.pearsonr(pred, human)
    else:
        r, p = 0.0, 1.0
    mae_val = np.mean(np.abs(pred - human))
    rmse_val = np.sqrt(np.mean((pred - human) ** 2))
    bias = np.mean(pred) - np.mean(human)
    pred_std = np.std(pred)
    pred_mean = np.mean(pred)
    return {'r': r, 'mae': mae_val, 'rmse': rmse_val, 'bias': bias,
            'pred_std': pred_std, 'pred_mean': pred_mean}


# --------------------------------------------------------------------------- #
#  Collect predictions from each model
# --------------------------------------------------------------------------- #

def collect_neural_predictions(model, agg_data, device, batch_size=8):
    """For models that take (word_lists, predictability, word_lengths) -> dict."""
    model.eval()
    pred_trt, pred_ffd, pred_gaze, pred_skip = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(agg_data), batch_size):
            batch = agg_data[i:i + batch_size]
            word_lists, pred_vals, wlens = collate_aggregated(batch, device)

            h_trt = pad_sequence(
                [torch.tensor(a.mean_trt, dtype=torch.float32) for a in batch],
                batch_first=True).to(device)
            h_ffd = pad_sequence(
                [torch.tensor(a.mean_ffd, dtype=torch.float32) for a in batch],
                batch_first=True).to(device)
            h_gaze = pad_sequence(
                [torch.tensor(a.mean_gaze, dtype=torch.float32) for a in batch],
                batch_first=True).to(device)
            h_skip = pad_sequence(
                [torch.tensor(a.skip_rate, dtype=torch.float32) for a in batch],
                batch_first=True).to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(word_lists, pred_vals, wlens)

            for b in range(len(batch)):
                sl = len(batch[b].tokens)
                pred_trt.extend(pred['total_reading_time'][b, :sl].cpu().tolist())
                pred_ffd.extend(pred['first_fixation'][b, :sl].cpu().tolist())
                if 'gaze_duration' in pred:
                    pred_gaze.extend(pred['gaze_duration'][b, :sl].cpu().tolist())
                else:
                    pred_gaze.extend(pred['gaze'][b, :sl].cpu().tolist())
                pred_skip.extend(pred['skip_prob'][b, :sl].cpu().tolist())

    return np.array(pred_trt), np.array(pred_ffd), np.array(pred_gaze), np.array(pred_skip)


def collect_direct_llama_predictions(model, agg_data, device, batch_size=8):
    """For DirectRegressionLLaMA which returns different keys."""
    model.eval()
    pred_trt, pred_ffd, pred_gaze, pred_skip = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(agg_data), batch_size):
            batch = agg_data[i:i + batch_size]
            word_lists, pred_vals, wlens = collate_aggregated(batch, device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(word_lists, pred_vals, wlens)

            for b in range(len(batch)):
                sl = len(batch[b].tokens)
                pred_trt.extend(pred['total_reading_time'][b, :sl].cpu().tolist())
                pred_ffd.extend(pred['first_fixation'][b, :sl].cpu().tolist())
                pred_gaze.extend(pred['gaze_duration'][b, :sl].cpu().tolist())
                pred_skip.extend(pred['skip_prob'][b, :sl].cpu().tolist())

    return np.array(pred_trt), np.array(pred_ffd), np.array(pred_gaze), np.array(pred_skip)


def collect_bert_regression_predictions(model, agg_data, device, batch_size=8):
    """For BertDirectRegression which returns trt/ffd/gaze/skip keys."""
    model.eval()
    pred_trt, pred_ffd, pred_gaze, pred_skip = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(agg_data), batch_size):
            batch = agg_data[i:i + batch_size]
            word_lists, pred_vals, wlens = collate_aggregated(batch, device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(word_lists, pred_vals, wlens)

            for b in range(len(batch)):
                sl = len(batch[b].tokens)
                pred_trt.extend(pred['total_reading_time'][b, :sl].cpu().tolist())
                pred_ffd.extend(pred['first_fixation'][b, :sl].cpu().tolist())
                pred_gaze.extend(pred['gaze_duration'][b, :sl].cpu().tolist())
                pred_skip.extend(pred['skip_prob'][b, :sl].cpu().tolist())

    return np.array(pred_trt), np.array(pred_ffd), np.array(pred_gaze), np.array(pred_skip)


def collect_formula_predictions(agg_data, subtlex, device):
    """Original EZ Reader (simulation) + Differentiable EZ Reader (formula)."""
    from diff_ezreader import DifferentiableEZReader

    diff_ezr = DifferentiableEZReader().to(device)

    orig_trt, orig_ffd = [], []
    diff_trt, diff_ffd, diff_gaze, diff_skip = [], [], [], []

    for agg in agg_data:
        tokens = agg.tokens
        preds = agg.predictabilities
        wlens = [len(t) for t in tokens]

        l1f, l2f = compute_real_l1_l2(tokens, preds, subtlex)

        # Original EZ Reader (Monte Carlo simulation)
        freqs = [get_real_frequency(t, subtlex) for t in tokens]
        try:
            orig_result = run_original_simulation_averaged(tokens, freqs, preds, num_runs=20)
            orig_trt.extend(orig_result['total_reading_time'])
            orig_ffd.extend(orig_result['first_fixation_duration'])
        except Exception:
            orig_trt.extend([0.0] * len(tokens))
            orig_ffd.extend([0.0] * len(tokens))

        # Differentiable EZ Reader (formula-based, no neural net)
        with torch.no_grad():
            dr = diff_ezr(
                torch.tensor([l1f], dtype=torch.float32, device=device),
                torch.tensor([l2f], dtype=torch.float32, device=device),
                torch.tensor([preds], dtype=torch.float32, device=device),
                torch.tensor([wlens], dtype=torch.float32, device=device),
            )
        diff_trt.extend(dr['total_reading_time'][0].tolist())
        diff_ffd.extend(dr['first_fixation'][0].tolist())
        diff_gaze.extend(dr['gaze_duration'][0].tolist())
        diff_skip.extend(dr['skip_prob'][0].tolist())

    return (np.array(orig_trt), np.array(orig_ffd),
            np.array(diff_trt), np.array(diff_ffd), np.array(diff_gaze), np.array(diff_skip))


def collect_ohio_predictions(agg_data, device):
    """Ohio State RoBERTa (4 separate models, one per metric)."""
    from transformers import RobertaModel, RobertaTokenizer
    sys.path.insert(0, BASELINES_DIR)
    from model import RobertaForGazePrediction
    from run_ohio_state_on_geco import convert_to_ohio_format, evaluate_model

    ckpt_dir = os.path.join(BASELINES_DIR, 'checkpoints_ohio_state_roberta_base')
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    data = convert_to_ohio_format(agg_data, tokenizer)

    results = {}
    for metric in ['ffd', 'gaze', 'trt', 'skip']:
        path = os.path.join(ckpt_dir, f'best_model_{metric}.pth')
        if not os.path.exists(path):
            print(f"    WARNING: Ohio State checkpoint not found: {path}")
            results[metric] = (np.zeros(1), np.zeros(1))
            continue

        roberta = RobertaModel.from_pretrained(model_name)
        m = RobertaForGazePrediction(
            pretrained=roberta, input_dim=768,
            dropout_1=0.1, hidden_dim=385, activation="relu", dropout_2=0.1,
        ).to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        m.eval()

        r, mae_val, rmse_val, preds, targets = evaluate_model(m, data, metric, tokenizer, device)
        results[metric] = np.array(preds)

        del m, roberta
        torch.cuda.empty_cache()

    return results


def collect_toronto_predictions(agg_data, device):
    """Toronto CL RoBERTa predictions."""
    import pandas as pd
    sys.path.insert(0, TORONTO_DIR)
    import src.model as toronto_model
    import src.dataloader as toronto_dataloader

    # Monkey-patch the dataloader (same fix as the adapter)
    def _patched_getitem(self, ix):
        input_ids = self.ids['input_ids'][ix]
        attention_mask = self.ids['attention_mask'][ix]
        input_tokens = [self.tokenizer.convert_ids_to_tokens(x) for x in input_ids]
        word_ids = self.ids.word_ids(ix)
        seen = set()
        is_first_subword = []
        for wid in word_ids:
            if wid is not None and wid not in seen:
                is_first_subword.append(True)
                seen.add(wid)
            else:
                is_first_subword.append(False)
        features = -torch.ones((len(input_ids), 5))
        features[is_first_subword] = torch.Tensor(
            self.df[self.df.sentence_id == ix][toronto_dataloader.FEATURES_NAMES].to_numpy().copy()
        )
        return (input_tokens, torch.LongTensor(input_ids),
                torch.LongTensor(attention_mask), features)
    toronto_dataloader.EyeTrackingCSV.__getitem__ = _patched_getitem

    ckpt_dir = os.path.join(BASELINES_DIR, 'checkpoints_toronto_roberta-base')
    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]) if os.path.exists(ckpt_dir) else []

    if not ckpt_files:
        print(f"    WARNING: No Toronto CL checkpoints found in {ckpt_dir}")
        return None

    # Build DataFrame
    rows = []
    sid = 0
    for agg in agg_data:
        valid_indices = [i for i, t in enumerate(agg.tokens) if t.strip() != '']
        if not valid_indices:
            continue
        for new_wid, w_idx in enumerate(valid_indices):
            ffd = agg.mean_ffd[w_idx]
            gaze = agg.mean_gaze[w_idx]
            trt = agg.mean_trt[w_idx]
            skip = agg.skip_rate[w_idx]
            rows.append({
                'sentence_id': sid, 'word_id': new_wid, 'word': agg.tokens[w_idx],
                'nFix': trt / ffd if ffd > 0 and trt > 0 else 0.0,
                'FFD': ffd, 'GPT': gaze, 'TRT': trt,
                'fixProp': (1.0 - skip) * 100.0,
            })
        sid += 1
    df = pd.DataFrame(rows)

    toronto_model.device = device
    model_trainer = toronto_model.ModelTrainer(model_name='roberta-base')
    model_trainer.model.load_state_dict(
        torch.load(os.path.join(ckpt_dir, ckpt_files[0]), map_location=device, weights_only=False))
    model_trainer.model.eval()

    predict_df = model_trainer.predict(df)

    return {
        'ffd': predict_df['FFD'].values,
        'gaze': predict_df['GPT'].values,
        'trt': predict_df['TRT'].values,
        'skip': (100.0 - predict_df['fixProp'].values) / 100.0,  # convert fixProp -> skip_rate
    }


def collect_gpt2_surprisal_predictions(agg_data, subtlex):
    """GPT-2 surprisal + linear regression baseline. Trains and predicts inline."""
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    device_gpt = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("    Computing GPT-2 surprisal for all words...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2').to(device_gpt).eval()

    def compute_surprisal(sentence_tokens):
        text = ' '.join(sentence_tokens)
        enc = tokenizer(text, return_tensors='pt').to(device_gpt)
        with torch.no_grad():
            logits = gpt2(**enc).logits
        log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
        input_ids = enc['input_ids'][0]

        # Map subword surprisals to words
        word_surprisals = []
        subword_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        word_idx = 0
        current_surp = 0.0
        n_sub = 0

        for t_idx in range(1, len(input_ids)):
            token_str = subword_tokens[t_idx]
            token_id = input_ids[t_idx].item()
            surp = -log_probs[t_idx - 1, token_id].item() / math.log(2)  # bits
            current_surp += surp
            n_sub += 1

            # Check if this starts a new word (Ġ prefix)
            next_is_new_word = (t_idx + 1 < len(input_ids) and
                                subword_tokens[t_idx + 1].startswith('Ġ'))
            is_last = (t_idx + 1 == len(input_ids))

            if next_is_new_word or is_last:
                word_surprisals.append(current_surp)
                current_surp = 0.0
                n_sub = 0

        # Pad or truncate to match word count
        n_words = len(sentence_tokens)
        while len(word_surprisals) < n_words:
            word_surprisals.append(0.0)
        word_surprisals = word_surprisals[:n_words]
        return word_surprisals

    # Build feature matrix
    all_features = []  # [log_freq, predictability, word_length, surprisal]
    all_trt, all_ffd, all_gaze, all_skip = [], [], [], []

    for agg in agg_data:
        surps = compute_surprisal(agg.tokens)
        for i, token in enumerate(agg.tokens):
            freq = get_real_frequency(token, subtlex)
            log_freq = math.log(freq + 1)
            all_features.append([log_freq, agg.predictabilities[i], len(token), surps[i]])
            all_trt.append(agg.mean_trt[i])
            all_ffd.append(agg.mean_ffd[i])
            all_gaze.append(agg.mean_gaze[i])
            all_skip.append(agg.skip_rate[i])

    del gpt2
    torch.cuda.empty_cache()

    X = np.array(all_features)
    return X, np.array(all_trt), np.array(all_ffd), np.array(all_gaze), np.array(all_skip)


def fit_and_predict_linear(X_train, y_train, X_test):
    """OLS linear regression via normal equation with ridge."""
    X = np.column_stack([np.ones(len(X_train)), X_train])
    lam = 1e-4
    w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y_train)
    X_t = np.column_stack([np.ones(len(X_test)), X_test])
    return X_t @ w


# --------------------------------------------------------------------------- #
#  Main comparison table
# --------------------------------------------------------------------------- #

def print_comparison_table(all_results, human, dataset_name, n_words):
    """Print the main comparison table with correlation, MAE, bias, and prediction std."""
    print(f"\n{'=' * 110}")
    print(f"  {dataset_name} ({n_words:,} words)")
    print(f"{'=' * 110}")

    # --- Table 1: Correlation and MAE ---
    print(f"\n  {'Model':<30s} │ {'r_TRT':>7s} {'MAE_TRT':>9s} │ {'r_FFD':>7s} {'MAE_FFD':>9s} │ "
          f"{'r_Gaze':>7s} {'MAE_Gaze':>9s} │ {'r_Skip':>7s} {'MAE_Skip':>9s} │ {'Cog?':>5s}")
    print(f"  {'─'*30} ┼ {'─'*17} ┼ {'─'*17} ┼ {'─'*17} ┼ {'─'*17} ┼ {'─'*5}")

    # Pre-compute all metrics for reuse
    all_metrics = []
    for name, res in all_results:
        trt_m = compute_metrics(res['trt'], human['trt']) if res.get('trt') is not None else None
        ffd_m = compute_metrics(res['ffd'], human['ffd']) if res.get('ffd') is not None else None
        gaze_m = compute_metrics(res['gaze'], human['gaze']) if res.get('gaze') is not None else None
        skip_m = compute_metrics(res['skip'], human['skip']) if res.get('skip') is not None else None
        cog = res.get('cognitive', '')
        all_metrics.append((name, trt_m, ffd_m, gaze_m, skip_m, cog))

    for name, trt_m, ffd_m, gaze_m, skip_m, cog in all_metrics:
        def fmt_r(m):
            return f"{m['r']:>7.3f}" if m else "    -- "
        def fmt_mae(m, is_skip=False):
            if m is None:
                return "      -- "
            if is_skip:
                return f"  {m['mae']:.4f}"
            return f" {m['mae']:>6.1f}ms"

        print(f"  {name:<30s} │ {fmt_r(trt_m)} {fmt_mae(trt_m)} │ {fmt_r(ffd_m)} {fmt_mae(ffd_m)} │ "
              f"{fmt_r(gaze_m)} {fmt_mae(gaze_m)} │ {fmt_r(skip_m)} {fmt_mae(skip_m, True)} │ {cog:>5s}")

    # --- Table 2: Bias ---
    print(f"\n  {'Model':<30s} │ {'Bias_TRT':>10s} │ {'Bias_FFD':>10s} │ {'Bias_Gaze':>10s} │ {'Bias_Skip':>10s}")
    print(f"  {'─'*30} ┼ {'─'*10} ┼ {'─'*10} ┼ {'─'*10} ┼ {'─'*10}")
    for name, trt_m, ffd_m, gaze_m, skip_m, cog in all_metrics:
        def fmt_bias(m, is_skip=False):
            if m is None:
                return "        --"
            if is_skip:
                return f" {m['bias']:>+9.4f}"
            return f"{m['bias']:>+9.1f}ms" if abs(m['bias']) < 1000 else "        --"

        print(f"  {name:<30s} │ {fmt_bias(trt_m)} │ {fmt_bias(ffd_m)} │ {fmt_bias(gaze_m)} │ {fmt_bias(skip_m, True)}")

    # --- Table 3: Prediction Std (discrimination ability) ---
    human_std_trt = np.std(human['trt'])
    human_std_ffd = np.std(human['ffd'])
    human_std_gaze = np.std(human['gaze'])
    human_std_skip = np.std(human['skip'])

    print(f"\n  Prediction Std (how much variation each model produces vs human data)")
    print(f"  {'Model':<30s} │ {'std_TRT':>9s} │ {'std_FFD':>9s} │ {'std_Gaze':>9s} │ {'std_Skip':>9s}")
    print(f"  {'─'*30} ┼ {'─'*9} ┼ {'─'*9} ┼ {'─'*9} ┼ {'─'*9}")
    print(f"  {'** Human data **':<30s} │ {human_std_trt:>7.1f}ms │ {human_std_ffd:>7.1f}ms │ {human_std_gaze:>7.1f}ms │ {human_std_skip:>9.4f}")
    print(f"  {'─'*30} ┼ {'─'*9} ┼ {'─'*9} ┼ {'─'*9} ┼ {'─'*9}")

    for name, trt_m, ffd_m, gaze_m, skip_m, cog in all_metrics:
        def fmt_std(m, is_skip=False):
            if m is None:
                return "       --"
            if is_skip:
                return f" {m['pred_std']:>8.4f}"
            return f"{m['pred_std']:>7.1f}ms"

        print(f"  {name:<30s} │ {fmt_std(trt_m)} │ {fmt_std(ffd_m)} │ {fmt_std(gaze_m)} │ {fmt_std(skip_m, True)}")

    # --- Table 4: Std ratio (pred_std / human_std) ---
    print(f"\n  Std Ratio (pred_std / human_std): 1.0 = matches human variation, <1 = under-discriminating")
    print(f"  {'Model':<30s} │ {'TRT':>9s} │ {'FFD':>9s} │ {'Gaze':>9s} │ {'Skip':>9s}")
    print(f"  {'─'*30} ┼ {'─'*9} ┼ {'─'*9} ┼ {'─'*9} ┼ {'─'*9}")

    for name, trt_m, ffd_m, gaze_m, skip_m, cog in all_metrics:
        def fmt_ratio(m, human_std):
            if m is None or human_std == 0:
                return "       --"
            ratio = m['pred_std'] / human_std
            return f"    {ratio:>5.2f}"

        print(f"  {name:<30s} │ {fmt_ratio(trt_m, human_std_trt)} │ {fmt_ratio(ffd_m, human_std_ffd)} │ "
              f"{fmt_ratio(gaze_m, human_std_gaze)} │ {fmt_ratio(skip_m, human_std_skip)}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,
                        default=os.path.join(ROOT_DIR, "results/eval_all_models_v2_results.txt"))
    parser.add_argument("--skip_toronto", action="store_true",
                        help="Skip Toronto CL if not yet trained")
    parser.add_argument("--skip_gpt2", action="store_true",
                        help="Skip GPT-2 surprisal (slow)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sys.stdout = Logger(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ---- Load data ----
    data_dir = os.path.join(ROOT_DIR, "data")
    subtlex = load_subtlexus(os.path.join(data_dir, "SUBTLEXus.txt"))

    print("\nLoading GECO corpus...")
    geco_raw = load_geco(
        os.path.join(data_dir, "Geco_MonolingualReadingData.csv"),
        os.path.join(data_dir, "Geco_EnglishMaterial.csv"),
        os.path.join(data_dir, "geco_predictability.pkl"))
    train_raw, val_raw, test_raw = split_geco(geco_raw)
    aggregated = aggregate_by_sentence(geco_raw, min_participants=5)
    train_text_ids = set(sd.text_id for sd in train_raw)
    val_text_ids = set(sd.text_id for sd in val_raw)
    geco_test = [a for a in aggregated if a.text_id not in train_text_ids and a.text_id not in val_text_ids]
    geco_train = [a for a in aggregated if a.text_id in train_text_ids]

    print(f"Loading Provo corpus...")
    provo_raw = load_provo(os.path.join(data_dir, "Provo_Corpus-Eyetracking_Data.csv"))
    provo_agg = aggregate_by_sentence(provo_raw, min_participants=5)

    # Collect human data
    def get_human(agg_data):
        trt, ffd, gaze, skip = [], [], [], []
        for a in agg_data:
            trt.extend(a.mean_trt)
            ffd.extend(a.mean_ffd)
            gaze.extend(a.mean_gaze)
            skip.extend(a.skip_rate)
        return {'trt': np.array(trt), 'ffd': np.array(ffd),
                'gaze': np.array(gaze), 'skip': np.array(skip)}

    human_geco = get_human(geco_test)
    human_provo = get_human(provo_agg)
    print(f"  GECO test: {len(human_geco['trt']):,} words | Provo: {len(human_provo['trt']):,} words")

    # ---- Evaluate each model ----
    geco_results = []
    provo_results = []

    # --- 1. LLaMA + EZR v2 ---
    print("\n[1/8] Loading LLaMA + EZR v2...")
    from model_llama import NeuralEZReaderLLaMA
    ckpt = torch.load(os.path.join(ROOT_DIR, "checkpoints_v2/geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0/best_model.pt"),
                       map_location=device, weights_only=False)
    m = NeuralEZReaderLLaMA(model_name=ckpt.get('model_name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'),
                             freeze_layers=ckpt.get('freeze_layers', 16), hidden_dim=256).to(device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()

    trt, ffd, gaze, skip = collect_neural_predictions(m, geco_test, device)
    trt_corr = trt / np.clip(1.0 - skip, 0.05, 1.0)
    geco_results.append(("LLaMA+EZR v2 (ours)", {'trt': trt_corr, 'ffd': ffd, 'gaze': gaze, 'skip': skip, 'cognitive': 'Yes'}))

    trt, ffd, gaze, skip = collect_neural_predictions(m, provo_agg, device)
    trt_corr = trt / np.clip(1.0 - skip, 0.05, 1.0)
    provo_results.append(("LLaMA+EZR v2 (ours)", {'trt': trt_corr, 'ffd': ffd, 'gaze': gaze, 'skip': skip, 'cognitive': 'Yes'}))
    del m; torch.cuda.empty_cache()

    # --- 1b. LLaMA + EZR v2-delta ---
    v2delta_ckpt_path = os.path.join(ROOT_DIR, "checkpoints_v2_delta/geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0/best_model.pt")
    if os.path.exists(v2delta_ckpt_path):
        print("[1b/9] Loading LLaMA + EZR v2-delta...")
        from model_llama_v2_delta import NeuralEZReaderLLaMA as NeuralEZReaderLLaMAv2Delta
        ckpt = torch.load(v2delta_ckpt_path, map_location=device, weights_only=False)
        m = NeuralEZReaderLLaMAv2Delta(
            model_name=ckpt.get('model_name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'),
            freeze_layers=ckpt.get('freeze_layers', 16), hidden_dim=256).to(device)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        if 'delta' in ckpt:
            print(f"    Learned delta: {ckpt['delta']:.4f}")

        trt, ffd, gaze, skip = collect_neural_predictions(m, geco_test, device)
        trt_corr = trt / np.clip(1.0 - skip, 0.05, 1.0)
        geco_results.append(("LLaMA+EZR v2-delta (ours)", {'trt': trt_corr, 'ffd': ffd, 'gaze': gaze, 'skip': skip, 'cognitive': 'Yes'}))

        trt, ffd, gaze, skip = collect_neural_predictions(m, provo_agg, device)
        trt_corr = trt / np.clip(1.0 - skip, 0.05, 1.0)
        provo_results.append(("LLaMA+EZR v2-delta (ours)", {'trt': trt_corr, 'ffd': ffd, 'gaze': gaze, 'skip': skip, 'cognitive': 'Yes'}))
        del m; torch.cuda.empty_cache()
    else:
        print("[1b/9] Skipping LLaMA+EZR v2-delta (no checkpoint yet)")

    # --- 2. Direct LLaMA ---
    print("[2/9] Loading Direct LLaMA...")
    from model_llama_direct import DirectRegressionLLaMA
    ckpt = torch.load(os.path.join(ROOT_DIR, "checkpoints_v2/geco_direct_TinyLlama_TinyLlama-1.1B-Chat-v1.0/best_model.pt"),
                       map_location=device, weights_only=False)
    m = DirectRegressionLLaMA(model_name=ckpt.get('model_name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'),
                               freeze_layers=ckpt.get('freeze_layers', 16), hidden_dim=256).to(device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()

    trt, ffd, gaze, skip = collect_direct_llama_predictions(m, geco_test, device)
    geco_results.append(("Direct LLaMA (ours)", {'trt': trt, 'ffd': ffd, 'gaze': gaze, 'skip': skip, 'cognitive': 'No'}))
    trt, ffd, gaze, skip = collect_direct_llama_predictions(m, provo_agg, device)
    provo_results.append(("Direct LLaMA (ours)", {'trt': trt, 'ffd': ffd, 'gaze': gaze, 'skip': skip, 'cognitive': 'No'}))
    del m; torch.cuda.empty_cache()

    # --- 3. Ohio State RoBERTa (CMCL 2021, 6th) ---
    print("[3/8] Loading Ohio State RoBERTa...")
    ohio_geco = collect_ohio_predictions(geco_test, device)
    ohio_provo = collect_ohio_predictions(provo_agg, device)
    geco_results.append(("Ohio State RoBERTa (6th)", {'trt': ohio_geco.get('trt'), 'ffd': ohio_geco.get('ffd'),
                          'gaze': ohio_geco.get('gaze'), 'skip': ohio_geco.get('skip'), 'cognitive': 'No'}))
    provo_results.append(("Ohio State RoBERTa (6th)", {'trt': ohio_provo.get('trt'), 'ffd': ohio_provo.get('ffd'),
                           'gaze': ohio_provo.get('gaze'), 'skip': ohio_provo.get('skip'), 'cognitive': 'No'}))

    # --- 4. Toronto CL RoBERTa (CMCL 2021, 3rd) ---
    if not args.skip_toronto:
        print("[4/8] Loading Toronto CL RoBERTa...")
        toronto_geco = collect_toronto_predictions(geco_test, device)
        toronto_provo = collect_toronto_predictions(provo_agg, device)
        if toronto_geco is not None:
            geco_results.append(("Toronto CL RoBERTa (3rd)", {**toronto_geco, 'cognitive': 'No'}))
        if toronto_provo is not None:
            provo_results.append(("Toronto CL RoBERTa (3rd)", {**toronto_provo, 'cognitive': 'No'}))
    else:
        print("[4/8] Skipping Toronto CL (--skip_toronto)")

    # --- 5. BERT + EZR (cognitive model, not direct regression) ---
    print("[5/8] Loading BERT + EZR...")
    from model_bert import NeuralEZReaderBERT
    ckpt = torch.load(os.path.join(ROOT_DIR, "checkpoints_v2/geco_bert/best_model_bert.pt"),
                       map_location=device, weights_only=False)
    m = NeuralEZReaderBERT(bert_model_name=ckpt.get('bert_model_name', 'bert-base-uncased'),
                            freeze_bert_layers=ckpt.get('freeze_bert_layers', 8), hidden_dim=256).to(device)
    m.load_state_dict(ckpt['model_state_dict'], strict=False)
    m.eval()
    print("    Note: BERT+EZR loaded with strict=False (v1 checkpoint, skip_head randomly initialized)")

    trt, ffd, gaze, skip = collect_neural_predictions(m, geco_test, device)
    trt_corr = trt / np.clip(1.0 - skip, 0.05, 1.0)
    geco_results.append(("BERT+EZR (ours)", {'trt': trt_corr, 'ffd': ffd, 'gaze': gaze, 'skip': skip, 'cognitive': 'Yes'}))
    trt, ffd, gaze, skip = collect_neural_predictions(m, provo_agg, device)
    trt_corr = trt / np.clip(1.0 - skip, 0.05, 1.0)
    provo_results.append(("BERT+EZR (ours)", {'trt': trt_corr, 'ffd': ffd, 'gaze': gaze, 'skip': skip, 'cognitive': 'Yes'}))
    del m; torch.cuda.empty_cache()

    # --- 5b. BERT direct regression (no cognitive architecture) ---
    bert_direct_ckpt = None
    for candidate in [
        os.path.join(BASELINES_DIR, "checkpoints_bert_direct_bert-base-uncased/best_model.pt"),
        os.path.join(BASELINES_DIR, "checkpoints_bert_direct_bert-base-uncased/best_model_bert.pt"),
    ]:
        if os.path.exists(candidate):
            bert_direct_ckpt = candidate
            break

    if bert_direct_ckpt:
        print("[5b/9] Loading BERT direct regression...")
        from bert_regression import BertDirectRegression
        ckpt = torch.load(bert_direct_ckpt, map_location=device, weights_only=False)
        m = BertDirectRegression(
            bert_model_name=ckpt.get('bert_model_name', 'bert-base-uncased'),
            freeze_bert_layers=ckpt.get('freeze_bert_layers', 8), hidden_dim=256).to(device)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()

        trt, ffd, gaze, skip = collect_bert_regression_predictions(m, geco_test, device)
        geco_results.append(("BERT regression", {'trt': trt, 'ffd': ffd, 'gaze': gaze, 'skip': skip, 'cognitive': 'No'}))
        trt, ffd, gaze, skip = collect_bert_regression_predictions(m, provo_agg, device)
        provo_results.append(("BERT regression", {'trt': trt, 'ffd': ffd, 'gaze': gaze, 'skip': skip, 'cognitive': 'No'}))
        del m; torch.cuda.empty_cache()
    else:
        print("[5b/9] Skipping BERT direct regression (no checkpoint yet)")

    # --- 6. GPT-2 surprisal + linear regression ---
    if not args.skip_gpt2:
        print("[6/8] Computing GPT-2 surprisal baseline...")
        # Train on GECO train, evaluate on GECO test and Provo
        X_train, y_trt_train, y_ffd_train, y_gaze_train, y_skip_train = \
            collect_gpt2_surprisal_predictions(geco_train, subtlex)
        X_test_geco, _, _, _, _ = collect_gpt2_surprisal_predictions(geco_test, subtlex)
        X_test_provo, _, _, _, _ = collect_gpt2_surprisal_predictions(provo_agg, subtlex)

        gpt2_geco = {
            'trt': fit_and_predict_linear(X_train, y_trt_train, X_test_geco),
            'ffd': fit_and_predict_linear(X_train, y_ffd_train, X_test_geco),
            'gaze': fit_and_predict_linear(X_train, y_gaze_train, X_test_geco),
            'skip': np.clip(fit_and_predict_linear(X_train, y_skip_train, X_test_geco), 0, 1),
            'cognitive': 'No',
        }
        gpt2_provo = {
            'trt': fit_and_predict_linear(X_train, y_trt_train, X_test_provo),
            'ffd': fit_and_predict_linear(X_train, y_ffd_train, X_test_provo),
            'gaze': fit_and_predict_linear(X_train, y_gaze_train, X_test_provo),
            'skip': np.clip(fit_and_predict_linear(X_train, y_skip_train, X_test_provo), 0, 1),
            'cognitive': 'No',
        }
        geco_results.append(("GPT-2 surprisal + linear", gpt2_geco))
        provo_results.append(("GPT-2 surprisal + linear", gpt2_provo))
    else:
        print("[6/8] Skipping GPT-2 surprisal (--skip_gpt2)")

    # --- 7. Original EZ Reader + 8. Differentiable EZ Reader ---
    print("[7-8/8] Computing formula-based models...")
    (orig_trt_g, orig_ffd_g, diff_trt_g, diff_ffd_g, diff_gaze_g, diff_skip_g) = \
        collect_formula_predictions(geco_test, subtlex, device)
    (orig_trt_p, orig_ffd_p, diff_trt_p, diff_ffd_p, diff_gaze_p, diff_skip_p) = \
        collect_formula_predictions(provo_agg, subtlex, device)

    geco_results.append(("Original EZ Reader", {'trt': orig_trt_g, 'ffd': orig_ffd_g,
                          'gaze': None, 'skip': None, 'cognitive': 'Yes'}))
    geco_results.append(("Diff EZ Reader (formula)", {'trt': diff_trt_g, 'ffd': diff_ffd_g,
                          'gaze': diff_gaze_g, 'skip': diff_skip_g, 'cognitive': 'Yes'}))

    provo_results.append(("Original EZ Reader", {'trt': orig_trt_p, 'ffd': orig_ffd_p,
                           'gaze': None, 'skip': None, 'cognitive': 'Yes'}))
    provo_results.append(("Diff EZ Reader (formula)", {'trt': diff_trt_p, 'ffd': diff_ffd_p,
                           'gaze': diff_gaze_p, 'skip': diff_skip_p, 'cognitive': 'Yes'}))

    # ---- Print results ----
    print_comparison_table(geco_results, human_geco, "GECO TEST SET (in-distribution)",
                           len(human_geco['trt']))
    print_comparison_table(provo_results, human_provo, "PROVO CORPUS (cross-corpus generalization)",
                           len(human_provo['trt']))

    # ---- Summary ----
    print(f"\n{'=' * 110}")
    print("  NOTES")
    print(f"{'=' * 110}")
    print("  - LLaMA+EZR v2 TRT is corrected: TRT/(1-skip), matching human conditional-on-fixation data")
    print("  - Ohio State trains 4 separate models (one per metric)")
    print("  - Toronto CL trains one model predicting all metrics jointly")
    print("  - GPT-2 surprisal uses 4 features (log_freq, predictability, word_length, surprisal)")
    print("  - Original EZ Reader uses Monte Carlo simulation (20 runs averaged)")
    print("  - Diff EZ Reader uses formula-based L1/L2 from published EZR parameters")
    print("  - 'Cog?' = Yes if model has explicit cognitive architecture (E-Z Reader)")
    print("  - Prediction Std shows how much variation a model produces; near-zero = predicting the mean")
    print("  - Std Ratio = pred_std / human_std; values << 1.0 indicate the model under-discriminates")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
