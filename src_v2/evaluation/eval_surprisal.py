"""
Surprisal Analysis: Correlate LLaMA surprisal with learned L1/L2 values.

If L1 correlates with surprisal (information-theoretic word difficulty), then
the model learned cognitively meaningful representations: harder-to-predict
words take longer to process, matching E-Z Reader theory (Reichle et al., 2003).

Analyses:
  1. Correlation matrix: L1, L2, surprisal, freq, length, predictability
  2. Correlations with human reading times
  3. Partial correlations (controlling for freq/length)
  4. Regression: what linguistic features predict L1?
  5. Unique variance: does L1 predict human RT beyond surprisal?
  6. Quintile analysis: mean L1/L2 by surprisal bin

Usage:
    python3 -u src_v2/eval_surprisal.py
"""

import os
import sys
import csv
import math
import time

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lm_model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'archive', 'original_ezreader'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from model_llama import NeuralEZReaderLLaMA
from data_loader import load_provo, aggregate_by_sentence
from geco_loader import load_geco, split_geco


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


def pearson_r(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
#  Phase 1: Collect L1/L2/skip from trained model
# --------------------------------------------------------------------------- #

def load_trained_model(ckpt_dir, device):
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
    print(f"  Trained model loaded (epoch {epoch}), base: {model_name}")
    return model, model_name


def collect_ezr_predictions(sentences, subtlex, model, device):
    """Get L1/L2/skip/reading time predictions from the trained model."""
    words = []
    total = len(sentences)

    for s_idx, agg in enumerate(sentences):
        if (s_idx + 1) % 50 == 0:
            print(f"  Collecting predictions {s_idx+1}/{total}...", flush=True)

        tokens = agg.tokens
        preds = agg.predictabilities
        wlens = [len(t) for t in tokens]

        pred_t = torch.tensor([preds], dtype=torch.float32).to(device)
        wlen_t = torch.tensor([wlens], dtype=torch.float32).to(device)

        with torch.no_grad():
            result = model([tokens], pred_t, wlen_t)

        for i in range(len(tokens)):
            freq = get_real_frequency(tokens[i], subtlex)
            words.append({
                'token': tokens[i],
                'freq': freq,
                'log_freq': math.log10(max(1, freq)),
                'pred': preds[i],
                'wlen': wlens[i],
                'l1': result['L1'][0, i].cpu().item(),
                'l2': result['L2'][0, i].cpu().item(),
                'skip': result['skip_prob'][0, i].cpu().item(),
                'ffd': result['first_fixation'][0, i].cpu().item(),
                'gaze': result['gaze_duration'][0, i].cpu().item(),
                'trt': result['total_reading_time'][0, i].cpu().item(),
                'h_ffd': agg.mean_ffd[i],
                'h_gaze': agg.mean_gaze[i],
                'h_trt': agg.mean_trt[i],
                'h_skip': agg.skip_rate[i],
            })

    return words


# --------------------------------------------------------------------------- #
#  Phase 2: Surprisal from base (pre-trained) CausalLM
# --------------------------------------------------------------------------- #

def load_causal_lm(model_name, device):
    """Load the base CausalLM (with LM head) for surprisal computation."""
    print(f"  Loading base CausalLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    lm.eval()
    n_params = sum(p.numel() for p in lm.parameters()) / 1e6
    print(f"  CausalLM loaded ({n_params:.0f}M params)")
    return lm, tokenizer


def compute_surprisal(sentences, lm, tokenizer, device):
    """Compute per-word surprisal (in bits) for each sentence.

    Surprisal = -log2 P(word | left context).
    For multi-subword words, surprisal = sum of subword surprisals
    (equivalent to -log2 of the joint probability via chain rule).
    """
    all_surprisals = []
    total = len(sentences)

    for s_idx, agg in enumerate(sentences):
        if (s_idx + 1) % 50 == 0:
            print(f"  Computing surprisal {s_idx+1}/{total}...", flush=True)

        tokens = agg.tokens
        n_words = len(tokens)

        # Tokenize with word alignment tracking
        enc = tokenizer(
            [tokens],
            is_split_into_words=True,
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            logits = lm(input_ids=input_ids, attention_mask=attention_mask).logits

        # Per-token surprisal: logits[:,t,:] predicts token at position t+1
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_surprisal = (-token_log_probs[0] / math.log(2)).cpu()  # bits

        # Aggregate subword surprisals to word level
        word_ids = enc.word_ids(batch_index=0)
        word_surprisals = [0.0] * n_words

        for tok_pos in range(1, len(word_ids)):  # skip position 0 (BOS)
            w_id = word_ids[tok_pos]
            if w_id is not None and w_id < n_words:
                # token at position tok_pos has surprisal at index tok_pos-1
                word_surprisals[w_id] += token_surprisal[tok_pos - 1].item()

        all_surprisals.append(word_surprisals)

    return all_surprisals


# --------------------------------------------------------------------------- #
#  Statistical helpers
# --------------------------------------------------------------------------- #

def partial_corr(x, y, covariates):
    """Partial Pearson r between x and y, controlling for covariates."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    covs = np.column_stack(
        [np.ones(len(x))] + [np.array(c, dtype=float) for c in covariates]
    )
    beta_x = np.linalg.lstsq(covs, x, rcond=None)[0]
    beta_y = np.linalg.lstsq(covs, y, rcond=None)[0]
    return pearson_r(x - covs @ beta_x, y - covs @ beta_y)


def ols_regression(y, predictors, names):
    """OLS regression. Returns (R², adj_R², betas, SEs, t_stats, p_values)."""
    y = np.array(y, dtype=float)
    X = np.column_stack(
        [np.ones(len(y))] + [np.array(p, dtype=float) for p in predictors]
    )
    n, k = X.shape

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else r2

    mse = ss_res / (n - k) if n > k else 1e-10
    try:
        var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
        se = np.sqrt(np.abs(var_beta))
    except np.linalg.LinAlgError:
        se = np.ones(k) * float('inf')

    t_stats = beta / np.where(se > 0, se, 1e-10)
    p_vals = 2 * (1 - sp_stats.t.cdf(np.abs(t_stats), df=max(1, n - k)))

    all_names = ['intercept'] + list(names)
    return (
        r2, adj_r2,
        dict(zip(all_names, beta)),
        dict(zip(all_names, se)),
        dict(zip(all_names, t_stats)),
        dict(zip(all_names, p_vals)),
    )


# --------------------------------------------------------------------------- #
#  Printing
# --------------------------------------------------------------------------- #

def print_section(title):
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")


def print_correlation_matrix(words):
    print_section("CORRELATION MATRIX")

    var_defs = [
        ('L1',         [w['l1'] for w in words]),
        ('L2',         [w['l2'] for w in words]),
        ('L1+L2',      [w['l1'] + w['l2'] for w in words]),
        ('Skip',       [w['skip'] for w in words]),
        ('Surprisal',  [w['surprisal'] for w in words]),
        ('Log Freq',   [w['log_freq'] for w in words]),
        ('Word Len',   [w['wlen'] for w in words]),
        ('Cloze Pred', [w['pred'] for w in words]),
    ]

    names = [v[0] for v in var_defs]
    header = f"  {'':>12}" + "".join(f" {n:>10}" for n in names)
    print(header)
    print(f"  {'-' * (12 + 11 * len(names))}")

    for i, (name_i, vals_i) in enumerate(var_defs):
        row = f"  {name_i:>12}"
        for j, (name_j, vals_j) in enumerate(var_defs):
            if j < i:
                row += f" {'':>10}"
            elif j == i:
                row += f" {'1.000':>10}"
            else:
                row += f" {pearson_r(vals_i, vals_j):>10.3f}"
        print(row)


def print_human_correlations(words):
    print_section("CORRELATIONS WITH HUMAN READING TIMES")

    predictors = [
        ('L1',         [w['l1'] for w in words]),
        ('L2',         [w['l2'] for w in words]),
        ('L1+L2',      [w['l1'] + w['l2'] for w in words]),
        ('Surprisal',  [w['surprisal'] for w in words]),
        ('Log Freq',   [w['log_freq'] for w in words]),
        ('Word Len',   [w['wlen'] for w in words]),
        ('Cloze Pred', [w['pred'] for w in words]),
        ('Pred FFD',   [w['ffd'] for w in words]),
        ('Pred Gaze',  [w['gaze'] for w in words]),
        ('Pred TRT',   [w['trt'] for w in words]),
        ('Pred Skip',  [w['skip'] for w in words]),
    ]

    human_vars = [
        ('h_FFD',  [w['h_ffd'] for w in words]),
        ('h_Gaze', [w['h_gaze'] for w in words]),
        ('h_TRT',  [w['h_trt'] for w in words]),
        ('h_Skip', [w['h_skip'] for w in words]),
    ]

    header = f"  {'':>12}" + "".join(f" {n:>10}" for n, _ in human_vars)
    print(header)
    print(f"  {'-' * (12 + 11 * len(human_vars))}")

    for pred_name, pred_vals in predictors:
        row = f"  {pred_name:>12}"
        for _, h_vals in human_vars:
            row += f" {pearson_r(pred_vals, h_vals):>10.3f}"
        print(row)


def print_partial_correlations(words):
    print_section("PARTIAL CORRELATIONS (controlling for word length & log frequency)")

    covs = [
        [w['wlen'] for w in words],
        [w['log_freq'] for w in words],
    ]
    surp = [w['surprisal'] for w in words]

    pairs = [
        ('L1',        [w['l1'] for w in words]),
        ('L2',        [w['l2'] for w in words]),
        ('L1+L2',     [w['l1'] + w['l2'] for w in words]),
        ('Skip',      [w['skip'] for w in words]),
        ('Pred FFD',  [w['ffd'] for w in words]),
        ('Pred Gaze', [w['gaze'] for w in words]),
        ('Pred TRT',  [w['trt'] for w in words]),
    ]

    print(f"\n  {'Variable':>12} {'r(., Surp)':>12} {'partial r':>12}  "
          f"(controlling for word_len + log_freq)")
    print(f"  {'-' * 60}")

    for name, vals in pairs:
        r_raw = pearson_r(vals, surp)
        r_part = partial_corr(vals, surp, covs)
        print(f"  {name:>12} {r_raw:>12.3f} {r_part:>12.3f}")

    # L1 vs human data, controlling for surprisal + length + freq
    print(f"\n  {'Variable':>12} {'r(., h_TRT)':>12} {'partial r':>12}  "
          f"(controlling for surprisal + word_len + log_freq)")
    print(f"  {'-' * 70}")

    covs2 = covs + [surp]
    h_trt = [w['h_trt'] for w in words]

    for name, vals in [
        ('L1',       [w['l1'] for w in words]),
        ('L2',       [w['l2'] for w in words]),
        ('Pred TRT', [w['trt'] for w in words]),
    ]:
        r_raw = pearson_r(vals, h_trt)
        r_part = partial_corr(vals, h_trt, covs2)
        print(f"  {name:>12} {r_raw:>12.3f} {r_part:>12.3f}")


def print_regression_what_predicts_l1(words):
    print_section("REGRESSION: What predicts L1?")
    print("  Model: L1 ~ surprisal + log_freq + word_length + predictability\n")

    y = [w['l1'] for w in words]
    predictors = [
        [w['surprisal'] for w in words],
        [w['log_freq'] for w in words],
        [w['wlen'] for w in words],
        [w['pred'] for w in words],
    ]
    names = ['surprisal', 'log_freq', 'word_length', 'predictability']

    r2, adj_r2, betas, ses, t_stats, p_vals = ols_regression(y, predictors, names)

    print(f"  R² = {r2:.4f},  Adjusted R² = {adj_r2:.4f}")
    print(f"\n  {'Predictor':>16} {'Beta':>10} {'SE':>10} {'t':>10} {'p':>12}")
    print(f"  {'-' * 60}")
    for name in ['intercept'] + names:
        sig = ('***' if p_vals[name] < 0.001 else
               '**'  if p_vals[name] < 0.01 else
               '*'   if p_vals[name] < 0.05 else '')
        print(f"  {name:>16} {betas[name]:>10.4f} {ses[name]:>10.4f} "
              f"{t_stats[name]:>10.2f} {p_vals[name]:>11.2e} {sig}")


def print_regression_unique_variance(words):
    print_section("UNIQUE VARIANCE: Does L1 predict human RT beyond surprisal?")

    for h_name, h_key in [('FFD', 'h_ffd'), ('Gaze', 'h_gaze'), ('TRT', 'h_trt')]:
        print(f"\n  --- Human {h_name} ---")
        y = [w[h_key] for w in words]

        surp = [w['surprisal'] for w in words]
        l1 = [w['l1'] for w in words]
        log_freq = [w['log_freq'] for w in words]
        wlen = [w['wlen'] for w in words]

        # Model 1: surprisal + controls
        r2_surp, _, _, _, _, _ = ols_regression(
            y, [surp, log_freq, wlen], ['surprisal', 'log_freq', 'wlen']
        )
        # Model 2: L1 + controls
        r2_l1, _, _, _, _, _ = ols_regression(
            y, [l1, log_freq, wlen], ['L1', 'log_freq', 'wlen']
        )
        # Model 3: both + controls
        r2_both, _, betas, _, t_stats, p_vals = ols_regression(
            y, [surp, l1, log_freq, wlen], ['surprisal', 'L1', 'log_freq', 'wlen']
        )

        print(f"  R²(surprisal + controls):    {r2_surp:.4f}")
        print(f"  R²(L1 + controls):           {r2_l1:.4f}")
        print(f"  R²(both + controls):         {r2_both:.4f}")
        print(f"  deltaR² L1 beyond surprisal: {r2_both - r2_surp:+.4f}")
        print(f"  deltaR² surp beyond L1:      {r2_both - r2_l1:+.4f}")

        for name in ['surprisal', 'L1']:
            sig = ('***' if p_vals[name] < 0.001 else
                   '**'  if p_vals[name] < 0.01 else
                   '*'   if p_vals[name] < 0.05 else '')
            print(f"    {name:>12}: beta={betas[name]:>8.4f}, "
                  f"t={t_stats[name]:>7.2f}, p={p_vals[name]:.2e} {sig}")


def print_quintile_analysis(words):
    print_section("QUINTILE ANALYSIS: Mean values by surprisal bin")

    surps = np.array([w['surprisal'] for w in words])
    quintile_edges = np.percentile(surps, [20, 40, 60, 80])

    bins = []
    for w in words:
        s = w['surprisal']
        if s <= quintile_edges[0]:   bins.append(0)
        elif s <= quintile_edges[1]: bins.append(1)
        elif s <= quintile_edges[2]: bins.append(2)
        elif s <= quintile_edges[3]: bins.append(3)
        else:                        bins.append(4)

    header = (f"  {'':>16}"
              + "".join(f" {'Q' + str(q+1):>10}" for q in range(5))
              + f" {'r(Q,val)':>10}")
    print(header)
    print(f"  {'-' * (16 + 11 * 6)}")

    # Surprisal range per quintile
    row = f"  {'Mean Surp':>16}"
    for q in range(5):
        vals = [w['surprisal'] for w, b in zip(words, bins) if b == q]
        row += f" {np.mean(vals):>10.2f}"
    row += f" {'':>10}"
    print(row)

    row = f"  {'N words':>16}"
    for q in range(5):
        row += f" {sum(1 for b in bins if b == q):>10d}"
    row += f" {'':>10}"
    print(row)

    var_defs = [
        ('L1 (ms)',   'l1'),
        ('L2 (ms)',   'l2'),
        ('L1+L2',     None),
        ('Pred FFD',  'ffd'),
        ('Pred Gaze', 'gaze'),
        ('Pred TRT',  'trt'),
        ('Skip prob',  'skip'),
        ('h_FFD',     'h_ffd'),
        ('h_Gaze',    'h_gaze'),
        ('h_TRT',     'h_trt'),
        ('h_Skip',    'h_skip'),
    ]

    for var_name, key in var_defs:
        row = f"  {var_name:>16}"
        q_means = []
        for q in range(5):
            if key is None:
                vals = [w['l1'] + w['l2'] for w, b in zip(words, bins) if b == q]
            else:
                vals = [w[key] for w, b in zip(words, bins) if b == q]
            m = np.mean(vals)
            q_means.append(m)
            row += f" {m:>10.2f}"

        r = pearson_r(list(range(5)), q_means)
        row += f" {r:>10.3f}"
        print(row)


def print_key_findings(words, corpus_name):
    print_section(f"KEY FINDINGS — {corpus_name}")

    surp = np.array([w['surprisal'] for w in words])
    l1 = np.array([w['l1'] for w in words])
    l2 = np.array([w['l2'] for w in words])

    r_l1_surp = pearson_r(l1, surp)
    r_l2_surp = pearson_r(l2, surp)
    r_l1l2_surp = pearson_r(l1 + l2, surp)

    covs = [
        [w['wlen'] for w in words],
        [w['log_freq'] for w in words],
    ]
    r_l1_surp_partial = partial_corr(l1.tolist(), surp.tolist(), covs)

    print(f"\n  1. L1-Surprisal correlation:")
    print(f"     r(L1, surprisal)          = {r_l1_surp:.3f}")
    print(f"     r(L2, surprisal)          = {r_l2_surp:.3f}")
    print(f"     r(L1+L2, surprisal)       = {r_l1l2_surp:.3f}")
    print(f"     partial r(L1, surp | controls) = {r_l1_surp_partial:.3f}")

    if r_l1_surp > 0.3:
        print(f"     -> STRONG: L1 captures surprisal signal")
    elif r_l1_surp > 0.15:
        print(f"     -> MODERATE: L1 partially captures surprisal")
    else:
        print(f"     -> WEAK: L1 does not strongly track surprisal")

    # Does L1 predict RT beyond surprisal?
    h_trt = [w['h_trt'] for w in words]
    r2_surp, _, _, _, _, _ = ols_regression(
        h_trt,
        [surp.tolist(), covs[0], covs[1]],
        ['surprisal', 'wlen', 'log_freq'],
    )
    r2_both, _, _, _, _, _ = ols_regression(
        h_trt,
        [surp.tolist(), l1.tolist(), covs[0], covs[1]],
        ['surprisal', 'L1', 'wlen', 'log_freq'],
    )
    delta_r2 = r2_both - r2_surp

    print(f"\n  2. Unique variance in human TRT:")
    print(f"     R²(surprisal + controls)  = {r2_surp:.4f}")
    print(f"     R²(+ L1)                  = {r2_both:.4f}")
    print(f"     deltaR² from adding L1    = {delta_r2:+.4f}")

    if delta_r2 > 0.01:
        print(f"     -> L1 captures meaningful variance beyond surprisal")
    elif delta_r2 > 0.001:
        print(f"     -> L1 adds modest unique variance beyond surprisal")
    else:
        print(f"     -> L1 adds little beyond surprisal for TRT prediction")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    results_path = os.path.join(
        os.path.dirname(__file__), '..', 'results', 'eval_surprisal_results.txt'
    )
    sys.stdout = Logger(results_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'v2')
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    subtlex = load_subtlexus(os.path.join(data_dir, 'SUBTLEXus.txt'))
    print(f"SUBTLEXus: {len(subtlex)} entries")

    # ---- Load datasets ----
    geco_mat = os.path.join(data_dir, 'Geco_EnglishMaterial.csv')
    geco_rd = os.path.join(data_dir, 'Geco_MonolingualReadingData.csv')
    geco_pred = os.path.join(data_dir, 'geco_predictability.pkl')
    print(f"\nLoading GECO...")
    geco_raw = load_geco(geco_rd, geco_mat, geco_pred)
    train_raw, val_raw, _ = split_geco(geco_raw)
    geco_agg = aggregate_by_sentence(geco_raw, min_participants=5)
    train_ids = set(sd.text_id for sd in train_raw)
    val_ids = set(sd.text_id for sd in val_raw)
    geco_test = [a for a in geco_agg
                 if a.text_id not in train_ids and a.text_id not in val_ids]
    n_geco_words = sum(len(a.tokens) for a in geco_test)
    print(f"  GECO test: {len(geco_test)} sentences, {n_geco_words} words")

    provo_et = os.path.join(data_dir, 'Provo_Corpus-Eyetracking_Data.csv')
    print(f"Loading Provo...")
    provo_raw = load_provo(provo_et)
    provo_sents = aggregate_by_sentence(provo_raw, min_participants=10)
    n_provo_words = sum(len(a.tokens) for a in provo_sents)
    print(f"  Provo: {len(provo_sents)} sentences, {n_provo_words} words")

    # ---- Phase 1: Trained model predictions ----
    print(f"\n{'#' * 100}")
    print(f"#  PHASE 1: Collecting L1/L2/skip from trained model")
    print(f"{'#' * 100}")

    model, model_name = load_trained_model(ckpt_dir, device)

    print(f"\n  GECO test set:")
    t0 = time.time()
    geco_words = collect_ezr_predictions(geco_test, subtlex, model, device)
    print(f"  {len(geco_words)} words in {time.time()-t0:.1f}s")

    print(f"\n  Provo:")
    t0 = time.time()
    provo_words = collect_ezr_predictions(provo_sents, subtlex, model, device)
    print(f"  {len(provo_words)} words in {time.time()-t0:.1f}s")

    del model
    torch.cuda.empty_cache()
    print(f"  Trained model freed from GPU")

    # ---- Phase 2: Surprisal from base CausalLM ----
    print(f"\n{'#' * 100}")
    print(f"#  PHASE 2: Computing surprisal from base CausalLM")
    print(f"{'#' * 100}")

    lm, tokenizer = load_causal_lm(model_name, device)

    print(f"\n  GECO test set:")
    t0 = time.time()
    geco_surprisals = compute_surprisal(geco_test, lm, tokenizer, device)
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\n  Provo:")
    t0 = time.time()
    provo_surprisals = compute_surprisal(provo_sents, lm, tokenizer, device)
    print(f"  Done in {time.time()-t0:.1f}s")

    del lm
    torch.cuda.empty_cache()
    print(f"  CausalLM freed from GPU")

    # ---- Merge surprisal into word data ----
    idx = 0
    for s_idx, agg in enumerate(geco_test):
        for w_idx in range(len(agg.tokens)):
            geco_words[idx]['surprisal'] = geco_surprisals[s_idx][w_idx]
            idx += 1
    assert idx == len(geco_words)

    idx = 0
    for s_idx, agg in enumerate(provo_sents):
        for w_idx in range(len(agg.tokens)):
            provo_words[idx]['surprisal'] = provo_surprisals[s_idx][w_idx]
            idx += 1
    assert idx == len(provo_words)

    # ---- Phase 3: Analysis ----
    for corpus_name, words in [
        ("GECO TEST SET (in-distribution)", geco_words),
        ("PROVO CORPUS (cross-corpus)", provo_words),
    ]:
        print(f"\n\n{'#' * 100}")
        print(f"#  {corpus_name}")
        print(f"{'#' * 100}")
        print(f"  {len(words)} words")

        surps = [w['surprisal'] for w in words]
        l1s = [w['l1'] for w in words]
        print(f"  Surprisal: mean={np.mean(surps):.2f}, std={np.std(surps):.2f}, "
              f"range=[{np.min(surps):.2f}, {np.max(surps):.2f}] bits")
        print(f"  L1:        mean={np.mean(l1s):.1f}, std={np.std(l1s):.1f} ms")

        print_correlation_matrix(words)
        print_human_correlations(words)
        print_partial_correlations(words)
        print_regression_what_predicts_l1(words)
        print_regression_unique_variance(words)
        print_quintile_analysis(words)
        print_key_findings(words, corpus_name.split('(')[0].strip())

    print(f"\n\nDone! Results saved to: {os.path.abspath(results_path)}")


if __name__ == '__main__':
    main()
