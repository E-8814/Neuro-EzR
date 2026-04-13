"""
Training for model_llama_reichle on GECO corpus.

Uses NeuralEZReaderReichle (hybrid Reichle formula + LLM residual + explicit
motor stages + refixation gate + constant pF + parallel skip head).

Differences from train_faithful_sh_geco.py:
  - Collate produces word-level frequencies from SUBTLEX (replaces the
    `predictability` input). Frequencies are looked up per word, OOV -> 1.0.
  - Model is called as model(word_lists, frequencies, word_lengths).
  - Logs the Reichle cognitive parameters each epoch
    (alpha1, alpha2, alpha3, delta, epsilon, M1, M2, I, pF, reg_weight,
    lambda_refix, refix_pivot) so you can watch them drift from literature.
  - Small L2 regularizer on the LLM residual to keep the model close to
    the pure Reichle formula unless data force it away. Makes alpha values
    interpretable.
  - Drops regularizers specific to the old l1_scale calibration scheme.

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 -u src_v2/lm_train/train_reichle_geco.py
  CUDA_VISIBLE_DEVICES=1 python3 -u src_v2/lm_train/train_reichle_geco.py \
      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import csv
import os
import sys
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lm_model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'archive', 'original_ezreader'))

from model_llama_reichle import NeuralEZReaderReichle
from data_loader import aggregate_by_sentence
from geco_loader import load_geco, split_geco


# --------------------------------------------------------------------------- #
#  Hyperparameters
# --------------------------------------------------------------------------- #

LAMBDA_DELTA = 5.0           # keep delta in a sensible range (0.20 - 0.50)
LAMBDA_PRIOR = 10.0          # keep mean skip around SKIP_TARGET
LAMBDA_RESIDUAL = 0.001      # L2 on residual head to keep model near formula
SKIP_TARGET = 0.45


# --------------------------------------------------------------------------- #
#  SUBTLEX frequency lookup
# --------------------------------------------------------------------------- #

def load_subtlex(path):
    """Load SUBTLEXus word frequencies (raw counts)."""
    freq = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            freq[row['Word'].lower()] = int(row['FREQcount'])
    return freq


def word_frequency(token, subtlex):
    """Look up raw SUBTLEX count for a word, with punctuation stripping and
    length-based fallback for OOV items (same as tune_orig_ezreader.py)."""
    w = token.lower().strip(".,;:!?\"'()[]{}").replace("\u2019", "'")
    if w in subtlex:
        return max(1, subtlex[w])
    for variant in (w.replace("'", ""), w.split("'")[0], w.split("-")[0]):
        if variant in subtlex:
            return max(1, subtlex[variant])
    length = len(w)
    if length <= 3:
        return 50000
    if length <= 5:
        return 10000
    if length <= 7:
        return 2000
    return 500


# --------------------------------------------------------------------------- #
#  Collate
# --------------------------------------------------------------------------- #

def _freq_tensor_for_tokens(tokens, subtlex):
    return torch.tensor(
        [float(word_frequency(t, subtlex)) for t in tokens],
        dtype=torch.float32,
    )


def collate_sentences(batch, device, subtlex):
    """Per-participant SentenceData -> tensors for model.forward()."""
    word_lists = [sd.tokens for sd in batch]
    freqs = pad_sequence(
        [_freq_tensor_for_tokens(sd.tokens, subtlex) for sd in batch],
        batch_first=True,
        padding_value=1.0,
    ).to(device)
    wlens = pad_sequence(
        [torch.tensor([len(t) for t in sd.tokens], dtype=torch.float32) for sd in batch],
        batch_first=True,
    ).to(device)
    h_trt = pad_sequence(
        [torch.tensor(sd.total_reading_times, dtype=torch.float32) for sd in batch],
        batch_first=True,
    ).to(device)
    h_ffd = pad_sequence(
        [torch.tensor(sd.first_fixation_durations, dtype=torch.float32) for sd in batch],
        batch_first=True,
    ).to(device)
    h_gaze = pad_sequence(
        [torch.tensor(sd.gaze_durations, dtype=torch.float32) for sd in batch],
        batch_first=True,
    ).to(device)
    h_skip = pad_sequence(
        [torch.tensor([1.0 if s else 0.0 for s in sd.skip_flags], dtype=torch.float32) for sd in batch],
        batch_first=True,
    ).to(device)
    return word_lists, freqs, wlens, h_trt, h_ffd, h_gaze, h_skip


def collate_aggregated(batch, device, subtlex):
    """Per-sentence aggregated data -> tensors for evaluation."""
    word_lists = [a.tokens for a in batch]
    freqs = pad_sequence(
        [_freq_tensor_for_tokens(a.tokens, subtlex) for a in batch],
        batch_first=True,
        padding_value=1.0,
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
    return word_lists, freqs, wlens, h_trt, h_ffd, h_gaze, h_skip


# --------------------------------------------------------------------------- #
#  Logger (tee stdout to a file)
# --------------------------------------------------------------------------- #

class Logger(object):
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
#  Loss
# --------------------------------------------------------------------------- #

def compute_loss(pred, human_trt, human_ffd, human_gaze, human_skip, delta):
    pred_trt = pred['conditional_trt'].float()
    pred_ffd = pred['first_fixation'].float()
    pred_gaze = pred['gaze_duration'].float()
    pred_skip = pred['skip_prob'].float()
    residual = pred['residual'].float()

    # Mask: only fixated words for time losses (skipped words have no FFD/Gaze/TRT).
    fixated = (human_skip < 0.5)

    if fixated.sum() > 0:
        trt_loss = F.mse_loss(pred_trt[fixated], human_trt[fixated])
        ffd_loss = F.mse_loss(pred_ffd[fixated], human_ffd[fixated])
        gaze_loss = F.mse_loss(pred_gaze[fixated], human_gaze[fixated])
    else:
        zero = torch.tensor(0.0, device=pred_trt.device)
        trt_loss = zero
        ffd_loss = zero
        gaze_loss = zero

    # Skip loss: BCE on all words.
    skip_pred = pred_skip.clamp(1e-6, 1 - 1e-6)
    skip_loss = F.binary_cross_entropy(skip_pred, human_skip)

    # Delta regularizer (keep L2/L1 ratio in the range published variants use).
    delta_low = F.relu(0.20 - delta)
    delta_high = F.relu(delta - 0.50)
    delta_reg = LAMBDA_DELTA * (delta_low ** 2 + delta_high ** 2)

    # Skip prior: encourage mean skip rate near SKIP_TARGET.
    mean_skip = pred_skip.mean()
    skip_prior = LAMBDA_PRIOR * (mean_skip - SKIP_TARGET) ** 2

    # Residual regularizer: keep the model close to the pure Reichle formula.
    residual_reg = LAMBDA_RESIDUAL * (residual ** 2).mean()

    total = (
        0.25 * trt_loss
        + 0.25 * ffd_loss
        + 0.25 * gaze_loss
        + 0.4 * skip_loss
        + skip_prior
        + delta_reg
        + residual_reg
    )

    return total, {
        'trt': trt_loss.item(),
        'ffd': ffd_loss.item(),
        'gaze': gaze_loss.item(),
        'skip': skip_loss.item(),
        'residual': residual_reg.item(),
        'total': total.item(),
    }


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #

def evaluate_detailed(model, agg_data, device, subtlex, batch_size=8):
    model.eval()
    all_pred_trt, all_human_trt = [], []
    all_pred_ffd, all_human_ffd = [], []
    all_pred_gaze, all_human_gaze = [], []
    all_pred_skip, all_human_skip = [], []
    all_pred_l1, all_pred_l2 = [], []
    all_residual, all_surprisal = [], []

    with torch.no_grad():
        for i in range(0, len(agg_data), batch_size):
            batch = agg_data[i:i + batch_size]
            word_lists, freqs, wlens, h_trt, h_ffd, h_gaze, h_skip = collate_aggregated(
                batch, device, subtlex
            )

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(word_lists, freqs, wlens)

            for b in range(len(batch)):
                seq_len = len(batch[b].tokens)
                all_pred_trt.extend(pred['conditional_trt'][b, :seq_len].cpu().tolist())
                all_human_trt.extend(batch[b].mean_trt)
                all_pred_ffd.extend(pred['first_fixation'][b, :seq_len].cpu().tolist())
                all_human_ffd.extend(batch[b].mean_ffd)
                all_pred_gaze.extend(pred['gaze_duration'][b, :seq_len].cpu().tolist())
                all_human_gaze.extend(batch[b].mean_gaze)
                all_pred_skip.extend(pred['skip_prob'][b, :seq_len].cpu().tolist())
                all_human_skip.extend(batch[b].skip_rate)
                all_pred_l1.extend(pred['L1'][b, :seq_len].cpu().tolist())
                all_pred_l2.extend(pred['L2'][b, :seq_len].cpu().tolist())
                all_residual.extend(pred['residual'][b, :seq_len].cpu().tolist())
                all_surprisal.extend(pred['word_surprisal'][b, :seq_len].cpu().tolist())

    def corr(a, b):
        a, b = np.array(a), np.array(b)
        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            return np.corrcoef(a, b)[0, 1]
        return 0.0

    pred_trt = np.array(all_pred_trt)
    pred_ffd = np.array(all_pred_ffd)
    pred_gaze = np.array(all_pred_gaze)
    pred_skip = np.array(all_pred_skip)
    human_trt = np.array(all_human_trt)
    human_ffd = np.array(all_human_ffd)
    human_gaze = np.array(all_human_gaze)
    human_skip = np.array(all_human_skip)

    return {
        'r_trt': corr(pred_trt, human_trt),
        'r_ffd': corr(pred_ffd, human_ffd),
        'r_gaze': corr(pred_gaze, human_gaze),
        'r_skip': corr(pred_skip, human_skip),
        'mae_trt': np.mean(np.abs(pred_trt - human_trt)),
        'mae_ffd': np.mean(np.abs(pred_ffd - human_ffd)),
        'mae_gaze': np.mean(np.abs(pred_gaze - human_gaze)),
        'bias_trt': np.mean(pred_trt) - np.mean(human_trt),
        'bias_ffd': np.mean(pred_ffd) - np.mean(human_ffd),
        'bias_gaze': np.mean(pred_gaze) - np.mean(human_gaze),
        'mean_pred_trt': np.mean(pred_trt),
        'mean_human_trt': np.mean(human_trt),
        'mean_l1': np.mean(all_pred_l1),
        'std_l1': np.std(all_pred_l1),
        'mean_l2': np.mean(all_pred_l2),
        'std_l2': np.std(all_pred_l2),
        'mean_skip': np.mean(all_pred_skip),
        'mean_residual_abs': np.mean(np.abs(all_residual)),
        'mean_surprisal': np.mean(all_surprisal),
    }


# --------------------------------------------------------------------------- #
#  Reporting helpers
# --------------------------------------------------------------------------- #

def print_cog_params(model):
    ezr = model.ezreader
    print(f"  Reichle formula parameters:")
    print(f"    alpha1 (base ms)       = {model.alpha1.item():7.2f}    (Reichle 2003: 104)")
    print(f"    alpha2 (freq coef)     = {model.alpha2.item():7.4f}    (Reichle 2003: 3.5)")
    print(f"    alpha3 (surprisal coef)= {model.alpha3.item():7.4f}    (our init 4.0 ms/nat)")
    print(f"    delta  (L2/L1)         = {model.delta.item():7.4f}    (Reichle 2003: 0.34)")
    print(f"  Cognitive cascade:")
    print(f"    epsilon (ecc exponent) = {ezr.epsilon.item():7.4f}    (Reichle: 1.15)")
    print(f"    M1 (labile)    = {ezr.M1.item():7.2f} ms  (Reichle: 125)")
    print(f"    M2 (non-labile)= {ezr.M2.item():7.2f} ms  (Reichle: 25)")
    print(f"    I  (integration)= {ezr.I.item():7.2f} ms  (Reichle: 25)")
    print(f"    pF (integration failure) = {ezr.pF.item():.4f}  (Reichle: ~0.01)")
    print(f"    reg_weight (regression cost) = {ezr.reg_weight.item():.4f}")
    print(f"    lambda_refix     = {ezr.lambda_refix.item():7.4f}")
    print(f"    refix_pivot      = {ezr.refix_pivot.item():7.2f} chars")


def print_sample_predictions(model, agg_data, device, subtlex, n_sentences=3, n_words=8):
    model.eval()
    with torch.no_grad():
        for s_idx in range(min(n_sentences, len(agg_data))):
            s = agg_data[s_idx]
            word_list = [s.tokens]
            freqs = _freq_tensor_for_tokens(s.tokens, subtlex).unsqueeze(0).to(device)
            wl = torch.tensor(
                [len(t) for t in s.tokens], dtype=torch.float32
            ).unsqueeze(0).to(device)
            p = model(word_list, freqs, wl)

            title = ' '.join(s.tokens[:6]) + ('...' if len(s.tokens) > 6 else '')
            print(f"  Sentence {s_idx+1}: \"{title}\"")
            print(f"  {'word':<14s} {'L1':>5s} {'L2':>5s} {'surp':>5s} {'res':>5s} | "
                  f"{'cTRT':>5s} {'hTRT':>5s} {'err':>5s} | "
                  f"{'pFFD':>5s} {'hFFD':>5s} | "
                  f"{'pGaze':>5s} {'hGaze':>5s} | "
                  f"{'skip':>5s} {'hSkip':>5s}")
            print(f"  {'-' * 120}")

            for i in range(min(n_words, len(s.tokens))):
                l1 = p['L1'][0, i].item()
                l2 = p['L2'][0, i].item()
                sp = p['word_surprisal'][0, i].item()
                rs = p['residual'][0, i].item()
                ct = p['conditional_trt'][0, i].item()
                pf = p['first_fixation'][0, i].item()
                pg = p['gaze_duration'][0, i].item()
                ps = p['skip_prob'][0, i].item()
                ht = s.mean_trt[i]
                hf = s.mean_ffd[i]
                hg = s.mean_gaze[i]
                hs = s.skip_rate[i]
                err = ct - ht
                print(
                    f"  {s.tokens[i]:<14s} {l1:5.0f} {l2:5.0f} {sp:5.2f} {rs:+5.1f} | "
                    f"{ct:5.0f} {ht:5.0f} {err:+5.0f} | "
                    f"{pf:5.0f} {hf:5.0f} | "
                    f"{pg:5.0f} {hg:5.0f} | "
                    f"{ps:5.2f} {hs:5.2f}"
                )
            print()


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #

def train(
    data_dir="../data",
    num_epochs=50,
    lm_lr=2e-5,
    head_lr=5e-4,
    cog_lr=1e-3,
    batch_size=8,
    accumulation_steps=8,
    save_dir="../../checkpoints/reichle/geco_tinyllama",
    log_path="../../logs/train_reichle_geco.log",
    seed=42,
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    freeze_layers=12,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- SUBTLEX ----
    subtlex_path = os.path.join(data_dir, "SUBTLEXus.txt")
    print(f"Loading SUBTLEX from {subtlex_path}...")
    subtlex = load_subtlex(subtlex_path)
    print(f"  {len(subtlex):,} entries")

    # ---- GECO data ----
    reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
    material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
    pred_path = os.path.join(data_dir, "geco_predictability.pkl")

    print("Loading GECO Corpus...")
    raw_dataset = load_geco(reading_path, material_path, pred_path)
    print(f"  Raw per-participant observations: {len(raw_dataset):,}")

    train_raw, val_raw, test_raw = split_geco(raw_dataset)
    print(f"  Train: {len(train_raw):,} | Val: {len(val_raw):,} | Test: {len(test_raw):,}")

    aggregated = aggregate_by_sentence(raw_dataset, min_participants=5)
    print(f"  Aggregated sentences (min 5 participants): {len(aggregated)}")

    train_text_ids = set(sd.text_id for sd in train_raw)
    val_text_ids = set(sd.text_id for sd in val_raw)
    train_agg = [a for a in aggregated if a.text_id in train_text_ids]
    val_agg = [a for a in aggregated if a.text_id in val_text_ids]
    test_agg = [a for a in aggregated if a.text_id not in train_text_ids and a.text_id not in val_text_ids]
    print(f"  Aggregated: {len(train_agg)} train | {len(val_agg)} val | {len(test_agg)} test")

    # ---- Model ----
    print(f"\nLoading model: {model_name}")
    print(f"  Freezing first {freeze_layers} layers")
    model = NeuralEZReaderReichle(
        model_name=model_name,
        freeze_layers=freeze_layers,
        hidden_dim=256,
    ).to(device)

    # ---- Optimizer: three param groups (LM, heads, cognitive) ----
    lm_params, head_params, cog_params = [], [], []

    # Cognitive parameter names (attributes of the root model or ezreader).
    cog_name_prefixes = (
        "alpha1", "alpha2", "alpha3", "_delta_raw",
        "ezreader.epsilon",
        "ezreader._M1_raw", "ezreader._M2_raw", "ezreader._I_raw",
        "ezreader.lambda_refix", "ezreader.refix_pivot",
        "ezreader._pF_raw", "ezreader._reg_weight_raw",
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("llama."):
            lm_params.append(param)
        elif any(name.startswith(p) or name == p for p in cog_name_prefixes):
            cog_params.append(param)
        else:
            head_params.append(param)

    n_lm_trainable = sum(p.numel() for p in lm_params)
    n_head_trainable = sum(p.numel() for p in head_params)
    n_cog_trainable = sum(p.numel() for p in cog_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Total parameters:      {total_params:,}")
    print(f"  Frozen (LM):           {n_frozen:,}")
    print(f"  Trainable LM:          {n_lm_trainable:,}   (lr={lm_lr})")
    print(f"  Trainable heads:       {n_head_trainable:,}   (lr={head_lr})")
    print(f"  Trainable cognitive:   {n_cog_trainable}    (lr={cog_lr})")

    optimizer = optim.AdamW([
        {"params": lm_params, "lr": lm_lr, "weight_decay": 0.01},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.0},
        {"params": cog_params, "lr": cog_lr, "weight_decay": 0.0},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-7
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout = Logger(log_path)

    best_val_corr = -1.0

    print("\n" + "=" * 90)
    print(f"Training REICHLE model on GECO")
    print(f"  Model: {model_name}")
    print(f"  L1 = alpha1 - alpha2*ln(freq) + alpha3*surprisal + residual_LLM")
    print(f"  L2 = delta * base_L1 | FFD = L1_ecc + M1 + M2")
    print(f"  Gaze = FFD + P_refix * (L2 + M1 + M2)")
    print(f"  TRT = Gaze + I + pF * reg_weight * prev_gaze")
    print(f"  Skip = learned head (parallel parafoveal), detached from TRT gradient")
    print(f"  Batch: {batch_size} | Accum: {accumulation_steps} | "
          f"Effective: {batch_size * accumulation_steps}")
    print(f"  AMP: {use_amp}")
    print(f"  Regularizers: delta_reg(0.20-0.50) + skip_prior({SKIP_TARGET}) "
          f"+ residual_L2({LAMBDA_RESIDUAL})")
    print("=" * 90)
    print_cog_params(model)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()

        epoch_data = train_raw.copy()
        random.shuffle(epoch_data)

        epoch_loss = 0.0
        epoch_trt = epoch_ffd = epoch_gaze = epoch_skip = epoch_res = 0.0
        n_samples = 0

        optimizer.zero_grad()

        n_batches = (len(epoch_data) + batch_size - 1) // batch_size
        for step in range(n_batches):
            batch = epoch_data[step * batch_size : (step + 1) * batch_size]
            word_lists, freqs, wlens, h_trt, h_ffd, h_gaze, h_skip = collate_sentences(
                batch, device, subtlex
            )

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(word_lists, freqs, wlens)
            loss, parts = compute_loss(pred, h_trt, h_ffd, h_gaze, h_skip, model.delta)

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += parts['total']
            epoch_trt += parts['trt']
            epoch_ffd += parts['ffd']
            epoch_gaze += parts['gaze']
            epoch_skip += parts['skip']
            epoch_res += parts['residual']
            n_samples += len(batch)

        epoch_loss /= n_samples
        epoch_trt /= n_samples
        epoch_ffd /= n_samples
        epoch_gaze /= n_samples
        epoch_skip /= n_samples
        epoch_res /= n_samples
        elapsed = time.time() - t0

        # ---- Validation ----
        val = evaluate_detailed(model, val_agg, device, subtlex)
        scheduler.step(val['mae_trt'])
        lm_lr_now = optimizer.param_groups[0]['lr']
        head_lr_now = optimizer.param_groups[1]['lr']
        cog_lr_now = optimizer.param_groups[2]['lr']

        is_best = val['r_trt'] > best_val_corr

        print(f"\n[Epoch {epoch:3d}/{num_epochs}] {elapsed:.1f}s | "
              f"lm_lr={lm_lr_now:.2e} head_lr={head_lr_now:.2e} cog_lr={cog_lr_now:.2e}")
        print(f"  Train: loss={epoch_loss:.1f} "
              f"(trt={epoch_trt:.0f} ffd={epoch_ffd:.0f} gaze={epoch_gaze:.0f} "
              f"skip={epoch_skip:.3f} res={epoch_res:.3f}) | {n_samples:,} samples")
        print(f"  Val:   r_TRT={val['r_trt']:.3f}  r_FFD={val['r_ffd']:.3f}  "
              f"r_Gaze={val['r_gaze']:.3f}  r_skip={val['r_skip']:.3f}")
        print(f"  Val:   MAE_TRT={val['mae_trt']:.1f}ms  MAE_FFD={val['mae_ffd']:.1f}ms  "
              f"MAE_Gaze={val['mae_gaze']:.1f}ms")
        print(f"  Val:   Bias_TRT={val['bias_trt']:+.1f}ms  Bias_FFD={val['bias_ffd']:+.1f}ms")
        print(f"  Pred:  mean_TRT={val['mean_pred_trt']:.0f}ms "
              f"(human={val['mean_human_trt']:.0f}ms) | "
              f"L1={val['mean_l1']:.0f}+/-{val['std_l1']:.0f}  "
              f"L2={val['mean_l2']:.0f}+/-{val['std_l2']:.0f}  "
              f"mean_skip={val['mean_skip']:.3f}")
        print(f"  Reichle: a1={model.alpha1.item():.2f} a2={model.alpha2.item():.3f} "
              f"a3={model.alpha3.item():.3f} d={model.delta.item():.3f} "
              f"eps={model.ezreader.epsilon.item():.3f} "
              f"M1={model.ezreader.M1.item():.1f} M2={model.ezreader.M2.item():.1f} "
              f"pF={model.ezreader.pF.item():.4f} "
              f"res_abs_mean={val['mean_residual_abs']:.2f} "
              f"surp_mean={val['mean_surprisal']:.2f}")

        print_sample_predictions(model, val_agg, device, subtlex, n_sentences=2, n_words=8)

        if is_best:
            print(f"  ** NEW BEST (r_TRT={val['r_trt']:.3f}) **")
            best_val_corr = val['r_trt']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'model_name': model_name,
                'freeze_layers': freeze_layers,
                'hidden_dim': 256,
                'val_metrics': val,
                'cog_params': {
                    'alpha1': model.alpha1.item(),
                    'alpha2': model.alpha2.item(),
                    'alpha3': model.alpha3.item(),
                    'delta': model.delta.item(),
                    'epsilon': model.ezreader.epsilon.item(),
                    'M1': model.ezreader.M1.item(),
                    'M2': model.ezreader.M2.item(),
                    'I': model.ezreader.I.item(),
                    'pF': model.ezreader.pF.item(),
                    'reg_weight': model.ezreader.reg_weight.item(),
                    'lambda_refix': model.ezreader.lambda_refix.item(),
                    'refix_pivot': model.ezreader.refix_pivot.item(),
                },
            }, os.path.join(save_dir, "best_model.pt"))

    # ---- Final ----
    print("\n" + "=" * 90)
    print(f"Training complete! Best val r_TRT = {best_val_corr:.3f}")
    print("=" * 90)
    print_cog_params(model)

    if test_agg:
        test = evaluate_detailed(model, test_agg, device, subtlex)
        print(f"\nTest set results:")
        print(f"  r_TRT={test['r_trt']:.3f}  r_FFD={test['r_ffd']:.3f}  "
              f"r_Gaze={test['r_gaze']:.3f}  r_skip={test['r_skip']:.3f}")
        print(f"  MAE_TRT={test['mae_trt']:.1f}ms  MAE_FFD={test['mae_ffd']:.1f}ms  "
              f"MAE_Gaze={test['mae_gaze']:.1f}ms")
        print("\nSample test predictions:")
        print_sample_predictions(model, test_agg, device, subtlex, n_sentences=3, n_words=10)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--freeze", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lm_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--cog_lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.freeze is not None:
        freeze_layers = args.freeze
    else:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        n_layers = cfg.num_hidden_layers
        freeze_layers = int(n_layers * 0.75)
        print(f"Auto-freeze: {freeze_layers}/{n_layers} layers")

    model_short = args.model.replace("/", "_")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "checkpoints", "reichle", f"geco_{model_short}"
    )
    log_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "logs", "train_reichle_geco.log"
    )

    train(
        data_dir=data_dir,
        num_epochs=args.epochs,
        lm_lr=args.lm_lr,
        head_lr=args.head_lr,
        cog_lr=args.cog_lr,
        batch_size=args.batch_size,
        accumulation_steps=args.accum,
        save_dir=save_dir,
        log_path=log_path,
        model_name=args.model,
        freeze_layers=freeze_layers,
    )
