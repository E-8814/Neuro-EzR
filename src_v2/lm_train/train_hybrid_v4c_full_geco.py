"""
Train hybrid v4c_full on GECO.

Same recipe as train_hybrid_v4c_geco.py with adjustments for the
expanded cognitive parameter set:

  - cog_name_prefixes covers Mψ (4), I/M2/A untied (3),
    lambda_refix, ifail_offset/coef, p_correct.
  - Loss drops the skip prior hinge term (skip dynamics now driven
    purely by race + residual; lambda_refix and integration mechanics
    don't directly constrain skip).
  - Loss drops the v4c-specific `pF, reg_weight` regularizer
    references (those parameters no longer exist).
  - Per-epoch log line reports the new cognitive params.
  - Save paths bumped to hybrid_v4c_full.
"""

import csv
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lm_model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'archive', 'original_ezreader'))

from model_llama_hybrid_v4c_full import NeuralEZReaderHybrid
from data_loader import aggregate_by_sentence
from geco_loader import load_geco, split_geco


LAMBDA_DELTA = 5.0
LAMBDA_PRIOR = 20.0
LAMBDA_SKIP_RESIDUAL = 0.001
SKIP_MIN = 0.35
SKIP_MAX = 0.55
DELTA_MIN = 0.10
DELTA_MAX = 0.50

SIGMA2_TRT = 10000.0
SIGMA2_FFD = 3000.0
SIGMA2_GAZE = 4500.0

EARLY_STOP_PATIENCE = 10
WARMUP_EPOCHS = 2


def load_subtlex(path):
    freq = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            freq[row['Word'].lower()] = int(row['FREQcount'])
    return freq


def word_frequency(token, subtlex):
    w = token.lower().strip(".,;:!?\"'()[]{}").replace("’", "'")
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


def _freq_tensor_for_tokens(tokens, subtlex):
    return torch.tensor(
        [float(word_frequency(t, subtlex)) for t in tokens],
        dtype=torch.float32,
    )


def collate_sentences(batch, device, subtlex):
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


def compute_loss(pred, human_trt, human_ffd, human_gaze, human_skip, delta):
    pred_trt = pred['conditional_trt'].float()
    pred_ffd = pred['first_fixation'].float()
    pred_gaze = pred['gaze_duration'].float()
    pred_skip = pred['skip_prob'].float()
    residual_skip_logit = pred['residual_skip_logit'].float()

    fixated = (human_skip < 0.5)

    if fixated.sum() > 0:
        trt_mse = F.mse_loss(pred_trt[fixated], human_trt[fixated])
        ffd_mse = F.mse_loss(pred_ffd[fixated], human_ffd[fixated])
        gaze_mse = F.mse_loss(pred_gaze[fixated], human_gaze[fixated])
    else:
        zero = torch.tensor(0.0, device=pred_trt.device)
        trt_mse = zero
        ffd_mse = zero
        gaze_mse = zero

    trt_loss = trt_mse / SIGMA2_TRT
    ffd_loss = ffd_mse / SIGMA2_FFD
    gaze_loss = gaze_mse / SIGMA2_GAZE

    skip_pred = pred_skip.clamp(1e-6, 1 - 1e-6)
    skip_loss = F.binary_cross_entropy(skip_pred, human_skip)

    delta_low = F.relu(DELTA_MIN - delta)
    delta_high = F.relu(delta - DELTA_MAX)
    delta_reg = LAMBDA_DELTA * (delta_low ** 2 + delta_high ** 2)

    mean_skip = pred_skip.mean()
    skip_prior = LAMBDA_PRIOR * (
        F.relu(mean_skip - SKIP_MAX) + F.relu(SKIP_MIN - mean_skip)
    )

    skip_residual_reg = LAMBDA_SKIP_RESIDUAL * (residual_skip_logit ** 2).mean()

    total = (
        1.0 * trt_loss
        + 1.0 * ffd_loss
        + 1.0 * gaze_loss
        + 1.0 * skip_loss
        + skip_prior
        + delta_reg
        + skip_residual_reg
    )

    return total, {
        'trt': trt_mse.item(),
        'ffd': ffd_mse.item(),
        'gaze': gaze_mse.item(),
        'trt_norm': trt_loss.item(),
        'ffd_norm': ffd_loss.item(),
        'gaze_norm': gaze_loss.item(),
        'skip': skip_loss.item(),
        'skip_prior': skip_prior.item(),
        'skip_residual_reg': skip_residual_reg.item(),
        'total': total.item(),
    }


def evaluate_detailed(model, agg_data, device, subtlex, batch_size=8):
    model.eval()
    all_pred_trt, all_human_trt = [], []
    all_pred_ffd, all_human_ffd = [], []
    all_pred_gaze, all_human_gaze = [], []
    all_pred_skip, all_human_skip = [], []
    all_pred_l1, all_pred_l2, all_base_l1 = [], [], []
    all_race_logit, all_residual_logit = [], []
    all_ctx, all_formula = [], []
    all_sys_err, all_sigma = [], []
    all_p_fail, all_refix = [], []

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
                all_base_l1.extend(pred['base_L1'][b, :seq_len].cpu().tolist())
                all_race_logit.extend(pred['race_logit'][b, :seq_len].cpu().tolist())
                all_residual_logit.extend(pred['residual_skip_logit'][b, :seq_len].cpu().tolist())
                all_ctx.extend(pred['ctx'][b, :seq_len].cpu().tolist())
                all_formula.extend(pred['base_L1_formula'][b, :seq_len].cpu().tolist())
                all_sys_err.extend(pred['sys_err'][b, :seq_len].cpu().tolist())
                all_sigma.extend(pred['sigma_landing'][b, :seq_len].cpu().tolist())
                all_p_fail.extend(pred['p_fail'][b, :seq_len].cpu().tolist())
                all_refix.extend(pred['refix_prob'][b, :seq_len].cpu().tolist())

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
        'mean_base_l1': np.mean(all_base_l1),
        'std_base_l1': np.std(all_base_l1),
        'mean_l2': np.mean(all_pred_l2),
        'std_l2': np.std(all_pred_l2),
        'mean_skip': np.mean(all_pred_skip),
        'mean_race_logit': np.mean(all_race_logit),
        'std_race_logit': np.std(all_race_logit),
        'mean_residual_logit_abs': np.mean(np.abs(all_residual_logit)),
        'mean_ctx': float(np.mean(all_ctx)),
        'std_ctx': float(np.std(all_ctx)),
        'mean_abs_ctx': float(np.mean(np.abs(all_ctx))),
        'mean_abs_formula': float(np.mean(np.abs(all_formula))),
        'mean_sys_err': float(np.mean(all_sys_err)),
        'std_sys_err': float(np.std(all_sys_err)),
        'mean_sigma': float(np.mean(all_sigma)),
        'std_sigma': float(np.std(all_sigma)),
        'mean_p_fail': float(np.mean(all_p_fail)),
        'std_p_fail': float(np.std(all_p_fail)),
        'mean_refix': float(np.mean(all_refix)),
        'std_refix': float(np.std(all_refix)),
    }


def print_cog_params(model):
    ezr = model.ezreader
    print(f"  Cognitive cascade parameters:")
    print(f"    l1_base_offset  (alpha1_norm) = {model.l1_base_offset.item():7.2f} ms")
    print(f"    l1_freq_coef    (alpha2_norm) = {model.l1_freq_coef.item():7.2f}    (on normed log_freq)")
    print(f"    -- Reichle-unit equivalents --")
    print(f"    alpha1_reichle              = {model.alpha1_reichle.item():7.2f} ms  (Reichle 2003: 104)")
    print(f"    alpha2_reichle              = {model.alpha2_reichle.item():7.4f}     (Reichle 2003: 3.4)")
    print(f"    delta (L2/L1)               = {model.delta.item():7.4f}     (Reichle: 0.34, allowed [{DELTA_MIN}, {DELTA_MAX}])")
    print(f"    epsilon (ecc exp)           = {ezr.epsilon.item():7.4f}     (Reichle: 1.15)")
    print(f"    -- Saccade & integration timing (untied) --")
    print(f"    M1 (labile)                 = {ezr.M1.item():7.2f} ms  (Reichle: 125)")
    print(f"    M2 (non-labile)             = {ezr.M2.item():7.2f} ms  (Reichle: 25)")
    print(f"    I (integration_time)        = {ezr.I.item():7.2f} ms  (Reichle: 25)")
    print(f"    A (attention_shift)         = {ezr.A.item():7.2f} ms  (Reichle: 25)")
    print(f"    -- Mψ (saccade execution noise) --")
    print(f"    omega1                      = {ezr.omega1.item():7.4f}    (Reichle: 6)")
    print(f"    omega2                      = {ezr.omega2.item():7.4f}    (Reichle: 3)")
    print(f"    eta1                        = {ezr.eta1.item():7.4f}    (Reichle: 0.5)")
    print(f"    eta2                        = {ezr.eta2.item():7.4f}    (Reichle: 0.15)")
    print(f"    -- Refixation (landing-driven) --")
    print(f"    lambda_refix                = {ezr.lambda_refix.item():7.4f}   (Reichle: 0.16)")
    print(f"    -- Integration / regression --")
    print(f"    ifail_offset                = {ezr.ifail_offset.item():+7.4f}")
    print(f"    ifail_coef                  = {ezr.ifail_coef.item():+7.4f}   (sign: high freq → low p_fail)")
    print(f"    p_correct_regression        = {ezr.p_correct.item():7.4f}    (Reichle: 0.6)")
    print(f"    --")
    print(f"    skip_temperature            = {ezr.skip_temperature.item():7.2f} ms")
    print(f"    L1 soft floor               = {ezr.L1_SOFT_FLOOR} ms (no hard ceiling)")


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
            print(f"  {'word':<14s} {'bL1':>5s} {'L1':>5s} {'L2':>5s} {'sErr':>5s} {'sig':>4s} {'pf':>4s} {'rfx':>4s} | "
                  f"{'cTRT':>5s} {'hTRT':>5s} | "
                  f"{'pFFD':>5s} {'hFFD':>5s} | "
                  f"{'pGaze':>5s} {'hGaze':>5s} | "
                  f"{'skip':>5s} {'hSkip':>5s}")
            print(f"  {'-' * 150}")

            for i in range(min(n_words, len(s.tokens))):
                bl1 = p['base_L1'][0, i].item()
                l1 = p['L1'][0, i].item()
                l2 = p['L2'][0, i].item()
                se = p['sys_err'][0, i].item()
                sg = p['sigma_landing'][0, i].item()
                pf_v = p['p_fail'][0, i].item()
                rfx = p['refix_prob'][0, i].item()
                ct = p['conditional_trt'][0, i].item()
                pfx = p['first_fixation'][0, i].item()
                pg = p['gaze_duration'][0, i].item()
                ps = p['skip_prob'][0, i].item()
                ht = s.mean_trt[i]
                hf = s.mean_ffd[i]
                hg = s.mean_gaze[i]
                hs = s.skip_rate[i]
                print(
                    f"  {s.tokens[i]:<14s} {bl1:5.0f} {l1:5.0f} {l2:5.0f} {se:+5.2f} {sg:4.2f} {pf_v:4.2f} {rfx:4.2f} | "
                    f"{ct:5.0f} {ht:5.0f} | "
                    f"{pfx:5.0f} {hf:5.0f} | "
                    f"{pg:5.0f} {hg:5.0f} | "
                    f"{ps:5.2f} {hs:5.2f}"
                )
            print()


def train(
    data_dir="../data",
    num_epochs=50,
    lm_lr=2e-5,
    head_lr=5e-4,
    cog_lr=1e-3,
    batch_size=8,
    accumulation_steps=8,
    save_dir="../../checkpoints/hybrid_v4c_full/geco_tinyllama",
    log_path="../../logs/train_hybrid_v4c_full_geco.log",
    seed=42,
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    freeze_layers=12,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    subtlex_path = os.path.join(data_dir, "SUBTLEXus.txt")
    print(f"Loading SUBTLEX from {subtlex_path}...")
    subtlex = load_subtlex(subtlex_path)
    print(f"  {len(subtlex):,} entries")

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

    print(f"\nLoading model: {model_name}")
    print(f"  Freezing first {freeze_layers} layers")
    model = NeuralEZReaderHybrid(
        model_name=model_name,
        freeze_layers=freeze_layers,
        hidden_dim=256,
    ).to(device)

    cog_name_prefixes = (
        "_delta_raw",
        "l1_base_offset", "l1_freq_coef",
        "ezreader._epsilon_raw",
        "ezreader._M1_raw", "ezreader._M2_raw",
        "ezreader._I_raw", "ezreader._A_raw",
        "ezreader._omega1_raw", "ezreader._omega2_raw",
        "ezreader._eta1_raw", "ezreader._eta2_raw",
        "ezreader._lambda_refix_raw",
        "ezreader.ifail_offset", "ezreader.ifail_coef",
        "ezreader._p_correct_raw",
        "ezreader._skip_temperature_raw",
    )

    lm_params, head_params, cog_params = [], [], []
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
    print(f"  Trainable cognitive:   {n_cog_trainable}    (lr={cog_lr})  (Mψ + I/M2/A untied + ifail + p_correct + lambda_refix)")

    optimizer = optim.AdamW([
        {"params": lm_params, "lr": lm_lr, "weight_decay": 0.01},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.0},
        {"params": cog_params, "lr": cog_lr, "weight_decay": 0.0},
    ])

    n_batches_per_epoch = (len(train_raw) + batch_size - 1) // batch_size
    optimizer_steps_per_epoch = (n_batches_per_epoch + accumulation_steps - 1) // accumulation_steps
    total_optimizer_steps = num_epochs * optimizer_steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * optimizer_steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout = Logger(log_path)

    best_val_corr = -1.0
    patience_counter = 0

    print("\n" + "=" * 100)
    print(f"Training HYBRID v4c_full on GECO  (v4c + Mψ + landing-refix + integration + regression direction)")
    print(f"  Model: {model_name}")
    print(f"  base_L1 = alpha1 + alpha2 * (log(freq)-10)/5  + ctx_head(LLaMA_hidden)")
    print(f"  Mψ: sys_err = (7 - intended_sac_len)*(omega1 - log launch_dur)/omega2")
    print(f"      sigma   = eta1 + eta2 * intended_sac_len")
    print(f"      E[L1_ecc] = base_L1 * eps^(sys_err + (w-1)/2) * exp(0.5 sigma^2 (ln eps)^2)")
    print(f"  Refix: P_refix = clamp(lambda_refix * E[|landing - target|], 0, 0.95)")
    print(f"  L2 = delta * base_L1 | FFD = E[L1_ecc] + M1 + M2")
    print(f"  Gaze = FFD + P_refix * (L2 + M1 + M2)")
    print(f"  Integration: p_fail = sigmoid(ifail_offset + ifail_coef * log_freq_norm)")
    print(f"  Regression cost = p_correct * (M1+M2+0.1*L1+L2+I)")
    print(f"                  + (1-p_correct) * (M1+M2+0.1*L1[n-1]+L2[n-1]+I+A)")
    print(f"  TRT = Gaze + I + p_fail * regression_cost")
    print(f"  Skip = sigmoid((M1 - L1_next_parafoveal)/skip_temperature + residual)")
    print(f"  Loss: trt/SIGMA2 + ffd/SIGMA2 + gaze/SIGMA2 + bce + residual_reg + skip_prior")
    print(f"  LR: cosine warmup={WARMUP_EPOCHS} ep, {total_optimizer_steps} total steps")
    print(f"  Early stop: {EARLY_STOP_PATIENCE} epochs on r_TRT")
    print(f"  Batch: {batch_size} | Accum: {accumulation_steps} | "
          f"Effective: {batch_size * accumulation_steps}")
    print(f"  AMP: {use_amp}")
    print("=" * 100)
    print_cog_params(model)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()

        epoch_data = train_raw.copy()
        random.shuffle(epoch_data)

        epoch_loss = 0.0
        epoch_trt = epoch_ffd = epoch_gaze = epoch_skip = 0.0
        epoch_skip_prior = epoch_res_reg = 0.0
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
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += parts['total']
            epoch_trt += parts['trt']
            epoch_ffd += parts['ffd']
            epoch_gaze += parts['gaze']
            epoch_skip += parts['skip']
            epoch_skip_prior += parts['skip_prior']
            epoch_res_reg += parts['skip_residual_reg']
            n_samples += len(batch)

        epoch_loss /= n_samples
        epoch_trt /= n_samples
        epoch_ffd /= n_samples
        epoch_gaze /= n_samples
        epoch_skip /= n_samples
        epoch_skip_prior /= n_samples
        epoch_res_reg /= n_samples
        elapsed = time.time() - t0

        val = evaluate_detailed(model, val_agg, device, subtlex)
        lm_lr_now = optimizer.param_groups[0]['lr']
        head_lr_now = optimizer.param_groups[1]['lr']
        cog_lr_now = optimizer.param_groups[2]['lr']

        is_best = val['r_trt'] > best_val_corr

        print(f"\n[Epoch {epoch:3d}/{num_epochs}] {elapsed:.1f}s | "
              f"lm_lr={lm_lr_now:.2e} head_lr={head_lr_now:.2e} cog_lr={cog_lr_now:.2e}")
        print(f"  Train: loss={epoch_loss:.4f} "
              f"(trt_mse={epoch_trt:.0f} ffd_mse={epoch_ffd:.0f} gaze_mse={epoch_gaze:.0f} "
              f"skip_bce={epoch_skip:.3f} skip_prior={epoch_skip_prior:.3f} "
              f"res_reg={epoch_res_reg:.4f}) | {n_samples:,} samples")
        print(f"  Val:   r_TRT={val['r_trt']:.3f}  r_FFD={val['r_ffd']:.3f}  "
              f"r_Gaze={val['r_gaze']:.3f}  r_skip={val['r_skip']:.3f}")
        print(f"  Val:   MAE_TRT={val['mae_trt']:.1f}ms  MAE_FFD={val['mae_ffd']:.1f}ms  "
              f"MAE_Gaze={val['mae_gaze']:.1f}ms")
        print(f"  Val:   Bias_TRT={val['bias_trt']:+.1f}ms  Bias_FFD={val['bias_ffd']:+.1f}ms")
        print(f"  Pred:  base_L1={val['mean_base_l1']:.0f}+/-{val['std_base_l1']:.0f}  "
              f"L1_ecc={val['mean_l1']:.0f}+/-{val['std_l1']:.0f}  "
              f"L2={val['mean_l2']:.0f}+/-{val['std_l2']:.0f}  "
              f"mean_skip={val['mean_skip']:.3f}")
        print(f"  Mψ:    sys_err={val['mean_sys_err']:+.2f}+/-{val['std_sys_err']:.2f}  "
              f"sigma={val['mean_sigma']:.2f}+/-{val['std_sigma']:.2f}  "
              f"refix={val['mean_refix']:.3f}+/-{val['std_refix']:.3f}")
        print(f"  Integ: p_fail={val['mean_p_fail']:.3f}+/-{val['std_p_fail']:.3f}")
        print(f"  Cog: d={model.delta.item():.3f} "
              f"a1R={model.alpha1_reichle.item():.1f} "
              f"a2R={model.alpha2_reichle.item():.3f} "
              f"eps={model.ezreader.epsilon.item():.3f} "
              f"M1={model.ezreader.M1.item():.1f} M2={model.ezreader.M2.item():.1f} "
              f"I={model.ezreader.I.item():.1f} A={model.ezreader.A.item():.1f}")
        print(f"       w1={model.ezreader.omega1.item():.2f} w2={model.ezreader.omega2.item():.2f} "
              f"e1={model.ezreader.eta1.item():.3f} e2={model.ezreader.eta2.item():.3f} "
              f"lam={model.ezreader.lambda_refix.item():.3f} "
              f"pcr={model.ezreader.p_correct.item():.3f}")

        print_sample_predictions(model, val_agg, device, subtlex, n_sentences=2, n_words=8)

        if is_best:
            print(f"  ** NEW BEST (r_TRT={val['r_trt']:.3f}) **")
            best_val_corr = val['r_trt']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'model_name': model_name,
                'freeze_layers': freeze_layers,
                'hidden_dim': 256,
                'val_metrics': val,
                'cog_params': {
                    'l1_base_offset': model.l1_base_offset.item(),
                    'l1_freq_coef': model.l1_freq_coef.item(),
                    'alpha1_reichle': model.alpha1_reichle.item(),
                    'alpha2_reichle': model.alpha2_reichle.item(),
                    'delta': model.delta.item(),
                    'epsilon': model.ezreader.epsilon.item(),
                    'M1': model.ezreader.M1.item(),
                    'M2': model.ezreader.M2.item(),
                    'I': model.ezreader.I.item(),
                    'A': model.ezreader.A.item(),
                    'omega1': model.ezreader.omega1.item(),
                    'omega2': model.ezreader.omega2.item(),
                    'eta1': model.ezreader.eta1.item(),
                    'eta2': model.ezreader.eta2.item(),
                    'lambda_refix': model.ezreader.lambda_refix.item(),
                    'p_correct': model.ezreader.p_correct.item(),
                    'ifail_offset': model.ezreader.ifail_offset.item(),
                    'ifail_coef': model.ezreader.ifail_coef.item(),
                    'skip_temperature': model.ezreader.skip_temperature.item(),
                },
            }, os.path.join(save_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}: "
                      f"no r_TRT improvement for {EARLY_STOP_PATIENCE} epochs "
                      f"(best={best_val_corr:.3f}).")
                break

    print("\n" + "=" * 100)
    print(f"Training complete! Best val r_TRT = {best_val_corr:.3f}")
    print("=" * 100)
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
    parser.add_argument("--seed", type=int, default=42)
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
        os.path.dirname(__file__), "..", "..", "checkpoints",
        "hybrid_v4c_full",
        f"geco_{model_short}_seed{args.seed}",
    )
    log_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "logs",
        f"train_hybrid_v4c_full_geco_seed{args.seed}.log",
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
        seed=args.seed,
        model_name=args.model,
        freeze_layers=freeze_layers,
    )
