"""
Train the Insanity Neural EZ Reader on GECO.

Recipe differences vs train_hybrid_v4_geco.py:

  - Loss is Gaussian NLL on log(observed FFD / Gaze / TRT), with
    heteroscedastic per-word sigma from the model's variance heads.
    Replaces the MSE-on-means loss used in v1-v4. No SIGMA2 scaling
    constants are needed because NLL is dimensionally consistent
    across the three observables.
  - Skip supervision is BCE on the fixation probability head
    against a fixation indicator (1 - skipped). No hinge skip prior,
    no parafoveal race residual regularization.
  - Parameter groups: lm_params under `lm.*`, cog_params under
    `cascade.*`, head_params for everything else (projections,
    feature_fusion, reader_gru, l1_head, sigma heads, fixation head).
  - Mid-epoch progress line every 1/20 of an epoch so long epochs
    are not silent. Prints running loss parts + elapsed / remaining
    time for the epoch.
  - Early stopping is still on val r_TRT (computed against the
    cascade median, which serves as the point prediction for
    comparability with v1-v4 logs). Val mean NLL is reported
    alongside.

The model is model_llama_insanity.NeuralEZReaderInsanity. Its forward
signature is `model(word_lists, frequencies, word_lengths)` and it
returns a dict containing log-normal parameters (ffd_mu_log, ffd_sigma,
...), cascade medians (ffd_mean, ...), and a fixation probability.
"""

import csv
import math
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

from model_llama_insanity import NeuralEZReaderInsanity, log_normal_nll
from data_loader import aggregate_by_sentence
from geco_loader import load_geco, split_geco


# --------------------------------------------------------------------------- #
#  Hyperparameters
# --------------------------------------------------------------------------- #

LAMBDA_DELTA = 5.0
DELTA_MIN = 0.10
DELTA_MAX = 0.50

EARLY_STOP_PATIENCE = 10
WARMUP_EPOCHS = 2

PROGRESS_FRACTION = 20  # mid-epoch progress lines per epoch


# --------------------------------------------------------------------------- #
#  SUBTLEX frequency lookup (same as v4)
# --------------------------------------------------------------------------- #

def load_subtlex(path):
    freq = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            freq[row['Word'].lower()] = int(row['FREQcount'])
    return freq


def word_frequency(token, subtlex):
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


def _freq_tensor_for_tokens(tokens, subtlex):
    return torch.tensor(
        [float(word_frequency(t, subtlex)) for t in tokens],
        dtype=torch.float32,
    )


# --------------------------------------------------------------------------- #
#  Collate (same shape as v4)
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
#  Logger (same as v4)
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
#  Loss: log-normal NLL + BCE + delta hinge
# --------------------------------------------------------------------------- #

def _masked_lognormal_nll(mu_log, sigma, obs_ms, mask):
    """
    Gaussian NLL on log(obs_ms) with a Jacobian correction. Masked to
    valid entries; if no valid entry, returns a zero scalar.
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=mu_log.device)

    mu_log_m = mu_log[mask]
    sigma_m = sigma[mask]
    obs_m = obs_ms[mask].clamp(min=1.0)
    log_obs = torch.log(obs_m)

    residual = (log_obs - mu_log_m) / sigma_m
    nll = (
        0.5 * residual * residual
        + torch.log(sigma_m)
        + log_obs
        + 0.5 * math.log(2.0 * math.pi)
    )
    return nll.mean()


def compute_loss(pred, human_trt, human_ffd, human_gaze, human_skip, delta):
    ffd_mu_log = pred['ffd_mu_log'].float()
    gaze_mu_log = pred['gaze_mu_log'].float()
    trt_mu_log = pred['trt_mu_log'].float()
    ffd_sigma = pred['ffd_sigma'].float()
    gaze_sigma = pred['gaze_sigma'].float()
    trt_sigma = pred['trt_sigma'].float()
    fixation_prob = pred['fixation_prob'].float()

    fixated = (human_skip < 0.5) & (human_trt > 1.0) & (human_ffd > 1.0) & (human_gaze > 1.0)

    nll_ffd = _masked_lognormal_nll(ffd_mu_log, ffd_sigma, human_ffd, fixated)
    nll_gaze = _masked_lognormal_nll(gaze_mu_log, gaze_sigma, human_gaze, fixated)
    nll_trt = _masked_lognormal_nll(trt_mu_log, trt_sigma, human_trt, fixated)

    fix_target = 1.0 - human_skip.clamp(0.0, 1.0)
    bce_fix = F.binary_cross_entropy(
        fixation_prob.clamp(1e-6, 1 - 1e-6),
        fix_target,
    )

    delta_low = F.relu(DELTA_MIN - delta)
    delta_high = F.relu(delta - DELTA_MAX)
    delta_reg = LAMBDA_DELTA * (delta_low ** 2 + delta_high ** 2)

    total = nll_ffd + nll_gaze + nll_trt + bce_fix + delta_reg

    return total, {
        'nll_ffd': nll_ffd.item(),
        'nll_gaze': nll_gaze.item(),
        'nll_trt': nll_trt.item(),
        'bce_fix': bce_fix.item(),
        'delta_reg': delta_reg.item(),
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
    all_fix_prob, all_human_fix = [], []
    all_base_l1, all_l1_ecc, all_l2 = [], [], []
    all_sigma_ffd, all_sigma_gaze, all_sigma_trt = [], [], []
    all_surprisal = []

    nll_sum, nll_n = 0.0, 0

    with torch.no_grad():
        for i in range(0, len(agg_data), batch_size):
            batch = agg_data[i:i + batch_size]
            word_lists, freqs, wlens, h_trt, h_ffd, h_gaze, h_skip = collate_aggregated(
                batch, device, subtlex
            )

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(word_lists, freqs, wlens)

            # Aggregate-level NLL on each observable (means, not trials,
            # so this is an approximation but comparable across epochs).
            fixated = (h_skip < 0.5) & (h_ffd > 1.0) & (h_gaze > 1.0) & (h_trt > 1.0)
            if fixated.sum() > 0:
                batch_nll = (
                    _masked_lognormal_nll(pred['ffd_mu_log'].float(), pred['ffd_sigma'].float(), h_ffd, fixated)
                    + _masked_lognormal_nll(pred['gaze_mu_log'].float(), pred['gaze_sigma'].float(), h_gaze, fixated)
                    + _masked_lognormal_nll(pred['trt_mu_log'].float(), pred['trt_sigma'].float(), h_trt, fixated)
                ).item()
                nll_sum += batch_nll * int(fixated.sum().item())
                nll_n += int(fixated.sum().item())

            for b in range(len(batch)):
                seq_len = len(batch[b].tokens)
                all_pred_trt.extend(pred['trt_mean'][b, :seq_len].cpu().tolist())
                all_human_trt.extend(batch[b].mean_trt)
                all_pred_ffd.extend(pred['ffd_mean'][b, :seq_len].cpu().tolist())
                all_human_ffd.extend(batch[b].mean_ffd)
                all_pred_gaze.extend(pred['gaze_mean'][b, :seq_len].cpu().tolist())
                all_human_gaze.extend(batch[b].mean_gaze)
                all_fix_prob.extend(pred['fixation_prob'][b, :seq_len].cpu().tolist())
                all_human_fix.extend([1.0 - s for s in batch[b].skip_rate])
                all_base_l1.extend(pred['base_L1'][b, :seq_len].cpu().tolist())
                all_l1_ecc.extend(pred['L1_ecc'][b, :seq_len].cpu().tolist())
                all_l2.extend(pred['L2'][b, :seq_len].cpu().tolist())
                all_sigma_ffd.extend(pred['ffd_sigma'][b, :seq_len].cpu().tolist())
                all_sigma_gaze.extend(pred['gaze_sigma'][b, :seq_len].cpu().tolist())
                all_sigma_trt.extend(pred['trt_sigma'][b, :seq_len].cpu().tolist())
                all_surprisal.extend(pred['surprisal'][b, :seq_len].cpu().tolist())

    def corr(a, b):
        a, b = np.array(a), np.array(b)
        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            return np.corrcoef(a, b)[0, 1]
        return 0.0

    pred_trt = np.array(all_pred_trt)
    pred_ffd = np.array(all_pred_ffd)
    pred_gaze = np.array(all_pred_gaze)
    fix_prob = np.array(all_fix_prob)
    human_trt = np.array(all_human_trt)
    human_ffd = np.array(all_human_ffd)
    human_gaze = np.array(all_human_gaze)
    human_fix = np.array(all_human_fix)

    return {
        'r_trt': corr(pred_trt, human_trt),
        'r_ffd': corr(pred_ffd, human_ffd),
        'r_gaze': corr(pred_gaze, human_gaze),
        'r_fix': corr(fix_prob, human_fix),
        'mae_trt': np.mean(np.abs(pred_trt - human_trt)),
        'mae_ffd': np.mean(np.abs(pred_ffd - human_ffd)),
        'mae_gaze': np.mean(np.abs(pred_gaze - human_gaze)),
        'bias_trt': np.mean(pred_trt) - np.mean(human_trt),
        'bias_ffd': np.mean(pred_ffd) - np.mean(human_ffd),
        'bias_gaze': np.mean(pred_gaze) - np.mean(human_gaze),
        'mean_pred_trt': np.mean(pred_trt),
        'mean_human_trt': np.mean(human_trt),
        'mean_base_l1': np.mean(all_base_l1),
        'std_base_l1': np.std(all_base_l1),
        'mean_l1_ecc': np.mean(all_l1_ecc),
        'std_l1_ecc': np.std(all_l1_ecc),
        'mean_l2': np.mean(all_l2),
        'std_l2': np.std(all_l2),
        'mean_fix': np.mean(fix_prob),
        'mean_sigma_ffd': np.mean(all_sigma_ffd),
        'mean_sigma_gaze': np.mean(all_sigma_gaze),
        'mean_sigma_trt': np.mean(all_sigma_trt),
        'mean_surprisal': np.mean(all_surprisal),
        'std_surprisal': np.std(all_surprisal),
        'val_nll': nll_sum / max(nll_n, 1),
    }


# --------------------------------------------------------------------------- #
#  Reporting helpers
# --------------------------------------------------------------------------- #

def print_cog_params(model):
    c = model.cascade
    print(f"  Cognitive cascade parameters:")
    print(f"    delta  (L2/L1)       = {c.delta.item():7.4f}    (Reichle 2003: 0.34, allowed [{DELTA_MIN}, {DELTA_MAX}])")
    print(f"    epsilon (ecc exp)    = {c.epsilon.item():7.4f}    (Reichle: 1.15, constrained >= 1)")
    print(f"    M1 (labile)          = {c.M1.item():7.2f} ms  (Reichle: 125)")
    print(f"    M2 (non-labile)      = {c.M2.item():7.2f} ms  (Reichle: 25)")
    print(f"    I  (integration)     = {c.I.item():7.2f} ms  (Reichle: 25)")
    print(f"    pF (integ. failure)  = {c.pF.item():.4f}  (Reichle: ~0.01)")
    print(f"    reg_weight           = {c.reg_weight.item():.4f}")
    print(f"    lambda_refix         = {c.lambda_refix.item():7.4f}")
    print(f"    refix_pivot          = {c.refix_pivot.item():7.2f} chars")


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
            print(f"  {'word':<14s} {'surp':>5s} {'bL1':>5s} {'L1e':>5s} {'L2':>5s} | "
                  f"{'TRT':>5s} {'hTRT':>5s} {'err':>5s} | "
                  f"{'FFD':>5s} {'hFFD':>5s} | "
                  f"{'Gaze':>5s} {'hGaze':>5s} | "
                  f"{'sF':>4s} {'sG':>4s} {'sT':>4s} | "
                  f"{'fix':>5s} {'hfix':>5s}")
            print(f"  {'-' * 140}")

            for i in range(min(n_words, len(s.tokens))):
                bl1 = p['base_L1'][0, i].item()
                l1e = p['L1_ecc'][0, i].item()
                l2 = p['L2'][0, i].item()
                ft = p['ffd_mean'][0, i].item()
                gt = p['gaze_mean'][0, i].item()
                tt = p['trt_mean'][0, i].item()
                sf = p['ffd_sigma'][0, i].item()
                sg = p['gaze_sigma'][0, i].item()
                st = p['trt_sigma'][0, i].item()
                fx = p['fixation_prob'][0, i].item()
                su = p['surprisal'][0, i].item()
                ht = s.mean_trt[i]
                hf = s.mean_ffd[i]
                hg = s.mean_gaze[i]
                hs = 1.0 - s.skip_rate[i]
                err = tt - ht
                print(
                    f"  {s.tokens[i]:<14s} {su:5.1f} {bl1:5.0f} {l1e:5.0f} {l2:5.0f} | "
                    f"{tt:5.0f} {ht:5.0f} {err:+5.0f} | "
                    f"{ft:5.0f} {hf:5.0f} | "
                    f"{gt:5.0f} {hg:5.0f} | "
                    f"{sf:4.2f} {sg:4.2f} {st:4.2f} | "
                    f"{fx:5.2f} {hs:5.2f}"
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
    save_dir="../../checkpoints/insanity/geco_tinyllama",
    log_path="../../logs/train_insanity_geco.log",
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
    model = NeuralEZReaderInsanity(
        model_name=model_name,
        freeze_layers=freeze_layers,
        hidden_dim=256,
        gru_hidden=256,
    ).to(device)

    # Parameter groups by top-level namespace.
    lm_params, head_params, cog_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("lm."):
            lm_params.append(param)
        elif name.startswith("cascade."):
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

    progress_interval = max(1, n_batches_per_epoch // PROGRESS_FRACTION)

    print("\n" + "=" * 100)
    print(f"Training INSANITY model on GECO")
    print(f"  Model: {model_name}")
    print(f"  Observation model: log-normal per word")
    print(f"    log FFD  ~ N(log ffd_mean,  sigma_ffd)")
    print(f"    log Gaze ~ N(log gaze_mean, sigma_gaze)")
    print(f"    log TRT  ~ N(log trt_mean,  sigma_trt)")
    print(f"  Inputs: word_repr + log_freq + log_len + surprisal (from lm_head)")
    print(f"  Reader state: GRU({model.hidden_dim} -> {model.gru_hidden}) over words")
    print(f"  Loss: nll_ffd + nll_gaze + nll_trt + bce_fix + delta_reg")
    print(f"  LR: cosine warmup={WARMUP_EPOCHS} ep, {total_optimizer_steps} total steps")
    print(f"  Early stop: {EARLY_STOP_PATIENCE} epochs on r_TRT")
    print(f"  Batch: {batch_size} | Accum: {accumulation_steps} | "
          f"Effective: {batch_size * accumulation_steps}")
    print(f"  AMP: {use_amp}")
    print(f"  Progress: log every 1/{PROGRESS_FRACTION} of an epoch "
          f"({progress_interval} batches)")
    print("=" * 100)
    print_cog_params(model)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()

        epoch_data = train_raw.copy()
        random.shuffle(epoch_data)

        epoch_loss = 0.0
        epoch_nll_ffd = epoch_nll_gaze = epoch_nll_trt = epoch_bce = 0.0
        n_samples = 0

        # Running accumulators for the mid-epoch progress line.
        run_loss = 0.0
        run_nll_ffd = run_nll_gaze = run_nll_trt = run_bce = 0.0
        run_steps = 0
        run_t0 = time.time()

        optimizer.zero_grad()

        n_batches = (len(epoch_data) + batch_size - 1) // batch_size
        for step in range(n_batches):
            batch = epoch_data[step * batch_size : (step + 1) * batch_size]
            word_lists, freqs, wlens, h_trt, h_ffd, h_gaze, h_skip = collate_sentences(
                batch, device, subtlex
            )

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(word_lists, freqs, wlens)
            loss, parts = compute_loss(pred, h_trt, h_ffd, h_gaze, h_skip, model.cascade.delta)

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
            epoch_nll_ffd += parts['nll_ffd']
            epoch_nll_gaze += parts['nll_gaze']
            epoch_nll_trt += parts['nll_trt']
            epoch_bce += parts['bce_fix']
            n_samples += len(batch)

            run_loss += parts['total']
            run_nll_ffd += parts['nll_ffd']
            run_nll_gaze += parts['nll_gaze']
            run_nll_trt += parts['nll_trt']
            run_bce += parts['bce_fix']
            run_steps += 1

            if (step + 1) % progress_interval == 0 and (step + 1) < n_batches:
                frac_done = (step + 1) / n_batches
                dt_chunk = time.time() - run_t0
                dt_epoch = time.time() - t0
                eta_epoch = dt_epoch * (1.0 / frac_done - 1.0)
                print(
                    f"    [ep{epoch:3d} {100*frac_done:3.0f}% "
                    f"{step+1:>5d}/{n_batches}] "
                    f"loss={run_loss/run_steps:.3f} "
                    f"(nFFD={run_nll_ffd/run_steps:.2f} "
                    f"nGaze={run_nll_gaze/run_steps:.2f} "
                    f"nTRT={run_nll_trt/run_steps:.2f} "
                    f"bce={run_bce/run_steps:.3f}) "
                    f"| chunk={dt_chunk:.0f}s eta={eta_epoch:.0f}s"
                )
                run_loss = 0.0
                run_nll_ffd = run_nll_gaze = run_nll_trt = run_bce = 0.0
                run_steps = 0
                run_t0 = time.time()

        epoch_loss /= n_batches
        epoch_nll_ffd /= n_batches
        epoch_nll_gaze /= n_batches
        epoch_nll_trt /= n_batches
        epoch_bce /= n_batches
        elapsed = time.time() - t0

        val = evaluate_detailed(model, val_agg, device, subtlex)
        lm_lr_now = optimizer.param_groups[0]['lr']
        head_lr_now = optimizer.param_groups[1]['lr']
        cog_lr_now = optimizer.param_groups[2]['lr']

        is_best = val['r_trt'] > best_val_corr

        print(f"\n[Epoch {epoch:3d}/{num_epochs}] {elapsed:.1f}s | "
              f"lm_lr={lm_lr_now:.2e} head_lr={head_lr_now:.2e} cog_lr={cog_lr_now:.2e}")
        print(f"  Train: loss={epoch_loss:.4f} "
              f"(nll_ffd={epoch_nll_ffd:.3f} nll_gaze={epoch_nll_gaze:.3f} "
              f"nll_trt={epoch_nll_trt:.3f} bce_fix={epoch_bce:.3f}) "
              f"| {n_samples:,} samples")
        print(f"  Val:   r_TRT={val['r_trt']:.3f}  r_FFD={val['r_ffd']:.3f}  "
              f"r_Gaze={val['r_gaze']:.3f}  r_fix={val['r_fix']:.3f}")
        print(f"  Val:   MAE_TRT={val['mae_trt']:.1f}ms  MAE_FFD={val['mae_ffd']:.1f}ms  "
              f"MAE_Gaze={val['mae_gaze']:.1f}ms  val_NLL={val['val_nll']:.3f}")
        print(f"  Val:   Bias_TRT={val['bias_trt']:+.1f}ms  Bias_FFD={val['bias_ffd']:+.1f}ms")
        print(f"  Pred:  mean_TRT={val['mean_pred_trt']:.0f}ms "
              f"(human={val['mean_human_trt']:.0f}ms)")
        print(f"  Pred:  base_L1={val['mean_base_l1']:.0f}+/-{val['std_base_l1']:.0f}  "
              f"L1_ecc={val['mean_l1_ecc']:.0f}+/-{val['std_l1_ecc']:.0f}  "
              f"L2={val['mean_l2']:.0f}+/-{val['std_l2']:.0f}  "
              f"mean_fix={val['mean_fix']:.3f}")
        print(f"  Sigma: ffd={val['mean_sigma_ffd']:.3f} "
              f"gaze={val['mean_sigma_gaze']:.3f} "
              f"trt={val['mean_sigma_trt']:.3f}")
        print(f"  Surp:  mean={val['mean_surprisal']:.2f}+/-{val['std_surprisal']:.2f}")
        print(f"  Cog: d={model.cascade.delta.item():.3f} "
              f"eps={model.cascade.epsilon.item():.3f} "
              f"M1={model.cascade.M1.item():.1f} "
              f"M2={model.cascade.M2.item():.1f} "
              f"I={model.cascade.I.item():.1f} "
              f"pF={model.cascade.pF.item():.4f} "
              f"reg={model.cascade.reg_weight.item():.3f}")

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
                'gru_hidden': 256,
                'val_metrics': val,
                'cog_params': {
                    'delta': model.cascade.delta.item(),
                    'epsilon': model.cascade.epsilon.item(),
                    'M1': model.cascade.M1.item(),
                    'M2': model.cascade.M2.item(),
                    'I': model.cascade.I.item(),
                    'pF': model.cascade.pF.item(),
                    'reg_weight': model.cascade.reg_weight.item(),
                    'lambda_refix': model.cascade.lambda_refix.item(),
                    'refix_pivot': model.cascade.refix_pivot.item(),
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
              f"r_Gaze={test['r_gaze']:.3f}  r_fix={test['r_fix']:.3f}")
        print(f"  MAE_TRT={test['mae_trt']:.1f}ms  MAE_FFD={test['mae_ffd']:.1f}ms  "
              f"MAE_Gaze={test['mae_gaze']:.1f}ms  test_NLL={test['val_nll']:.3f}")
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
        os.path.dirname(__file__), "..", "..", "checkpoints", "insanity", f"geco_{model_short}"
    )
    log_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "logs", "train_insanity_geco.log"
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
