"""
Train the hybrid v2 Neural EZ Reader on GECO.

Same architecture as train_hybrid_geco.py but with four targeted changes
to the training recipe:

  (a) Hinge skip prior. Replaces the quadratic skip_prior
          LAMBDA_PRIOR * (mean_skip - 0.45)^2
      with a symmetric hinge
          LAMBDA_PRIOR * (relu(mean_skip - SKIP_MAX) + relu(SKIP_MIN - mean_skip))
      which has a constant, stronger gradient outside [SKIP_MIN, SKIP_MAX].
      Directly targets the skip-collapse failure mode observed in
      reichle_v3 (mean_skip drifted to 0.73 against a quadratic prior
      anchored at 0.45).

  (b) Loss-scale normalization. Each MSE duration loss is divided by a
      typical variance so TRT no longer dominates the gradient relative
      to FFD and gaze. TRT MSE was ~3-4x the gaze MSE and ~10x the FFD
      MSE in the unnormalized loss.

          trt_loss  = MSE(trt)  / SIGMA2_TRT
          ffd_loss  = MSE(ffd)  / SIGMA2_FFD
          gaze_loss = MSE(gaze) / SIGMA2_GAZE

      All losses are then weighted equally (1.0).

  (d) Skip head takes word_length as an extra input. Implemented in
      model_llama_hybrid_v2.NeuralEZReaderHybrid; this script imports
      that model.

  (e) Cosine LR schedule with warmup + best-checkpoint + patience-based
      early stopping. Replaces the ReduceLROnPlateau schedule used in
      train_hybrid_geco.py.
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

from model_llama_hybrid_v2 import NeuralEZReaderHybrid
from data_loader import aggregate_by_sentence
from geco_loader import load_geco, split_geco


# --------------------------------------------------------------------------- #
#  Hyperparameters
# --------------------------------------------------------------------------- #

LAMBDA_DELTA = 5.0
LAMBDA_PRIOR = 20.0
SKIP_MIN = 0.35
SKIP_MAX = 0.55
DELTA_MIN = 0.10
DELTA_MAX = 0.50

# Typical aggregated eye-tracking variances on GECO (rough reference values).
# Used to normalize the three MSE duration losses onto the same scale so
# TRT does not dominate FFD / Gaze in the gradient.
SIGMA2_TRT = 10000.0   # sigma ~ 100 ms
SIGMA2_FFD = 3000.0    # sigma ~ 55 ms
SIGMA2_GAZE = 4500.0   # sigma ~ 67 ms

EARLY_STOP_PATIENCE = 10
WARMUP_EPOCHS = 2


# --------------------------------------------------------------------------- #
#  Collate
# --------------------------------------------------------------------------- #

def collate_sentences(batch, device):
    word_lists = [sd.tokens for sd in batch]
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
    return word_lists, wlens, h_trt, h_ffd, h_gaze, h_skip


def collate_aggregated(batch, device):
    word_lists = [a.tokens for a in batch]
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
    return word_lists, wlens, h_trt, h_ffd, h_gaze, h_skip


# --------------------------------------------------------------------------- #
#  Logger
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

    # (b) Loss-scale normalization.
    trt_loss = trt_mse / SIGMA2_TRT
    ffd_loss = ffd_mse / SIGMA2_FFD
    gaze_loss = gaze_mse / SIGMA2_GAZE

    skip_pred = pred_skip.clamp(1e-6, 1 - 1e-6)
    skip_loss = F.binary_cross_entropy(skip_pred, human_skip)

    delta_low = F.relu(DELTA_MIN - delta)
    delta_high = F.relu(delta - DELTA_MAX)
    delta_reg = LAMBDA_DELTA * (delta_low ** 2 + delta_high ** 2)

    # (a) Symmetric hinge skip prior (constant gradient outside the band).
    mean_skip = pred_skip.mean()
    skip_prior = LAMBDA_PRIOR * (
        F.relu(mean_skip - SKIP_MAX) + F.relu(SKIP_MIN - mean_skip)
    )

    total = (
        1.0 * trt_loss
        + 1.0 * ffd_loss
        + 1.0 * gaze_loss
        + 1.0 * skip_loss
        + skip_prior
        + delta_reg
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
        'total': total.item(),
    }


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #

def evaluate_detailed(model, agg_data, device, batch_size=8):
    model.eval()
    all_pred_trt, all_human_trt = [], []
    all_pred_ffd, all_human_ffd = [], []
    all_pred_gaze, all_human_gaze = [], []
    all_pred_skip, all_human_skip = [], []
    all_pred_l1, all_pred_l2 = [], []

    with torch.no_grad():
        for i in range(0, len(agg_data), batch_size):
            batch = agg_data[i:i + batch_size]
            word_lists, wlens, h_trt, h_ffd, h_gaze, h_skip = collate_aggregated(
                batch, device
            )

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(word_lists, wlens)

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
    }


# --------------------------------------------------------------------------- #
#  Reporting helpers
# --------------------------------------------------------------------------- #

def print_cog_params(model):
    ezr = model.ezreader
    print(f"  Cognitive cascade parameters:")
    print(f"    delta  (L2/L1)         = {model.delta.item():7.4f}    (Reichle 2003: 0.34, allowed [{DELTA_MIN}, {DELTA_MAX}])")
    print(f"    l1_scale               = {model.l1_scale.item():7.2f}")
    print(f"    epsilon (ecc exponent) = {ezr.epsilon.item():7.4f}    (Reichle: 1.15, constrained >= 1)")
    print(f"    M1 (labile)    = {ezr.M1.item():7.2f} ms  (Reichle: 125)")
    print(f"    M2 (non-labile)= {ezr.M2.item():7.2f} ms  (Reichle: 25)")
    print(f"    I  (integration)= {ezr.I.item():7.2f} ms  (Reichle: 25)")
    print(f"    pF (integration failure) = {ezr.pF.item():.4f}  (Reichle: ~0.01)")
    print(f"    reg_weight (regression cost) = {ezr.reg_weight.item():.4f}")
    print(f"    lambda_refix     = {ezr.lambda_refix.item():7.4f}")
    print(f"    refix_pivot      = {ezr.refix_pivot.item():7.2f} chars")
    print(f"    L1 clamp         = [{ezr.L1_MIN}, {ezr.L1_MAX}] ms")


def print_sample_predictions(model, agg_data, device, n_sentences=3, n_words=8):
    model.eval()
    with torch.no_grad():
        for s_idx in range(min(n_sentences, len(agg_data))):
            s = agg_data[s_idx]
            word_list = [s.tokens]
            wl = torch.tensor(
                [len(t) for t in s.tokens], dtype=torch.float32
            ).unsqueeze(0).to(device)
            p = model(word_list, wl)

            title = ' '.join(s.tokens[:6]) + ('...' if len(s.tokens) > 6 else '')
            print(f"  Sentence {s_idx+1}: \"{title}\"")
            print(f"  {'word':<14s} {'L1':>5s} {'L2':>5s} | "
                  f"{'cTRT':>5s} {'hTRT':>5s} {'err':>5s} | "
                  f"{'pFFD':>5s} {'hFFD':>5s} | "
                  f"{'pGaze':>5s} {'hGaze':>5s} | "
                  f"{'skip':>5s} {'hSkip':>5s}")
            print(f"  {'-' * 110}")

            for i in range(min(n_words, len(s.tokens))):
                l1 = p['L1'][0, i].item()
                l2 = p['L2'][0, i].item()
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
                    f"  {s.tokens[i]:<14s} {l1:5.0f} {l2:5.0f} | "
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
    save_dir="../../checkpoints/hybrid_v2/geco_tinyllama",
    log_path="../../logs/train_hybrid_v2_geco.log",
    seed=42,
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    freeze_layers=12,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
        "_delta_raw", "l1_scale",
        "ezreader._epsilon_raw",
        "ezreader._M1_raw", "ezreader._M2_raw", "ezreader._I_raw",
        "ezreader.lambda_refix", "ezreader.refix_pivot",
        "ezreader._pF_raw", "ezreader._reg_weight_raw",
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

    # (e) Cosine schedule with warmup. Applied uniformly to all param groups.
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

    print("\n" + "=" * 90)
    print(f"Training HYBRID v2 model on GECO")
    print(f"  Model: {model_name}")
    print(f"  base_L1 = l1_head(LLaMA) * l1_scale  (neural, no formula)")
    print(f"  L2 = delta * base_L1 | FFD = L1_ecc + M1 + M2")
    print(f"  Gaze = FFD + P_refix * (L2 + M1 + M2)")
    print(f"  TRT = Gaze + I + pF * reg_weight * prev_gaze")
    print(f"  Skip head input: [projected, word_length/10]  (rec d)")
    print(f"  Loss: trt/SIGMA2 + ffd/SIGMA2 + gaze/SIGMA2 + bce  (rec b)")
    print(f"  Skip prior: hinge {LAMBDA_PRIOR}*[relu(skip-{SKIP_MAX})+relu({SKIP_MIN}-skip)]  (rec a)")
    print(f"  LR: cosine warmup={WARMUP_EPOCHS} ep, {total_optimizer_steps} total steps  (rec e)")
    print(f"  Early stop: {EARLY_STOP_PATIENCE} epochs on r_TRT  (rec e)")
    print(f"  Batch: {batch_size} | Accum: {accumulation_steps} | "
          f"Effective: {batch_size * accumulation_steps}")
    print(f"  AMP: {use_amp}")
    print("=" * 90)
    print_cog_params(model)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()

        epoch_data = train_raw.copy()
        random.shuffle(epoch_data)

        epoch_loss = 0.0
        epoch_trt = epoch_ffd = epoch_gaze = epoch_skip = epoch_skip_prior = 0.0
        n_samples = 0

        optimizer.zero_grad()

        n_batches = (len(epoch_data) + batch_size - 1) // batch_size
        for step in range(n_batches):
            batch = epoch_data[step * batch_size : (step + 1) * batch_size]
            word_lists, wlens, h_trt, h_ffd, h_gaze, h_skip = collate_sentences(
                batch, device
            )

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(word_lists, wlens)
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
            n_samples += len(batch)

        epoch_loss /= n_samples
        epoch_trt /= n_samples
        epoch_ffd /= n_samples
        epoch_gaze /= n_samples
        epoch_skip /= n_samples
        epoch_skip_prior /= n_samples
        elapsed = time.time() - t0

        val = evaluate_detailed(model, val_agg, device)
        lm_lr_now = optimizer.param_groups[0]['lr']
        head_lr_now = optimizer.param_groups[1]['lr']
        cog_lr_now = optimizer.param_groups[2]['lr']

        is_best = val['r_trt'] > best_val_corr

        print(f"\n[Epoch {epoch:3d}/{num_epochs}] {elapsed:.1f}s | "
              f"lm_lr={lm_lr_now:.2e} head_lr={head_lr_now:.2e} cog_lr={cog_lr_now:.2e}")
        print(f"  Train: loss={epoch_loss:.4f} "
              f"(trt_mse={epoch_trt:.0f} ffd_mse={epoch_ffd:.0f} gaze_mse={epoch_gaze:.0f} "
              f"skip_bce={epoch_skip:.3f} skip_prior={epoch_skip_prior:.3f}) | {n_samples:,} samples")
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
        print(f"  Cog: d={model.delta.item():.3f} "
              f"l1_scale={model.l1_scale.item():.2f} "
              f"eps={model.ezreader.epsilon.item():.3f} "
              f"M1={model.ezreader.M1.item():.1f} M2={model.ezreader.M2.item():.1f} "
              f"I={model.ezreader.I.item():.1f} "
              f"pF={model.ezreader.pF.item():.4f} "
              f"reg={model.ezreader.reg_weight.item():.3f}")

        print_sample_predictions(model, val_agg, device, n_sentences=2, n_words=8)

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
                    'delta': model.delta.item(),
                    'l1_scale': model.l1_scale.item(),
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
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}: "
                      f"no r_TRT improvement for {EARLY_STOP_PATIENCE} epochs "
                      f"(best={best_val_corr:.3f}).")
                break

    print("\n" + "=" * 90)
    print(f"Training complete! Best val r_TRT = {best_val_corr:.3f}")
    print("=" * 90)
    print_cog_params(model)

    if test_agg:
        test = evaluate_detailed(model, test_agg, device)
        print(f"\nTest set results:")
        print(f"  r_TRT={test['r_trt']:.3f}  r_FFD={test['r_ffd']:.3f}  "
              f"r_Gaze={test['r_gaze']:.3f}  r_skip={test['r_skip']:.3f}")
        print(f"  MAE_TRT={test['mae_trt']:.1f}ms  MAE_FFD={test['mae_ffd']:.1f}ms  "
              f"MAE_Gaze={test['mae_gaze']:.1f}ms")
        print("\nSample test predictions:")
        print_sample_predictions(model, test_agg, device, n_sentences=3, n_words=10)


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
        os.path.dirname(__file__), "..", "..", "checkpoints", "hybrid_v2", f"geco_{model_short}"
    )
    log_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "logs", "train_hybrid_v2_geco.log"
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
