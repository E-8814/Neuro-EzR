"""
Train v4c_v2_wide_prior with the SKIPFREEZE staged approach
(refined Option G).

Phase 1 (epoch 1): standard training. We track TWO checkpoints in
parallel:
  - `best_r_skip_phase1.pt` — saved whenever val r_skip is higher than
    any previously-seen value during phase 1. This is the *r_skip peak*
    snapshot.
  - `best_combined_phase1.pt` — saved on the standard combined-metric
    early-stopping criterion.

Phase 2 (epochs 2..end):
  - Reload `best_r_skip_phase1.pt` weights.
  - Freeze `skip_residual_head` (requires_grad=False on its parameters).
  - Continue training with the full loss (TRT + FFD + Gaze + skip BCE
    + skip prior + delta_reg + skip_residual_reg). Skip BCE still flows
    through cog/ctx via race_logit, providing implicit skip feedback,
    but the residual head itself is locked at its phase-1-peak state.
  - Track `best_model.pt` on the standard combined-metric criterion.

Final saved model is `best_model.pt` from phase 2 (or
`best_combined_phase1.pt` if training was 1 epoch only).

Empirical justification (from wide_prior log over 3 epochs):
  - M1 drift: −0.4%, ε drift: −0.5% — race_logit shift ~0.08 logit units
  - Predicted r_skip drift in phase 2: ~0.01-0.02 (negligible)
  - So freezing skip_residual_head alone is sufficient; no need to also
    freeze M1 / ε / skip_temperature.

Everything else identical to wide_prior:
  - WIDE skip prior bounds [0.35, 0.55], LAMBDA_PRIOR = 30
  - LAMBDA_SKIP_RESIDUAL = 0.001
  - First-word skip mask
  - Mid-epoch validation (5/epoch)
  - SIGMA2_FFD = 1500
  - cog_lr = 3e-4, head_lr = 5e-4, lm_lr = 2e-5
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

from model_llama_hybrid_v4c_v2 import NeuralEZReaderHybrid
from data_loader import aggregate_by_sentence
from geco_loader import load_geco, split_geco


LAMBDA_DELTA = 5.0
LAMBDA_PRIOR = 30.0
LAMBDA_SKIP_RESIDUAL = 0.001

SKIP_MIN = 0.35
SKIP_MAX = 0.55

DELTA_MIN = 0.10
DELTA_MAX = 0.50

SIGMA2_TRT = 10000.0
SIGMA2_FFD = 1500.0
SIGMA2_GAZE = 4500.0

EARLY_STOP_PATIENCE_VALS = 15
WARMUP_EPOCHS = 2
N_VALS_PER_EPOCH = 5


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
    if length <= 3: return 50000
    if length <= 5: return 10000
    if length <= 7: return 2000
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
        batch_first=True, padding_value=1.0,
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
        batch_first=True, padding_value=1.0,
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


def compute_loss(pred, h_trt, h_ffd, h_gaze, h_skip, delta):
    pred_trt = pred['conditional_trt'].float()
    pred_ffd = pred['first_fixation'].float()
    pred_gaze = pred['gaze_duration'].float()
    pred_skip = pred['skip_prob'].float()
    residual_skip_logit = pred['residual_skip_logit'].float()

    fixated = (h_skip < 0.5)
    if fixated.sum() > 0:
        trt_mse = F.mse_loss(pred_trt[fixated], h_trt[fixated])
        ffd_mse = F.mse_loss(pred_ffd[fixated], h_ffd[fixated])
        gaze_mse = F.mse_loss(pred_gaze[fixated], h_gaze[fixated])
    else:
        zero = torch.tensor(0.0, device=pred_trt.device)
        trt_mse = ffd_mse = gaze_mse = zero

    trt_loss = trt_mse / SIGMA2_TRT
    ffd_loss = ffd_mse / SIGMA2_FFD
    gaze_loss = gaze_mse / SIGMA2_GAZE

    skip_pred = pred_skip.clamp(1e-6, 1 - 1e-6)
    skip_loss = F.binary_cross_entropy(skip_pred, h_skip)

    delta_low = F.relu(DELTA_MIN - delta)
    delta_high = F.relu(delta - DELTA_MAX)
    delta_reg = LAMBDA_DELTA * (delta_low ** 2 + delta_high ** 2)

    mean_skip = pred_skip.mean()
    skip_prior = LAMBDA_PRIOR * (
        F.relu(mean_skip - SKIP_MAX) + F.relu(SKIP_MIN - mean_skip)
    )

    skip_residual_reg = LAMBDA_SKIP_RESIDUAL * (residual_skip_logit ** 2).mean()

    total = (
        trt_loss + ffd_loss + gaze_loss + skip_loss
        + skip_prior + delta_reg + skip_residual_reg
    )

    return total, {
        'trt': trt_mse.item(), 'ffd': ffd_mse.item(), 'gaze': gaze_mse.item(),
        'skip': skip_loss.item(), 'skip_prior': skip_prior.item(),
        'skip_residual_reg': skip_residual_reg.item(),
        'total': total.item(),
    }


def evaluate_detailed(model, agg_data, device, subtlex, batch_size=8):
    model.eval()
    pt, ht, pf_, hf_, pg, hg, ps, hs = [], [], [], [], [], [], [], []
    rl_, res_ = [], []
    with torch.no_grad():
        for i in range(0, len(agg_data), batch_size):
            batch = agg_data[i:i + batch_size]
            word_lists, freqs, wlens, *_ = collate_aggregated(batch, device, subtlex)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(word_lists, freqs, wlens)
            for b in range(len(batch)):
                seq_len = len(batch[b].tokens)
                pt.extend(pred['conditional_trt'][b, :seq_len].cpu().tolist())
                ht.extend(batch[b].mean_trt)
                pf_.extend(pred['first_fixation'][b, :seq_len].cpu().tolist())
                hf_.extend(batch[b].mean_ffd)
                pg.extend(pred['gaze_duration'][b, :seq_len].cpu().tolist())
                hg.extend(batch[b].mean_gaze)
                ps.extend(pred['skip_prob'][b, :seq_len].cpu().tolist())
                hs.extend(batch[b].skip_rate)
                rl_.extend(pred['race_logit'][b, :seq_len].cpu().tolist())
                res_.extend(pred['residual_skip_logit'][b, :seq_len].cpu().tolist())

    def corr(a, b):
        a, b = np.array(a), np.array(b)
        if len(a) > 2 and a.std() > 0 and b.std() > 0:
            return float(np.corrcoef(a, b)[0, 1])
        return 0.0

    pt_a, ht_a = np.array(pt), np.array(ht)
    pf_a, hf_a = np.array(pf_), np.array(hf_)
    pg_a, hg_a = np.array(pg), np.array(hg)
    ps_a, hs_a = np.array(ps), np.array(hs)
    return {
        'r_trt': corr(pt_a, ht_a), 'r_ffd': corr(pf_a, hf_a),
        'r_gaze': corr(pg_a, hg_a), 'r_skip': corr(ps_a, hs_a),
        'mae_trt': float(np.mean(np.abs(pt_a - ht_a))),
        'mae_ffd': float(np.mean(np.abs(pf_a - hf_a))),
        'mae_gaze': float(np.mean(np.abs(pg_a - hg_a))),
        'bias_trt': float(np.mean(pt_a) - np.mean(ht_a)),
        'bias_ffd': float(np.mean(pf_a) - np.mean(hf_a)),
        'mean_skip': float(np.mean(ps_a)),
        'std_skip': float(np.std(ps_a)),
        'mean_race_logit': float(np.mean(rl_)),
        'std_race_logit': float(np.std(rl_)),
        'mean_residual_logit_abs': float(np.mean(np.abs(res_))),
    }


def combined_metric(val):
    return 0.25 * (val['r_trt'] + val['r_ffd'] + val['r_gaze'] + val['r_skip'])


def save_checkpoint(model, save_path, epoch, val_step, val,
                    model_name, freeze_layers, phase, skip_frozen,
                    selection_criterion):
    """Save a checkpoint. `selection_criterion` describes why this was saved."""
    torch.save({
        'epoch': epoch, 'val_step': val_step,
        'model_state_dict': model.state_dict(),
        'model_name': model_name, 'freeze_layers': freeze_layers,
        'hidden_dim': 256,
        'phase': phase,
        'skip_residual_frozen': skip_frozen,
        'selection_criterion': selection_criterion,
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
            'lambda_refix': model.ezreader.lambda_refix.item(),
            'refix_pivot': model.ezreader.refix_pivot.item(),
            'skip_temperature': model.ezreader.skip_temperature.item(),
        },
    }, save_path)


def freeze_skip_residual(model):
    n = 0
    for p in model.skip_residual_head.parameters():
        p.requires_grad = False
        n += p.numel()
    print(f"  >> Froze skip_residual_head ({n:,} params).")


def train(data_dir, num_epochs, lm_lr, head_lr, cog_lr,
          batch_size, accumulation_steps, save_dir, log_path,
          seed, model_name, freeze_layers, phase1_epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    subtlex = load_subtlex(os.path.join(data_dir, "SUBTLEXus.txt"))

    raw = load_geco(
        os.path.join(data_dir, "Geco_MonolingualReadingData.csv"),
        os.path.join(data_dir, "Geco_EnglishMaterial.csv"),
        os.path.join(data_dir, "geco_predictability.pkl"),
    )
    train_raw, val_raw, _ = split_geco(raw)
    aggregated = aggregate_by_sentence(raw, min_participants=5)
    train_ids = set(s.text_id for s in train_raw)
    val_ids = set(s.text_id for s in val_raw)
    train_agg = [a for a in aggregated if a.text_id in train_ids]
    val_agg = [a for a in aggregated if a.text_id in val_ids]
    test_agg = [a for a in aggregated if a.text_id not in train_ids and a.text_id not in val_ids]
    print(f"  Aggregated: {len(train_agg)} train | {len(val_agg)} val | {len(test_agg)} test")

    print(f"  Skip prior: WIDE bounds [{SKIP_MIN}, {SKIP_MAX}], LAMBDA={LAMBDA_PRIOR}")
    print(f"  Skip residual L2 reg: lambda={LAMBDA_SKIP_RESIDUAL}")
    print(f"  Phase 1: {phase1_epochs} epoch(s) — train everything, save best-r_skip checkpoint")
    print(f"  Phase 2: {num_epochs - phase1_epochs} epoch(s) — reload best-r_skip, freeze skip_residual, continue")

    print(f"\nLoading model: {model_name}")
    model = NeuralEZReaderHybrid(
        model_name=model_name, freeze_layers=freeze_layers, hidden_dim=256,
    ).to(device)

    cog_prefixes = (
        "_delta_raw",
        "l1_base_offset", "l1_freq_coef",
        "ezreader._epsilon_raw",
        "ezreader._M1_raw", "ezreader._M2I_raw",
        "ezreader.lambda_refix", "ezreader.refix_pivot",
        "ezreader._skip_temperature_raw",
    )

    lm_p, head_p, cog_p = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.startswith("llama."):
            lm_p.append(param)
        elif any(name.startswith(p) or name == p for p in cog_prefixes):
            cog_p.append(param)
        else:
            head_p.append(param)

    optimizer = optim.AdamW([
        {"params": lm_p, "lr": lm_lr, "weight_decay": 0.01},
        {"params": head_p, "lr": head_lr, "weight_decay": 0.0},
        {"params": cog_p, "lr": cog_lr, "weight_decay": 0.0},
    ])

    n_batches_per_epoch = (len(train_raw) + batch_size - 1) // batch_size
    opt_steps_per_epoch = (n_batches_per_epoch + accumulation_steps - 1) // accumulation_steps
    total_steps = num_epochs * opt_steps_per_epoch
    warmup = WARMUP_EPOCHS * opt_steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout = Logger(log_path)

    best_r_skip_phase1 = -1.0          # Phase 1: best r_skip seen so far
    best_r_skip_phase1_path = os.path.join(save_dir, "best_r_skip_phase1.pt")
    best_combined_phase1 = -1.0        # Phase 1: best combined seen so far
    best_combined_phase1_path = os.path.join(save_dir, "best_combined_phase1.pt")

    best_combined_phase2 = -1.0        # Phase 2: tracks final best_model.pt
    best_model_path = os.path.join(save_dir, "best_model.pt")

    patience = 0
    total_val_steps = 0
    early_stop = False
    val_every = max(1, n_batches_per_epoch // N_VALS_PER_EPOCH)
    skip_frozen = False
    phase = 1

    print("\n" + "=" * 100)
    print(f"v4c_v2_wide_prior_skipfreeze (refined Option G)")
    print(f"  Phase 1 ({phase1_epochs} ep): train everything, track best-r_skip")
    print(f"  Phase 2 ({num_epochs - phase1_epochs} ep): reload best-r_skip, freeze skip_residual, continue")
    print("=" * 100)

    for epoch in range(1, num_epochs + 1):
        if early_stop: break

        # --- Phase transition: end of Phase 1 ---
        if epoch == phase1_epochs + 1 and not skip_frozen:
            print(f"\n>>> [Epoch {epoch}] Phase 1 → Phase 2 transition")
            if best_r_skip_phase1 < 0:
                print("    WARNING: no Phase 1 checkpoint saved. Continuing without reload.")
            else:
                ckpt = torch.load(best_r_skip_phase1_path, map_location=device,
                                  weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"    Reloaded best-r_skip from Phase 1 (r_skip={ckpt['val_metrics']['r_skip']:.3f}, "
                      f"val_step={ckpt['val_step']}).")
            freeze_skip_residual(model)
            skip_frozen = True
            phase = 2
            # Reset best-combined tracker for phase 2
            best_combined_phase2 = -1.0
            patience = 0

        t0 = time.time()
        model.train()

        epoch_data = train_raw.copy()
        random.shuffle(epoch_data)
        n_batches = (len(epoch_data) + batch_size - 1) // batch_size

        optimizer.zero_grad()
        for step in range(n_batches):
            batch = epoch_data[step * batch_size:(step + 1) * batch_size]
            word_lists, freqs, wlens, h_trt, h_ffd, h_gaze, h_skip = \
                collate_sentences(batch, device, subtlex)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(word_lists, freqs, wlens)
            loss, _ = compute_loss(pred, h_trt, h_ffd, h_gaze, h_skip, model.delta)
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            is_last = (step + 1) == n_batches
            if (step + 1) % val_every == 0 or is_last:
                total_val_steps += 1
                val = evaluate_detailed(model, val_agg, device, subtlex)
                combined = combined_metric(val)

                phase_label = f"phase {phase}" + (" (skip frozen)" if skip_frozen else "")
                print(f"\n  [val {total_val_steps}] epoch {epoch} batch {step+1}/{n_batches} | {phase_label}")
                print(f"    combined={combined:.4f} | r_TRT={val['r_trt']:.3f} "
                      f"r_FFD={val['r_ffd']:.3f} r_Gaze={val['r_gaze']:.3f} r_skip={val['r_skip']:.3f}")
                print(f"    Bias_TRT={val['bias_trt']:+.1f} Bias_FFD={val['bias_ffd']:+.1f} "
                      f"| mean_skip={val['mean_skip']:.3f} std_skip={val['std_skip']:.3f}")
                print(f"    Skip: race={val['mean_race_logit']:+.2f}±{val['std_race_logit']:.2f} "
                      f"residual_abs={val['mean_residual_logit_abs']:.3f}")
                print(f"    Cog: a1R={model.alpha1_reichle.item():.1f} "
                      f"eps={model.ezreader.epsilon.item():.3f} "
                      f"M1={model.ezreader.M1.item():.1f} d={model.delta.item():.3f}")

                if phase == 1:
                    # Track BOTH best-r_skip AND best-combined during phase 1
                    if val['r_skip'] > best_r_skip_phase1:
                        print(f"    ** PHASE 1 NEW BEST r_skip ({val['r_skip']:.3f}) **")
                        best_r_skip_phase1 = val['r_skip']
                        save_checkpoint(
                            model, best_r_skip_phase1_path, epoch, total_val_steps, val,
                            model_name, freeze_layers, phase=1, skip_frozen=False,
                            selection_criterion="best_r_skip_phase1",
                        )
                    if combined > best_combined_phase1:
                        best_combined_phase1 = combined
                        save_checkpoint(
                            model, best_combined_phase1_path, epoch, total_val_steps, val,
                            model_name, freeze_layers, phase=1, skip_frozen=False,
                            selection_criterion="best_combined_phase1",
                        )
                else:
                    # Phase 2: track best combined as the final model
                    if combined > best_combined_phase2:
                        print(f"    ** PHASE 2 NEW BEST combined ({combined:.4f}) **")
                        best_combined_phase2 = combined
                        patience = 0
                        save_checkpoint(
                            model, best_model_path, epoch, total_val_steps, val,
                            model_name, freeze_layers, phase=2, skip_frozen=True,
                            selection_criterion="best_combined_phase2",
                        )
                    else:
                        patience += 1
                        if patience >= EARLY_STOP_PATIENCE_VALS:
                            print(f"\n  Early stopping at val {total_val_steps} "
                                  f"(best phase-2 combined={best_combined_phase2:.4f}).")
                            early_stop = True
                            break

                model.train()

        if early_stop: break
        elapsed = time.time() - t0
        print(f"\n[Epoch {epoch}] {elapsed:.1f}s")

    print(f"\nTraining complete.")
    print(f"  Phase 1 best r_skip:  {best_r_skip_phase1:.4f}")
    print(f"  Phase 1 best combined: {best_combined_phase1:.4f}")
    print(f"  Phase 2 best combined: {best_combined_phase2:.4f}")

    # Decide which checkpoint is the final model
    if best_combined_phase2 > 0:
        # Phase 2 ran — use it.
        final_path = best_model_path
        print(f"  Final model: {final_path} (Phase 2 best combined)")
    elif best_combined_phase1 > 0:
        # Phase 2 didn't run (e.g., num_epochs == phase1_epochs)
        # Use phase 1's best combined as the final model
        import shutil
        shutil.copy(best_combined_phase1_path, best_model_path)
        final_path = best_model_path
        print(f"  Final model: {final_path} (Phase 1 best combined; Phase 2 didn't run)")
    else:
        print("  No best checkpoint saved; cannot run test eval.")
        return

    if test_agg:
        ckpt = torch.load(final_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\nLoaded final checkpoint (epoch {ckpt['epoch']}, val {ckpt['val_step']}, "
              f"phase {ckpt.get('phase', '?')}, skip_frozen={ckpt.get('skip_residual_frozen', '?')})")

        test = evaluate_detailed(model, test_agg, device, subtlex)
        print(f"\nTest set results:")
        print(f"  r_TRT={test['r_trt']:.3f}  r_FFD={test['r_ffd']:.3f}  "
              f"r_Gaze={test['r_gaze']:.3f}  r_skip={test['r_skip']:.3f}")
        print(f"  combined = {combined_metric(test):.4f}")
        print(f"  mean_skip = {test['mean_skip']:.3f}  std_skip = {test['std_skip']:.3f}")
        print(f"  residual_abs = {test['mean_residual_logit_abs']:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--freeze", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lm_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--cog_lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase1_epochs", type=int, default=1,
                        help="Number of epochs in Phase 1 before freezing skip_residual.")
    args = parser.parse_args()

    if args.freeze is not None:
        freeze_layers = args.freeze
    else:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        freeze_layers = int(cfg.num_hidden_layers * 0.75)
        print(f"Auto-freeze: {freeze_layers}/{cfg.num_hidden_layers} layers")

    if args.phase1_epochs >= args.epochs:
        print(f"WARNING: phase1_epochs ({args.phase1_epochs}) >= total epochs ({args.epochs}). "
              f"Phase 2 will not run.")

    model_short = args.model.replace("/", "_")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "checkpoints",
        "hybrid_v4c_v2_wide_prior_skipfreeze",
        f"geco_{model_short}_seed{args.seed}_p1ep{args.phase1_epochs}",
    )
    log_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "logs",
        f"train_hybrid_v4c_v2_wide_prior_skipfreeze_geco_seed{args.seed}_p1ep{args.phase1_epochs}.log",
    )

    train(
        data_dir=data_dir, num_epochs=args.epochs,
        lm_lr=args.lm_lr, head_lr=args.head_lr, cog_lr=args.cog_lr,
        batch_size=args.batch_size, accumulation_steps=args.accum,
        save_dir=save_dir, log_path=log_path,
        seed=args.seed, model_name=args.model, freeze_layers=freeze_layers,
        phase1_epochs=args.phase1_epochs,
    )
