"""
Per-participant training: train model_llama_faithful_sh on each GECO
participant individually and report results.

Diagnostic experiment: if single-participant r_TRT >> 0.46, the
inter-participant mixing is the bottleneck. If ~0.46 or lower,
the limitation is intrinsic to the task/features.

For each of 14 participants:
  1. Filter GECO to that participant's data
  2. Split into train/val/test by text_id (same split as main experiments)
  3. Train for 4 epochs
  4. Evaluate on that participant's val + test data
  5. Reset model and repeat for next participant

Usage:
  python3 -u src_v2/lm_train/train_faithful_sh_per_participant.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import os
import sys
import time
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lm_model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'archive', 'original_ezreader'))

from model_llama_faithful_sh import NeuralEZReaderLLaMA
from geco_loader import load_geco, split_geco


# --------------------------------------------------------------------------- #
#  Hyperparameters
# --------------------------------------------------------------------------- #

LAMBDA_L1 = 0.01
LAMBDA_L1_LOWER = 0.05
LAMBDA_DELTA = 5.0
LAMBDA_PRIOR = 10.0
L1_MAX = 400.0
L1_MIN = 60.0
SKIP_TARGET = 0.45
NUM_EPOCHS = 4


# --------------------------------------------------------------------------- #
#  Collate
# --------------------------------------------------------------------------- #

def collate_sentences(batch, device):
    word_lists = [sd.tokens for sd in batch]
    pred_vals = pad_sequence(
        [torch.tensor([w.predictability for w in sd.words], dtype=torch.float32) for sd in batch],
        batch_first=True,
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
    return word_lists, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip


# --------------------------------------------------------------------------- #
#  Loss
# --------------------------------------------------------------------------- #

def compute_loss(pred, human_trt, human_ffd, human_gaze, human_skip, delta):
    pred_trt = pred['conditional_trt'].float()
    pred_ffd = pred['first_fixation'].float()
    pred_gaze = pred['gaze_duration'].float()
    pred_skip = pred['skip_prob'].float()
    pred_l1 = pred['L1'].float()

    fixated = (human_skip < 0.5)

    if fixated.sum() > 0:
        trt_loss = F.mse_loss(pred_trt[fixated], human_trt[fixated])
        ffd_loss = F.mse_loss(pred_ffd[fixated], human_ffd[fixated])
        gaze_loss = F.mse_loss(pred_gaze[fixated], human_gaze[fixated])
    else:
        trt_loss = torch.tensor(0.0, device=pred_trt.device)
        ffd_loss = torch.tensor(0.0, device=pred_trt.device)
        gaze_loss = torch.tensor(0.0, device=pred_trt.device)

    skip_pred = pred_skip.clamp(1e-6, 1 - 1e-6)
    skip_loss = F.binary_cross_entropy(skip_pred, human_skip)

    l1_excess = F.relu(pred_l1 - L1_MAX)
    l1_reg = LAMBDA_L1 * l1_excess.mean()
    l1_deficit = F.relu(L1_MIN - pred_l1)
    l1_lower_reg = LAMBDA_L1_LOWER * l1_deficit.mean()

    delta_low = F.relu(0.20 - delta)
    delta_high = F.relu(delta - 0.50)
    delta_reg = LAMBDA_DELTA * (delta_low ** 2 + delta_high ** 2)

    mean_skip = pred_skip.mean()
    skip_prior = LAMBDA_PRIOR * (mean_skip - SKIP_TARGET) ** 2

    total = (0.25 * trt_loss + 0.25 * ffd_loss + 0.25 * gaze_loss + 0.4 * skip_loss
             + skip_prior + l1_reg + l1_lower_reg + delta_reg)

    return total, {
        'trt': trt_loss.item(),
        'ffd': ffd_loss.item(),
        'gaze': gaze_loss.item(),
        'skip': skip_loss.item(),
        'total': total.item(),
    }


# --------------------------------------------------------------------------- #
#  Evaluation (on single participant's raw data)
# --------------------------------------------------------------------------- #

def evaluate_participant(model, data, device, batch_size=8):
    model.eval()
    all_pred_trt, all_human_trt = [], []
    all_pred_ffd, all_human_ffd = [], []
    all_pred_gaze, all_human_gaze = [], []
    all_pred_skip, all_human_skip = [], []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            word_lists, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip = collate_sentences(batch, device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(word_lists, pred_vals, wlens)

            for b in range(len(batch)):
                seq_len = len(batch[b].tokens)
                for w_idx in range(seq_len):
                    w = batch[b].words[w_idx]
                    p_skip = pred['skip_prob'][b, w_idx].cpu().item()
                    h_sk = 1.0 if w.was_skipped else 0.0

                    all_pred_skip.append(p_skip)
                    all_human_skip.append(h_sk)

                    # Only include fixated words for time metrics
                    if not w.was_skipped:
                        p_trt = pred['conditional_trt'][b, w_idx].cpu().item()
                        p_ffd = pred['first_fixation'][b, w_idx].cpu().item()
                        p_gaze = pred['gaze_duration'][b, w_idx].cpu().item()

                        if w.total_reading_time > 0:
                            all_pred_trt.append(p_trt)
                            all_human_trt.append(w.total_reading_time)
                        if w.first_fixation_duration > 0:
                            all_pred_ffd.append(p_ffd)
                            all_human_ffd.append(w.first_fixation_duration)
                        if w.gaze_duration > 0:
                            all_pred_gaze.append(p_gaze)
                            all_human_gaze.append(w.gaze_duration)

    def corr(a, b):
        a, b = np.array(a), np.array(b)
        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            return np.corrcoef(a, b)[0, 1]
        return 0.0

    return {
        'r_trt': corr(all_pred_trt, all_human_trt),
        'r_ffd': corr(all_pred_ffd, all_human_ffd),
        'r_gaze': corr(all_pred_gaze, all_human_gaze),
        'r_skip': corr(all_pred_skip, all_human_skip),
        'mae_trt': np.mean(np.abs(np.array(all_pred_trt) - np.array(all_human_trt))) if all_pred_trt else 0,
        'mae_ffd': np.mean(np.abs(np.array(all_pred_ffd) - np.array(all_human_ffd))) if all_pred_ffd else 0,
        'n_fixated': len(all_pred_trt),
        'n_total': len(all_pred_skip),
        'skip_rate': np.mean(all_human_skip),
    }


# --------------------------------------------------------------------------- #
#  Train one participant
# --------------------------------------------------------------------------- #

def train_one_participant(model, train_data, val_data, test_data, participant_id,
                          device, batch_size=8, accumulation_steps=4,
                          lm_lr=2e-5, head_lr=5e-4):
    """Train model on one participant's data for NUM_EPOCHS epochs."""

    # Optimizer
    lm_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("llama."):
            lm_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW([
        {"params": lm_params, "lr": lm_lr, "weight_decay": 0.01},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.0},
    ])

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n  Training on participant {participant_id}: "
          f"{len(train_data)} train / {len(val_data)} val / {len(test_data)} test sentences")

    best_val_r = -1.0
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()

        epoch_data = train_data.copy()
        random.shuffle(epoch_data)

        epoch_loss = 0.0
        n_samples = 0
        optimizer.zero_grad()

        n_batches = (len(epoch_data) + batch_size - 1) // batch_size
        for step in range(n_batches):
            batch = epoch_data[step * batch_size : (step + 1) * batch_size]
            word_lists, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip = collate_sentences(batch, device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(word_lists, pred_vals, wlens)
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
            n_samples += len(batch)

        elapsed = time.time() - t0

        # Evaluate
        val_metrics = evaluate_participant(model, val_data, device)

        is_best = val_metrics['r_trt'] > best_val_r
        if is_best:
            best_val_r = val_metrics['r_trt']
            best_epoch = epoch

        print(f"    Epoch {epoch}/{NUM_EPOCHS} ({elapsed:.0f}s) | "
              f"r_TRT={val_metrics['r_trt']:.3f}  r_FFD={val_metrics['r_ffd']:.3f}  "
              f"r_Gaze={val_metrics['r_gaze']:.3f}  r_skip={val_metrics['r_skip']:.3f}  "
              f"MAE_TRT={val_metrics['mae_trt']:.0f}ms"
              f"{'  *BEST*' if is_best else ''}")

    # Final test evaluation
    test_metrics = evaluate_participant(model, test_data, device)
    print(f"    TEST: r_TRT={test_metrics['r_trt']:.3f}  r_FFD={test_metrics['r_ffd']:.3f}  "
          f"r_Gaze={test_metrics['r_gaze']:.3f}  r_skip={test_metrics['r_skip']:.3f}  "
          f"MAE_TRT={test_metrics['mae_trt']:.0f}ms  "
          f"skip_rate={test_metrics['skip_rate']:.2f}  "
          f"delta={model.delta.item():.3f}")

    return {
        'participant_id': participant_id,
        'best_val_r_trt': best_val_r,
        'best_epoch': best_epoch,
        'val': val_metrics,
        'test': test_metrics,
        'delta': model.delta.item(),
        'l1_scale': model.l1_scale.item(),
        'n_train': len(train_data),
    }


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main(model_name="meta-llama/Llama-3.2-1B", freeze_layers=12,
         batch_size=8, accumulation_steps=4, lm_lr=2e-5, head_lr=5e-4,
         seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load GECO data ----
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
    material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
    pred_path = os.path.join(data_dir, "geco_predictability.pkl")

    print("Loading GECO Corpus...")
    raw_dataset = load_geco(reading_path, material_path, pred_path)
    print(f"  Total observations: {len(raw_dataset):,}")

    # Get train/val/test text_id split (same as all other experiments)
    train_raw, val_raw, test_raw = split_geco(raw_dataset)
    train_text_ids = set(sd.text_id for sd in train_raw)
    val_text_ids = set(sd.text_id for sd in val_raw)
    test_text_ids = set(sd.text_id for sd in test_raw)

    # Get all participants
    participants = sorted(set(sd.participant_id for sd in raw_dataset))
    print(f"  Participants: {len(participants)} — {participants}")

    # ---- Load model once, save initial state ----
    print(f"\nLoading model: {model_name} (freeze={freeze_layers})")
    model = NeuralEZReaderLLaMA(
        model_name=model_name,
        freeze_layers=freeze_layers,
        hidden_dim=256,
    ).to(device)

    # Save initial state for resetting between participants
    init_state = copy.deepcopy(model.state_dict())

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable:,}")

    # ---- Run per-participant experiments ----
    all_results = []

    print("\n" + "=" * 90)
    print(f"PER-PARTICIPANT TRAINING ({NUM_EPOCHS} epochs each)")
    print(f"  Model: {model_name}")
    print(f"  FFD = L1 | Gaze = L1 + L2 | TRT = Gaze + regression | Skip = learned head")
    print("=" * 90)

    for p_idx, pid in enumerate(participants):
        # Filter data for this participant
        p_data = [sd for sd in raw_dataset if sd.participant_id == pid]
        p_train = [sd for sd in p_data if sd.text_id in train_text_ids]
        p_val = [sd for sd in p_data if sd.text_id in val_text_ids]
        p_test = [sd for sd in p_data if sd.text_id in test_text_ids]

        if len(p_train) < 50 or len(p_val) < 10:
            print(f"\n  [{p_idx+1}/{len(participants)}] Participant {pid}: "
                  f"too few data ({len(p_train)} train), skipping")
            continue

        print(f"\n  [{p_idx+1}/{len(participants)}] Participant {pid}")

        # Reset model to initial state
        model.load_state_dict(init_state)

        # Train and evaluate
        result = train_one_participant(
            model, p_train, p_val, p_test, pid,
            device, batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            lm_lr=lm_lr, head_lr=head_lr,
        )
        all_results.append(result)

    # ---- Summary ----
    print("\n" + "=" * 90)
    print("SUMMARY: Per-participant results (test set)")
    print("=" * 90)
    print(f"{'PID':<8s} {'n_train':>7s} | {'r_TRT':>6s} {'r_FFD':>6s} {'r_Gaze':>6s} {'r_skip':>6s} | "
          f"{'MAE_TRT':>7s} {'skip%':>6s} {'delta':>6s} {'best_ep':>7s}")
    print("-" * 90)

    for r in all_results:
        t = r['test']
        print(f"{r['participant_id']:<8s} {r['n_train']:>7d} | "
              f"{t['r_trt']:>6.3f} {t['r_ffd']:>6.3f} {t['r_gaze']:>6.3f} {t['r_skip']:>6.3f} | "
              f"{t['mae_trt']:>6.0f}ms {t['skip_rate']:>5.0%} {r['delta']:>6.3f} {r['best_epoch']:>7d}")

    # Averages
    if all_results:
        mean_r_trt = np.mean([r['test']['r_trt'] for r in all_results])
        mean_r_ffd = np.mean([r['test']['r_ffd'] for r in all_results])
        mean_r_gaze = np.mean([r['test']['r_gaze'] for r in all_results])
        mean_r_skip = np.mean([r['test']['r_skip'] for r in all_results])
        mean_mae_trt = np.mean([r['test']['mae_trt'] for r in all_results])
        std_r_trt = np.std([r['test']['r_trt'] for r in all_results])

        print("-" * 90)
        print(f"{'MEAN':<8s} {'':>7s} | "
              f"{mean_r_trt:>6.3f} {mean_r_ffd:>6.3f} {mean_r_gaze:>6.3f} {mean_r_skip:>6.3f} | "
              f"{mean_mae_trt:>6.0f}ms")
        print(f"{'STD':<8s} {'':>7s} | "
              f"{std_r_trt:>6.3f}")

        print(f"\n  For reference (trained on all participants, evaluated on aggregated):")
        print(f"    r_TRT = 0.454-0.466")
        print(f"\n  Interpretation:")
        print(f"    mean per-participant r_TRT = {mean_r_trt:.3f}")
        if mean_r_trt > 0.50:
            print(f"    >> HIGHER than population model → inter-participant mixing is the bottleneck")
        elif mean_r_trt > 0.43:
            print(f"    ~~ SIMILAR to population model → single-trial noise is the main limit")
        else:
            print(f"    << LOWER than population model → too little data per participant")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--freeze", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lm_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    args = parser.parse_args()

    if args.freeze is not None:
        freeze_layers = args.freeze
    else:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        n_layers = cfg.num_hidden_layers
        freeze_layers = int(n_layers * 0.75)
        print(f"Auto-freeze: {freeze_layers}/{n_layers} layers")

    main(
        model_name=args.model,
        freeze_layers=freeze_layers,
        batch_size=args.batch_size,
        accumulation_steps=args.accum,
        lm_lr=args.lm_lr,
        head_lr=args.head_lr,
    )
