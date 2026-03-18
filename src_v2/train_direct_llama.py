"""
Training script for the Direct Regression LLaMA baseline on GECO corpus.

This model has NO E-Z Reader architecture — it directly predicts FFD, Gaze,
TRT, and Skip from LLaMA word representations. This serves as a baseline
to test whether the cognitive architecture adds value.

Usage:
  python3 -u src_v2/train_direct_llama.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import os
import sys
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'archive', 'original_ezreader'))

from model_llama_direct import DirectRegressionLLaMA
from data_loader import aggregate_by_sentence
from geco_loader import load_geco, split_geco


# --------------------------------------------------------------------------- #
#  Regularization hyperparameters
# --------------------------------------------------------------------------- #

LAMBDA_PRIOR = 10.0       # skip prior weight
SKIP_TARGET = 0.45        # target mean skip rate
LAMBDA_TRT_SCALE = 0.001  # TRT scale matching penalty weight


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
#  Loss function
# --------------------------------------------------------------------------- #

def compute_loss(pred, human_trt, human_ffd, human_gaze, human_skip):
    pred_trt = pred['total_reading_time'].float()
    pred_ffd = pred['first_fixation'].float()
    pred_gaze = pred['gaze_duration'].float()
    pred_skip = pred['skip_prob'].float()

    trt_loss = nn.functional.mse_loss(pred_trt, human_trt)
    ffd_loss = nn.functional.mse_loss(pred_ffd, human_ffd)
    gaze_loss = nn.functional.mse_loss(pred_gaze, human_gaze)

    skip_pred = pred_skip.clamp(1e-6, 1 - 1e-6)
    skip_loss = nn.functional.binary_cross_entropy(skip_pred, human_skip)

    # Regularizers
    mean_skip = pred_skip.mean()
    skip_prior = LAMBDA_PRIOR * (mean_skip - SKIP_TARGET) ** 2
    trt_scale = LAMBDA_TRT_SCALE * (pred_trt.mean() - human_trt.mean()) ** 2

    total = (0.25 * trt_loss + 0.25 * ffd_loss + 0.25 * gaze_loss + 0.4 * skip_loss
             + skip_prior + trt_scale)

    return total, {
        'trt': trt_loss.item(),
        'ffd': ffd_loss.item(),
        'gaze': gaze_loss.item(),
        'skip': skip_loss.item(),
        'skip_prior': skip_prior.item(),
        'trt_scale': trt_scale.item(),
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
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for i in range(0, len(agg_data), batch_size):
            batch = agg_data[i:i + batch_size]
            word_lists, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip = collate_aggregated(batch, device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(word_lists, pred_vals, wlens)
            loss, _ = compute_loss(pred, h_trt, h_ffd, h_gaze, h_skip)
            total_loss += loss.item() * len(batch)
            n += len(batch)

            for b in range(len(batch)):
                seq_len = len(batch[b].tokens)
                all_pred_trt.extend(pred['total_reading_time'][b, :seq_len].cpu().tolist())
                all_human_trt.extend(batch[b].mean_trt)
                all_pred_ffd.extend(pred['first_fixation'][b, :seq_len].cpu().tolist())
                all_human_ffd.extend(batch[b].mean_ffd)
                all_pred_gaze.extend(pred['gaze_duration'][b, :seq_len].cpu().tolist())
                all_human_gaze.extend(batch[b].mean_gaze)
                all_pred_skip.extend(pred['skip_prob'][b, :seq_len].cpu().tolist())
                all_human_skip.extend(batch[b].skip_rate)

    avg_loss = total_loss / max(n, 1)

    def corr(a, b):
        a, b = np.array(a), np.array(b)
        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            return np.corrcoef(a, b)[0, 1]
        return 0.0

    return {
        'loss': avg_loss,
        'r_trt': corr(all_pred_trt, all_human_trt),
        'r_ffd': corr(all_pred_ffd, all_human_ffd),
        'r_gaze': corr(all_pred_gaze, all_human_gaze),
        'r_skip': corr(all_pred_skip, all_human_skip),
        'mae_trt': np.mean(np.abs(np.array(all_pred_trt) - np.array(all_human_trt))),
        'mae_ffd': np.mean(np.abs(np.array(all_pred_ffd) - np.array(all_human_ffd))),
        'mean_pred_trt': np.mean(all_pred_trt),
        'mean_pred_ffd': np.mean(all_pred_ffd),
        'mean_pred_gaze': np.mean(all_pred_gaze),
        'mean_human_trt': np.mean(all_human_trt),
        'mean_human_ffd': np.mean(all_human_ffd),
        'mean_human_gaze': np.mean(all_human_gaze),
    }


# --------------------------------------------------------------------------- #
#  Print helpers
# --------------------------------------------------------------------------- #

def print_sample_predictions(model, agg_data, device, n_sentences=3, n_words=8):
    model.eval()
    with torch.no_grad():
        for s_idx in range(min(n_sentences, len(agg_data))):
            s = agg_data[s_idx]
            word_list = [s.tokens]
            pv = torch.tensor(
                s.predictabilities, dtype=torch.float32
            ).unsqueeze(0).to(device)
            wl = torch.tensor(
                [len(t) for t in s.tokens], dtype=torch.float32
            ).unsqueeze(0).to(device)
            p = model(word_list, pv, wl)

            title = ' '.join(s.tokens[:6]) + ('...' if len(s.tokens) > 6 else '')
            print(f"  Sentence {s_idx+1}: \"{title}\"")
            print(f"  {'word':<14s} | {'pTRT':>5s} {'hTRT':>5s} {'err':>5s} | "
                  f"{'pFFD':>5s} {'hFFD':>5s} | {'pGaze':>5s} {'hGaze':>5s} | "
                  f"{'pSkip':>5s} {'hSkip':>5s}")
            print(f"  {'-'*90}")

            for i in range(min(n_words, len(s.tokens))):
                pt = p['total_reading_time'][0, i].item()
                pf = p['first_fixation'][0, i].item()
                pg = p['gaze_duration'][0, i].item()
                ps = p['skip_prob'][0, i].item()
                ht = s.mean_trt[i]
                hf = s.mean_ffd[i]
                hg = s.mean_gaze[i]
                hs = s.skip_rate[i]
                err = pt - ht
                print(
                    f"  {s.tokens[i]:<14s} | "
                    f"{pt:5.0f} {ht:5.0f} {err:+5.0f} | "
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
    batch_size=8,
    accumulation_steps=8,
    save_dir="../checkpoints_v2/geco_direct_llama",
    seed=42,
    model_name="meta-llama/Llama-3.2-1B",
    freeze_layers=12,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load GECO data ----
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
    print(f"  Aggregated: {len(train_agg)} train | {len(val_agg)} val | {len(test_agg)} test sentences")

    # ---- Model ----
    print(f"\nLoading model: {model_name}")
    print(f"  Freezing first {freeze_layers} layers")
    model = DirectRegressionLLaMA(
        model_name=model_name,
        freeze_layers=freeze_layers,
        hidden_dim=256,
    ).to(device)

    # ---- Differential learning rates (2 groups: LM + heads) ----
    lm_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("llama."):
            lm_params.append(param)
        else:
            head_params.append(param)

    n_lm_trainable = sum(p.numel() for p in lm_params)
    n_head_trainable = sum(p.numel() for p in head_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Total parameters:    {total_params:,}")
    print(f"  Frozen (LM):         {n_frozen:,}")
    print(f"  Trainable (LM):      {n_lm_trainable:,} (lr={lm_lr})")
    print(f"  Trainable (heads):   {n_head_trainable:,} (lr={head_lr})")

    optimizer = optim.AdamW([
        {"params": lm_params, "lr": lm_lr, "weight_decay": 0.01},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.0},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-7
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(save_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_dir, "training_log.txt"))

    best_val_corr = -1.0

    print("\n" + "=" * 90)
    print(f"Training DIRECT REGRESSION (LLaMA, NO E-Z Reader) on GECO Corpus")
    print(f"  Model: {model_name}")
    print(f"  Batch size: {batch_size} | Gradient accumulation steps: {accumulation_steps}")
    print(f"  Effective batch size: {batch_size * accumulation_steps}")
    print(f"  Mixed precision (AMP): {use_amp}")
    print(f"  Loss weights: 0.25*TRT + 0.25*FFD + 0.25*Gaze + 0.4*Skip + regularizers")
    print("=" * 90)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()

        epoch_data = train_raw.copy()
        random.shuffle(epoch_data)

        epoch_loss = 0.0
        epoch_trt = 0.0
        epoch_ffd = 0.0
        epoch_gaze = 0.0
        epoch_skip = 0.0
        n_samples = 0

        optimizer.zero_grad()

        n_batches = (len(epoch_data) + batch_size - 1) // batch_size
        for step in range(n_batches):
            batch = epoch_data[step * batch_size : (step + 1) * batch_size]
            word_lists, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip = collate_sentences(batch, device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(word_lists, pred_vals, wlens)
            loss, parts = compute_loss(pred, h_trt, h_ffd, h_gaze, h_skip)

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
            n_samples += len(batch)

        epoch_loss /= n_samples
        epoch_trt /= n_samples
        epoch_ffd /= n_samples
        epoch_gaze /= n_samples
        epoch_skip /= n_samples
        elapsed = time.time() - t0

        # ---- Validation ----
        val_metrics = evaluate_detailed(model, val_agg, device)
        scheduler.step(val_metrics['loss'])
        lm_lr_now = optimizer.param_groups[0]['lr']
        head_lr_now = optimizer.param_groups[1]['lr']

        is_best = val_metrics['r_trt'] > best_val_corr

        print(f"\n[Epoch {epoch:3d}/{num_epochs}] {elapsed:.1f}s | "
              f"lm_lr={lm_lr_now:.2e} head_lr={head_lr_now:.2e}")
        print(f"  Train: loss={epoch_loss:.1f} "
              f"(trt={epoch_trt:.0f} ffd={epoch_ffd:.0f} gaze={epoch_gaze:.0f} skip={epoch_skip:.3f}) "
              f"| {n_samples:,} samples")
        print(f"  Val:   loss={val_metrics['loss']:.1f} | "
              f"r_TRT={val_metrics['r_trt']:.3f}  "
              f"r_FFD={val_metrics['r_ffd']:.3f}  "
              f"r_Gaze={val_metrics['r_gaze']:.3f}  "
              f"r_skip={val_metrics['r_skip']:.3f}")
        print(f"  Val:   MAE_TRT={val_metrics['mae_trt']:.1f}ms  "
              f"MAE_FFD={val_metrics['mae_ffd']:.1f}ms")
        print(f"  Pred:  mean_TRT={val_metrics['mean_pred_trt']:.0f}ms "
              f"(human={val_metrics['mean_human_trt']:.0f}ms) | "
              f"mean_FFD={val_metrics['mean_pred_ffd']:.0f}ms "
              f"(human={val_metrics['mean_human_ffd']:.0f}ms) | "
              f"mean_Gaze={val_metrics['mean_pred_gaze']:.0f}ms "
              f"(human={val_metrics['mean_human_gaze']:.0f}ms)")
        print(f"  Scales: trt_scale={model.trt_scale.item():.1f} "
              f"ffd_scale={model.ffd_scale.item():.1f} "
              f"gaze_scale={model.gaze_scale.item():.1f}")

        print_sample_predictions(model, train_agg, device, n_sentences=2, n_words=8)

        if is_best:
            print(f"  ** NEW BEST (r_TRT={val_metrics['r_trt']:.3f}) **")
            best_val_corr = val_metrics['r_trt']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'model_name': model_name,
                'freeze_layers': freeze_layers,
                'val_metrics': val_metrics,
            }, os.path.join(save_dir, "best_model.pt"))

    # ---- Final summary ----
    print("\n" + "=" * 90)
    print(f"Training complete!")
    print(f"Best validation r_TRT = {best_val_corr:.3f}")
    print("=" * 90)

    # ---- Test set ----
    test_metrics = evaluate_detailed(model, test_agg, device)
    print(f"\nTest set results:")
    print(f"  r_TRT={test_metrics['r_trt']:.3f}  "
          f"r_FFD={test_metrics['r_ffd']:.3f}  "
          f"r_Gaze={test_metrics['r_gaze']:.3f}  "
          f"r_skip={test_metrics['r_skip']:.3f}")
    print(f"  MAE_TRT={test_metrics['mae_trt']:.1f}ms  "
          f"MAE_FFD={test_metrics['mae_ffd']:.1f}ms")
    print(f"  mean_TRT={test_metrics['mean_pred_trt']:.0f}ms "
          f"(human={test_metrics['mean_human_trt']:.0f}ms)")

    print("\nSample test predictions:")
    print_sample_predictions(model, test_agg, device, n_sentences=3, n_words=10)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B",
                        help="HuggingFace causal LM name")
    parser.add_argument("--freeze", type=int, default=None,
                        help="Number of layers to freeze (default: 3/4 of total layers)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=8)
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

    model_short = args.model.replace("/", "_")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    save_dir = os.path.join(os.path.dirname(__file__), "..", f"checkpoints_v2/geco_direct_{model_short}")

    train(
        data_dir=data_dir,
        num_epochs=args.epochs,
        lm_lr=args.lm_lr,
        head_lr=args.head_lr,
        batch_size=args.batch_size,
        accumulation_steps=args.accum,
        save_dir=save_dir,
        model_name=args.model,
        freeze_layers=freeze_layers,
    )
