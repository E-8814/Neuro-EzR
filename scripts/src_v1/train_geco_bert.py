"""
Supervised training for the BERT-based Differentiable Neural EZ Reader on GECO corpus.
GPU-optimized variant with mini-batching and automatic mixed precision (AMP).

Uses the same architecture as train_geco_bert.py but processes multiple sentences
per forward pass and uses FP16 on CUDA for higher GPU utilization.
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ez_reader'))

from model_bert import NeuralEZReaderBERT
from data_loader import aggregate_by_sentence, split_aggregated
from geco_loader import load_geco, split_geco


# --------------------------------------------------------------------------- #
#  Collate batch of SentenceData / AggregatedSentence into padded tensors
# --------------------------------------------------------------------------- #

def collate_sentences(batch, device):
    """Pad a list of SentenceData into batched tensors on *device*."""
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
    h_skip = pad_sequence(
        [torch.tensor([1.0 if s else 0.0 for s in sd.skip_flags], dtype=torch.float32) for sd in batch],
        batch_first=True,
    ).to(device)
    return word_lists, pred_vals, wlens, h_trt, h_ffd, h_skip


def collate_aggregated(batch, device):
    """Pad a list of AggregatedSentence into batched tensors on *device*."""
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
    h_skip = pad_sequence(
        [torch.tensor(a.skip_rate, dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    return word_lists, pred_vals, wlens, h_trt, h_ffd, h_skip


# --------------------------------------------------------------------------- #
#  Logger (dual stdout + file)
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
#  Loss function (3-component, same as train_bert.py)
# --------------------------------------------------------------------------- #

def compute_loss(pred, human_trt, human_ffd, human_skip):
    """
    Combined loss:
      - MSE on total reading time
      - MSE on first fixation duration
      - BCE on skip probability
    """
    trt_loss = nn.functional.mse_loss(pred['total_reading_time'], human_trt)
    ffd_loss = nn.functional.mse_loss(pred['first_fixation'], human_ffd)

    skip_pred = pred['skip_prob'].clamp(1e-6, 1 - 1e-6)
    skip_loss = nn.functional.binary_cross_entropy(skip_pred, human_skip)

    total = 0.5 * trt_loss + 0.3 * ffd_loss + 0.2 * skip_loss

    return total, {
        'trt': trt_loss.item(),
        'ffd': ffd_loss.item(),
        'skip': skip_loss.item(),
        'total': total.item(),
    }


# --------------------------------------------------------------------------- #
#  Evaluation (on aggregated data for clean metrics)
# --------------------------------------------------------------------------- #

def evaluate_detailed(model, agg_data, device, batch_size=16):
    """Evaluate on aggregated data. Returns loss, correlations, error metrics."""
    model.eval()
    all_pred_trt, all_human_trt = [], []
    all_pred_ffd, all_human_ffd = [], []
    all_pred_skip, all_human_skip = [], []
    all_pred_l1, all_pred_l2 = [], []
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for i in range(0, len(agg_data), batch_size):
            batch = agg_data[i:i + batch_size]
            word_lists, pred_vals, wlens, h_trt, h_ffd, h_skip = collate_aggregated(batch, device)

            pred = model(word_lists, pred_vals, wlens)
            loss, _ = compute_loss(pred, h_trt, h_ffd, h_skip)
            total_loss += loss.item() * len(batch)
            n += len(batch)

            for b in range(len(batch)):
                seq_len = len(batch[b].tokens)
                all_pred_trt.extend(pred['total_reading_time'][b, :seq_len].cpu().tolist())
                all_human_trt.extend(batch[b].mean_trt)
                all_pred_ffd.extend(pred['first_fixation'][b, :seq_len].cpu().tolist())
                all_human_ffd.extend(batch[b].mean_ffd)
                all_pred_skip.extend(pred['skip_prob'][b, :seq_len].cpu().tolist())
                all_human_skip.extend(batch[b].skip_rate)
                all_pred_l1.extend(pred['L1'][b, :seq_len].cpu().tolist())
                all_pred_l2.extend(pred['L2'][b, :seq_len].cpu().tolist())

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
        'r_skip': corr(all_pred_skip, all_human_skip),
        'mae_trt': np.mean(np.abs(np.array(all_pred_trt) - np.array(all_human_trt))),
        'mae_ffd': np.mean(np.abs(np.array(all_pred_ffd) - np.array(all_human_ffd))),
        'mean_pred_trt': np.mean(all_pred_trt),
        'mean_human_trt': np.mean(all_human_trt),
        'mean_l1': np.mean(all_pred_l1),
        'std_l1': np.std(all_pred_l1),
        'mean_l2': np.mean(all_pred_l2),
        'std_l2': np.std(all_pred_l2),
    }


# --------------------------------------------------------------------------- #
#  Print sample predictions
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
            print(f"  {'word':<14s} {'L1':>5s} {'L2':>5s} | {'pTRT':>5s} {'hTRT':>5s} {'err':>5s} | "
                  f"{'pFFD':>5s} {'hFFD':>5s} | {'pSkip':>5s} {'hSkip':>5s}")
            print(f"  {'-'*80}")

            for i in range(min(n_words, len(s.tokens))):
                l1 = p['L1'][0, i].item()
                l2 = p['L2'][0, i].item()
                pt = p['total_reading_time'][0, i].item()
                pf = p['first_fixation'][0, i].item()
                ps = p['skip_prob'][0, i].item()
                ht = s.mean_trt[i]
                hf = s.mean_ffd[i]
                hs = s.skip_rate[i]
                err = pt - ht
                print(
                    f"  {s.tokens[i]:<14s} {l1:5.0f} {l2:5.0f} | "
                    f"{pt:5.0f} {ht:5.0f} {err:+5.0f} | "
                    f"{pf:5.0f} {hf:5.0f} | "
                    f"{ps:5.2f} {hs:5.2f}"
                )
            print()


def print_ezreader_params(model):
    ezr = model.ezreader
    print("  Learned EZ Reader parameters:")
    print(f"    saccade_time     = {ezr.saccade_time.item():.1f}ms (init=150)")
    print(f"    attention_shift  = {ezr.attention_shift.item():.1f}ms (init=25)")
    print(f"    skip_sharpness   = {ezr.skip_sharpness.item():.2f} (init=8)")
    print(f"    eccentricity     = {ezr.eccentricity.item():.4f} (init=0.1)")
    print(f"    integration_cost = {ezr.integration_cost.item():.4f} (init=0.08)")
    print(f"    l1_scale         = {model.l1_scale.item():.1f} (init=50)")
    print(f"    l2_scale         = {model.l2_scale.item():.1f} (init=30)")


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #

def train(
    data_dir="../data",
    num_epochs=50,
    bert_lr=2e-5,
    head_lr=5e-4,
    batch_size=16,
    accumulation_steps=4,
    save_dir="../checkpoints_v1/geco_bert",
    seed=42,
    bert_model_name="bert-base-uncased",
    freeze_bert_layers=8,
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

    # Split raw data by text_id
    train_raw, val_raw, test_raw = split_geco(raw_dataset)
    print(f"  Train: {len(train_raw):,} | Val: {len(val_raw):,} | Test: {len(test_raw):,}")

    # Aggregate for clean evaluation metrics (GECO has 14 participants)
    aggregated = aggregate_by_sentence(raw_dataset, min_participants=5)
    print(f"  Aggregated sentences (min 5 participants): {len(aggregated)}")

    # Split aggregated by text_id
    train_text_ids = set(sd.text_id for sd in train_raw)
    val_text_ids = set(sd.text_id for sd in val_raw)
    train_agg = [a for a in aggregated if a.text_id in train_text_ids]
    val_agg = [a for a in aggregated if a.text_id in val_text_ids]
    test_agg = [a for a in aggregated if a.text_id not in train_text_ids and a.text_id not in val_text_ids]
    print(f"  Aggregated: {len(train_agg)} train | {len(val_agg)} val | {len(test_agg)} test sentences")

    # ---- Model ----
    print(f"\nLoading BERT model: {bert_model_name}")
    print(f"  Freezing first {freeze_bert_layers} BERT layers")
    model = NeuralEZReaderBERT(
        bert_model_name=bert_model_name,
        freeze_bert_layers=freeze_bert_layers,
    ).to(device)

    # ---- Differential learning rates ----
    bert_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("bert."):
            bert_params.append(param)
        else:
            head_params.append(param)

    n_bert_trainable = sum(p.numel() for p in bert_params)
    n_head_trainable = sum(p.numel() for p in head_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Total parameters:    {total_params:,}")
    print(f"  Frozen (BERT):       {n_frozen:,}")
    print(f"  Trainable (BERT):    {n_bert_trainable:,} (lr={bert_lr})")
    print(f"  Trainable (heads):   {n_head_trainable:,} (lr={head_lr})")

    optimizer = optim.AdamW([
        {"params": bert_params, "lr": bert_lr, "weight_decay": 0.01},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.0},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-7
    )

    # AMP (mixed precision) — only on CUDA
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(save_dir, exist_ok=True)

    # Redirect stdout to also write to a log file
    sys.stdout = Logger(os.path.join(save_dir, "training_log.txt"))

    best_val_corr = -1.0

    print("\n" + "=" * 90)
    print("Training (BERT + Differentiable EZ Reader) on GECO Corpus [GPU-optimized]")
    print(f"  Batch size: {batch_size} | Gradient accumulation steps: {accumulation_steps}")
    print(f"  Effective batch size: {batch_size * accumulation_steps}")
    print(f"  Mixed precision (AMP): {use_amp}")
    print("=" * 90)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()

        # Shuffle all per-participant observations
        epoch_data = train_raw.copy()
        random.shuffle(epoch_data)

        epoch_loss = 0.0
        epoch_trt = 0.0
        epoch_ffd = 0.0
        epoch_skip = 0.0
        n_samples = 0

        optimizer.zero_grad()

        n_batches = (len(epoch_data) + batch_size - 1) // batch_size
        for step in range(n_batches):
            batch = epoch_data[step * batch_size : (step + 1) * batch_size]
            word_lists, pred_vals, wlens, h_trt, h_ffd, h_skip = collate_sentences(batch, device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(word_lists, pred_vals, wlens)
            loss, parts = compute_loss(pred, h_trt, h_ffd, h_skip)

            # Scale loss for gradient accumulation
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
            epoch_skip += parts['skip']
            n_samples += len(batch)

        epoch_loss /= n_samples
        epoch_trt /= n_samples
        epoch_ffd /= n_samples
        epoch_skip /= n_samples
        elapsed = time.time() - t0

        # ---- Detailed validation (on aggregated data) ----
        val_metrics = evaluate_detailed(model, val_agg, device)
        scheduler.step(val_metrics['loss'])
        bert_lr_now = optimizer.param_groups[0]['lr']
        head_lr_now = optimizer.param_groups[1]['lr']

        is_best = val_metrics['r_trt'] > best_val_corr
        show = True  # Always show for BERT (slower training)

        if show:
            print(f"\n[Epoch {epoch:3d}/{num_epochs}] {elapsed:.1f}s | "
                  f"bert_lr={bert_lr_now:.2e} head_lr={head_lr_now:.2e}")
            print(f"  Train: loss={epoch_loss:.1f} "
                  f"(trt={epoch_trt:.0f} ffd={epoch_ffd:.0f} skip={epoch_skip:.3f}) "
                  f"| {n_samples:,} samples")
            print(f"  Val:   loss={val_metrics['loss']:.1f} | "
                  f"r_TRT={val_metrics['r_trt']:.3f}  "
                  f"r_FFD={val_metrics['r_ffd']:.3f}  "
                  f"r_skip={val_metrics['r_skip']:.3f}")
            print(f"  Val:   MAE_TRT={val_metrics['mae_trt']:.1f}ms  "
                  f"MAE_FFD={val_metrics['mae_ffd']:.1f}ms")
            print(f"  Pred:  mean_TRT={val_metrics['mean_pred_trt']:.0f}ms "
                  f"(human={val_metrics['mean_human_trt']:.0f}ms) | "
                  f"L1={val_metrics['mean_l1']:.0f}+/-{val_metrics['std_l1']:.0f}ms  "
                  f"L2={val_metrics['mean_l2']:.0f}+/-{val_metrics['std_l2']:.0f}ms")

            print_ezreader_params(model)
            print_sample_predictions(model, train_agg, device, n_sentences=2, n_words=8)

            if is_best:
                print(f"  ** NEW BEST (r_TRT={val_metrics['r_trt']:.3f}) **")

        # ---- Save best ----
        if is_best:
            best_val_corr = val_metrics['r_trt']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'bert_model_name': bert_model_name,
                'freeze_bert_layers': freeze_bert_layers,
                'val_metrics': val_metrics,
            }, os.path.join(save_dir, "best_model_bert.pt"))

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
          f"r_skip={test_metrics['r_skip']:.3f}")
    print(f"  MAE_TRT={test_metrics['mae_trt']:.1f}ms  "
          f"MAE_FFD={test_metrics['mae_ffd']:.1f}ms")
    print(f"  mean_TRT={test_metrics['mean_pred_trt']:.0f}ms "
          f"(human={test_metrics['mean_human_trt']:.0f}ms)")

    print("\nSample test predictions:")
    print_sample_predictions(model, test_agg, device, n_sentences=3, n_words=10)

    print("Final learned parameters:")
    print_ezreader_params(model)


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    save_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints_v1/geco_bert")

    train(
        data_dir=data_dir,
        num_epochs=50,
        bert_lr=2e-5,
        head_lr=5e-4,
        batch_size=16,
        accumulation_steps=4,
        save_dir=save_dir,
        bert_model_name="bert-base-uncased",
        freeze_bert_layers=8,
    )
