"""
GPU-efficient training for the Differentiable Neural EZ Reader v2 (LSTM) on GECO.

Changes from v1 (train_geco_lstm_gpu.py):
  - Uses DifferentiableEZReader v2 (improved FFD + regressions)
  - Skip prior regularizer: lambda_prior * (mean_skip - 0.45)^2
  - L1 range regularizer: lambda_l1 * mean(relu(L1 - 200))
  - L1 lower bound regularizer: lambda_l1_lower * mean(relu(50 - L1))
  - TRT scale loss: lambda_trt_scale * (mean_pred_TRT - mean_human_TRT)^2
  - Loss weights: 0.3*TRT + 0.2*FFD + 0.2*Gaze + 0.3*Skip + regularizers
  - Saves to checkpoints_v2/geco_lstm/
"""

import os
import sys
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'archive', 'original_ezreader'))

from model_lstm import NeuralEZReader, Vocabulary
from data_loader import aggregate_by_sentence
from geco_loader import load_geco, split_geco


# --------------------------------------------------------------------------- #
#  Regularization hyperparameters
# --------------------------------------------------------------------------- #

LAMBDA_PRIOR = 10.0   # skip prior weight
LAMBDA_L1 = 0.01      # L1 upper range penalty weight
LAMBDA_L1_LOWER = 0.01  # L1 lower bound penalty weight
LAMBDA_TRT_SCALE = 0.001  # TRT scale matching penalty weight
SKIP_TARGET = 0.45     # target mean skip rate
L1_MAX = 200.0         # soft L1 ceiling (ms)
L1_MIN = 50.0          # soft L1 floor (ms)


# --------------------------------------------------------------------------- #
#  Dataset: pre-tensorized with padding
# --------------------------------------------------------------------------- #

class SentenceDataset(Dataset):
    """Pre-tensorized sentence dataset for batched training."""

    def __init__(self, sentence_data_list, vocab):
        self.samples = []
        for sd in sentence_data_list:
            tokens = sd.tokens
            token_ids = vocab.encode_sentence(tokens)
            pred_vals = torch.tensor(
                [w.predictability for w in sd.words], dtype=torch.float32
            )
            wlens = torch.tensor(
                [len(t) for t in tokens], dtype=torch.float32
            )
            h_trt = torch.tensor(sd.total_reading_times, dtype=torch.float32)
            h_ffd = torch.tensor(sd.first_fixation_durations, dtype=torch.float32)
            h_gaze = torch.tensor(sd.gaze_durations, dtype=torch.float32)
            h_skip = torch.tensor(
                [1.0 if s else 0.0 for s in sd.skip_flags], dtype=torch.float32
            )
            self.samples.append((token_ids, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class AggDataset(Dataset):
    """Pre-tensorized aggregated dataset for batched evaluation."""

    def __init__(self, agg_list, vocab):
        self.samples = []
        self.agg_refs = agg_list
        for agg in agg_list:
            token_ids = vocab.encode_sentence(agg.tokens)
            pred_vals = torch.tensor(agg.predictabilities, dtype=torch.float32)
            wlens = torch.tensor([len(t) for t in agg.tokens], dtype=torch.float32)
            h_trt = torch.tensor(agg.mean_trt, dtype=torch.float32)
            h_ffd = torch.tensor(agg.mean_ffd, dtype=torch.float32)
            h_gaze = torch.tensor(agg.mean_gaze, dtype=torch.float32)
            h_skip = torch.tensor(agg.skip_rate, dtype=torch.float32)
            self.samples.append((token_ids, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_pad(batch):
    """Pad variable-length sentences to the max length in the batch, return mask."""
    token_ids, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip = zip(*batch)

    lengths = [len(t) for t in token_ids]
    max_len = max(lengths)
    B = len(batch)

    t_ids = torch.zeros(B, max_len, dtype=torch.long)
    t_pred = torch.zeros(B, max_len)
    t_wlen = torch.zeros(B, max_len)
    t_trt = torch.zeros(B, max_len)
    t_ffd = torch.zeros(B, max_len)
    t_gaze = torch.zeros(B, max_len)
    t_skip = torch.zeros(B, max_len)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i in range(B):
        L = lengths[i]
        t_ids[i, :L] = token_ids[i]
        t_pred[i, :L] = pred_vals[i]
        t_wlen[i, :L] = wlens[i]
        t_trt[i, :L] = h_trt[i]
        t_ffd[i, :L] = h_ffd[i]
        t_gaze[i, :L] = h_gaze[i]
        t_skip[i, :L] = h_skip[i]
        mask[i, :L] = True

    return t_ids, t_pred, t_wlen, t_trt, t_ffd, t_gaze, t_skip, mask


# --------------------------------------------------------------------------- #
#  Masked loss (v2: with regularizers)
# --------------------------------------------------------------------------- #

def compute_loss_masked(pred, h_trt, h_ffd, h_gaze, h_skip, mask):
    """
    Masked loss: only compute over real (non-padded) positions.
    v2: adds skip prior + L1 range regularizers.
    """
    m = mask.flatten()
    p_trt = pred['total_reading_time'].flatten()[m]
    p_ffd = pred['first_fixation'].flatten()[m]
    p_gaze = pred['gaze_duration'].flatten()[m]
    p_skip = pred['skip_prob'].flatten()[m].clamp(1e-6, 1 - 1e-6)
    p_l1 = pred['L1'].flatten()[m]

    t_trt = h_trt.flatten()[m]
    t_ffd = h_ffd.flatten()[m]
    t_gaze = h_gaze.flatten()[m]
    t_skip = h_skip.flatten()[m]

    trt_loss = nn.functional.mse_loss(p_trt, t_trt)
    ffd_loss = nn.functional.mse_loss(p_ffd, t_ffd)
    gaze_loss = nn.functional.mse_loss(p_gaze, t_gaze)
    skip_loss = nn.functional.binary_cross_entropy(p_skip, t_skip)

    # Regularizers
    mean_skip = p_skip.mean()
    skip_prior = LAMBDA_PRIOR * (mean_skip - SKIP_TARGET) ** 2

    l1_excess = torch.nn.functional.relu(p_l1 - L1_MAX)
    l1_reg = LAMBDA_L1 * l1_excess.mean()

    l1_deficit = torch.nn.functional.relu(L1_MIN - p_l1)
    l1_lower_reg = LAMBDA_L1_LOWER * l1_deficit.mean()

    trt_scale = LAMBDA_TRT_SCALE * (p_trt.mean() - t_trt.mean()) ** 2

    total = 0.3 * trt_loss + 0.2 * ffd_loss + 0.2 * gaze_loss + 0.3 * skip_loss + skip_prior + l1_reg + l1_lower_reg + trt_scale

    return total, {
        'trt': trt_loss.item(),
        'ffd': ffd_loss.item(),
        'gaze': gaze_loss.item(),
        'skip': skip_loss.item(),
        'skip_prior': skip_prior.item(),
        'l1_reg': l1_reg.item(),
        'l1_lower': l1_lower_reg.item(),
        'trt_scale': trt_scale.item(),
        'total': total.item(),
    }


# --------------------------------------------------------------------------- #
#  Batched evaluation
# --------------------------------------------------------------------------- #

def evaluate_batched(model, dataloader, device):
    """Evaluate on batched data. Returns loss, correlations, error metrics."""
    model.eval()
    all_pred_trt, all_human_trt = [], []
    all_pred_ffd, all_human_ffd = [], []
    all_pred_skip, all_human_skip = [], []
    all_pred_l1, all_pred_l2 = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            t_ids, t_pred, t_wlen, t_trt, t_ffd, t_gaze, t_skip, mask = batch
            t_ids = t_ids.to(device)
            t_pred = t_pred.to(device)
            t_wlen = t_wlen.to(device)
            t_trt = t_trt.to(device)
            t_ffd = t_ffd.to(device)
            t_gaze = t_gaze.to(device)
            t_skip = t_skip.to(device)
            mask = mask.to(device)

            pred = model(t_ids, t_pred, t_wlen)
            loss, _ = compute_loss_masked(pred, t_trt, t_ffd, t_gaze, t_skip, mask)
            total_loss += loss.item()
            n_batches += 1

            m = mask.flatten()
            all_pred_trt.extend(pred['total_reading_time'].flatten()[m].cpu().tolist())
            all_human_trt.extend(t_trt.flatten()[m].cpu().tolist())
            all_pred_ffd.extend(pred['first_fixation'].flatten()[m].cpu().tolist())
            all_human_ffd.extend(t_ffd.flatten()[m].cpu().tolist())
            all_pred_skip.extend(pred['skip_prob'].flatten()[m].cpu().tolist())
            all_human_skip.extend(t_skip.flatten()[m].cpu().tolist())
            all_pred_l1.extend(pred['L1'].flatten()[m].cpu().tolist())
            all_pred_l2.extend(pred['L2'].flatten()[m].cpu().tolist())

    avg_loss = total_loss / max(n_batches, 1)

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
#  Print helpers
# --------------------------------------------------------------------------- #

def print_sample_predictions(model, agg_data, vocab, device, n_sentences=3, n_words=8):
    model.eval()
    with torch.no_grad():
        for s_idx in range(min(n_sentences, len(agg_data))):
            s = agg_data[s_idx]
            ids = vocab.encode_sentence(s.tokens).unsqueeze(0).to(device)
            pv = torch.tensor(s.predictabilities, dtype=torch.float32).unsqueeze(0).to(device)
            wl = torch.tensor([len(t) for t in s.tokens], dtype=torch.float32).unsqueeze(0).to(device)
            p = model(ids, pv, wl)

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
    """Print the learned differentiable EZ Reader v2 parameters."""
    ezr = model.ezreader
    print("  Learned EZ Reader v2 parameters:")
    print(f"    saccade_time          = {ezr.saccade_time.item():.1f}ms (init=150)")
    print(f"    attention_shift       = {ezr.attention_shift.item():.1f}ms (init=25)")
    print(f"    eccentricity          = {ezr.eccentricity.item():.4f} (init=0.1)")
    print(f"    ffd_offset            = {ezr.ffd_offset.item():.1f}ms (init=100)")
    print(f"    l2_contribution       = {ezr.l2_contribution.item():.4f} (init=0.3)")
    print(f"    regression_threshold  = {ezr.regression_threshold.item():.1f}ms (init=50)")
    print(f"    regression_sharpness  = {ezr.regression_sharpness.item():.4f} (init=0.1)")
    print(f"    regression_cost_scale = {ezr.regression_cost_scale.item():.4f} (init=1.0)")
    print(f"    l1_scale              = {model.l1_scale.item():.1f} (init=8)")
    print(f"    l2_scale              = {model.l2_scale.item():.1f} (init=8)")


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #

def train(
    data_dir="../data",
    num_epochs=100,
    batch_size=64,
    lr=1e-3,
    save_dir="../checkpoints_v2/geco_lstm",
    seed=42,
    num_workers=2,
    gpu=0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
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

    # Aggregate for clean evaluation metrics
    aggregated = aggregate_by_sentence(raw_dataset, min_participants=5)
    print(f"  Aggregated sentences (min 5 participants): {len(aggregated)}")

    train_text_ids = set(sd.text_id for sd in train_raw)
    val_text_ids = set(sd.text_id for sd in val_raw)
    train_agg = [a for a in aggregated if a.text_id in train_text_ids]
    val_agg = [a for a in aggregated if a.text_id in val_text_ids]
    test_agg = [a for a in aggregated if a.text_id not in train_text_ids and a.text_id not in val_text_ids]
    print(f"  Aggregated: {len(train_agg)} train | {len(val_agg)} val | {len(test_agg)} test sentences")

    # ---- Vocabulary ----
    vocab = Vocabulary()
    vocab.build_from_sentences([sd.tokens for sd in raw_dataset])
    vocab.freeze()
    print(f"  Vocab: {len(vocab)} words")

    # ---- Pre-tensorize datasets ----
    print("  Pre-tensorizing datasets...")
    t0 = time.time()
    train_dataset = SentenceDataset(train_raw, vocab)
    val_dataset = AggDataset(val_agg, vocab)
    test_dataset = AggDataset(test_agg, vocab)
    print(f"  Tensorized in {time.time()-t0:.1f}s")

    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_pad, num_workers=num_workers, pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_pad, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_pad, num_workers=0, pin_memory=True,
    )

    print(f"  Train: {len(train_loader)} batches of {batch_size}")

    # ---- Model ----
    model = NeuralEZReader(vocab_size=len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} parameters")

    os.makedirs(save_dir, exist_ok=True)
    best_val_corr = -1.0

    print("\n" + "=" * 90)
    print("Training (LSTM + Differentiable EZ Reader v2) on GECO Corpus [GPU-batched]")
    print(f"  Regularizers: skip_prior(lambda={LAMBDA_PRIOR}, target={SKIP_TARGET}) + "
          f"l1_range(lambda={LAMBDA_L1}, max={L1_MAX}) + "
          f"l1_lower(lambda={LAMBDA_L1_LOWER}, min={L1_MIN}) + "
          f"trt_scale(lambda={LAMBDA_TRT_SCALE})")
    print("=" * 90)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()

        epoch_loss = 0.0
        epoch_trt = 0.0
        epoch_ffd = 0.0
        epoch_skip = 0.0
        n_batches = 0

        for batch in train_loader:
            t_ids, t_pred, t_wlen, t_trt, t_ffd, t_gaze, t_skip, mask = batch
            t_ids = t_ids.to(device, non_blocking=True)
            t_pred = t_pred.to(device, non_blocking=True)
            t_wlen = t_wlen.to(device, non_blocking=True)
            t_trt = t_trt.to(device, non_blocking=True)
            t_ffd = t_ffd.to(device, non_blocking=True)
            t_gaze = t_gaze.to(device, non_blocking=True)
            t_skip = t_skip.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            pred = model(t_ids, t_pred, t_wlen)
            loss, parts = compute_loss_masked(pred, t_trt, t_ffd, t_gaze, t_skip, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += parts['total']
            epoch_trt += parts['trt']
            epoch_ffd += parts['ffd']
            epoch_skip += parts['skip']
            n_batches += 1

        epoch_loss /= n_batches
        epoch_trt /= n_batches
        epoch_ffd /= n_batches
        epoch_skip /= n_batches
        elapsed = time.time() - t0

        # ---- Validation ----
        val_metrics = evaluate_batched(model, val_loader, device)
        scheduler.step(val_metrics['loss'])
        lr_now = optimizer.param_groups[0]['lr']

        is_best = val_metrics['r_trt'] > best_val_corr
        show = True

        if show:
            print(f"\n[Epoch {epoch:3d}/{num_epochs}] {elapsed:.1f}s | lr={lr_now:.6f}")
            print(f"  Train: loss={epoch_loss:.1f} (trt={epoch_trt:.0f} ffd={epoch_ffd:.0f} skip={epoch_skip:.3f}) "
                  f"| {n_batches} batches")
            print(f"  Val:   loss={val_metrics['loss']:.1f} | "
                  f"r_TRT={val_metrics['r_trt']:.3f}  r_FFD={val_metrics['r_ffd']:.3f}  r_skip={val_metrics['r_skip']:.3f}")
            print(f"  Val:   MAE_TRT={val_metrics['mae_trt']:.1f}ms  MAE_FFD={val_metrics['mae_ffd']:.1f}ms")
            print(f"  Pred:  mean_TRT={val_metrics['mean_pred_trt']:.0f}ms (human={val_metrics['mean_human_trt']:.0f}ms) | "
                  f"L1={val_metrics['mean_l1']:.0f}+/-{val_metrics['std_l1']:.0f}ms  "
                  f"L2={val_metrics['mean_l2']:.0f}+/-{val_metrics['std_l2']:.0f}ms")

            print_ezreader_params(model)
            print_sample_predictions(model, train_agg, vocab, device, n_sentences=2, n_words=8)

            if is_best:
                print(f"  ** NEW BEST (r_TRT={val_metrics['r_trt']:.3f}) **")

        # ---- Save best ----
        if is_best:
            best_val_corr = val_metrics['r_trt']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'val_metrics': val_metrics,
            }, os.path.join(save_dir, "best_model_lstm.pt"))

    # ---- Final summary ----
    print("\n" + "=" * 90)
    print(f"Training complete!")
    print(f"Best validation r_TRT = {best_val_corr:.3f}")
    print("=" * 90)

    # ---- Test set ----
    test_metrics = evaluate_batched(model, test_loader, device)
    print(f"\nTest set results:")
    print(f"  r_TRT={test_metrics['r_trt']:.3f}  r_FFD={test_metrics['r_ffd']:.3f}  r_skip={test_metrics['r_skip']:.3f}")
    print(f"  MAE_TRT={test_metrics['mae_trt']:.1f}ms  MAE_FFD={test_metrics['mae_ffd']:.1f}ms")
    print(f"  mean_TRT={test_metrics['mean_pred_trt']:.0f}ms (human={test_metrics['mean_human_trt']:.0f}ms)")

    print("\nSample test predictions:")
    print_sample_predictions(model, test_agg, vocab, device, n_sentences=3, n_words=10)

    print("Final learned parameters:")
    print_ezreader_params(model)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    save_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints_v2/geco_lstm")

    train(
        data_dir=data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=save_dir,
        gpu=args.gpu,
    )
