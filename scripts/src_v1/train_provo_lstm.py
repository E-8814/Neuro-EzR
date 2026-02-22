"""
Supervised training for the Differentiable Neural EZ Reader (v2).

Now uses ALL per-participant data (12,684 observations) for training,
not just the 151 aggregated sentences.

Validation/test still uses aggregated data for clean correlation metrics.
"""

import os
import sys
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ez_reader'))

from model_lstm import NeuralEZReader, Vocabulary
from data_loader import (
    load_provo, split_dataset,
    aggregate_by_sentence, split_aggregated,
)


# --------------------------------------------------------------------------- #
#  Loss function
# --------------------------------------------------------------------------- #

def compute_loss(pred, human_trt, human_ffd, human_gaze, human_skip):
    """
    Combined loss with 4 components:
      - MSE on total reading time (main signal)
      - MSE on first fixation duration (drives L1)
      - MSE on gaze duration (drives L1+L2, forces L2 to be meaningful)
      - BCE on skip probability (drives skip head)
    """
    trt_loss = nn.functional.mse_loss(pred['total_reading_time'], human_trt)
    ffd_loss = nn.functional.mse_loss(pred['first_fixation'], human_ffd)
    gaze_loss = nn.functional.mse_loss(pred['gaze_duration'], human_gaze)

    skip_pred = pred['skip_prob'].clamp(1e-6, 1 - 1e-6)
    skip_loss = nn.functional.binary_cross_entropy(skip_pred, human_skip)

    # Gaze loss is key: gaze = (1-skip)*(L1+L2), while FFD = (1-skip)*L1
    # So gaze_loss - ffd_loss effectively trains L2
    total = 0.3 * trt_loss + 0.2 * ffd_loss + 0.2 * gaze_loss + 0.3 * skip_loss

    return total, {
        'trt': trt_loss.item(),
        'ffd': ffd_loss.item(),
        'gaze': gaze_loss.item(),
        'skip': skip_loss.item(),
        'total': total.item(),
    }


# --------------------------------------------------------------------------- #
#  Evaluation (on aggregated data for clean metrics)
# --------------------------------------------------------------------------- #

def evaluate_detailed(model, agg_data, vocab, device):
    """
    Evaluate on aggregated data. Returns loss, correlations for TRT/FFD/skip,
    plus per-word prediction arrays for analysis.
    """
    model.eval()
    all_pred_trt, all_human_trt = [], []
    all_pred_ffd, all_human_ffd = [], []
    all_pred_skip, all_human_skip = [], []
    all_pred_l1, all_pred_l2 = [], []
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for agg in agg_data:
            token_ids = vocab.encode_sentence(agg.tokens).unsqueeze(0).to(device)
            pred_vals = torch.tensor(agg.predictabilities, dtype=torch.float32).unsqueeze(0).to(device)
            wlens = torch.tensor([len(t) for t in agg.tokens], dtype=torch.float32).unsqueeze(0).to(device)
            h_trt = torch.tensor(agg.mean_trt, dtype=torch.float32).unsqueeze(0).to(device)
            h_ffd = torch.tensor(agg.mean_ffd, dtype=torch.float32).unsqueeze(0).to(device)
            h_gaze = torch.tensor(agg.mean_gaze, dtype=torch.float32).unsqueeze(0).to(device)
            h_skip = torch.tensor(agg.skip_rate, dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(token_ids, pred_vals, wlens)
            loss, _ = compute_loss(pred, h_trt, h_ffd, h_gaze, h_skip)
            total_loss += loss.item()
            n += 1

            all_pred_trt.extend(pred['total_reading_time'][0].cpu().tolist())
            all_human_trt.extend(agg.mean_trt)
            all_pred_ffd.extend(pred['first_fixation'][0].cpu().tolist())
            all_human_ffd.extend(agg.mean_ffd)
            all_pred_skip.extend(pred['skip_prob'][0].cpu().tolist())
            all_human_skip.extend(agg.skip_rate)
            all_pred_l1.extend(pred['L1'][0].cpu().tolist())
            all_pred_l2.extend(pred['L2'][0].cpu().tolist())

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
#  Print detailed per-word predictions for sample sentences
# --------------------------------------------------------------------------- #

def print_sample_predictions(model, agg_data, vocab, device, n_sentences=3, n_words=8):
    """Print per-word predictions for a few sample sentences."""
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


# --------------------------------------------------------------------------- #
#  Print learned EZ Reader parameters
# --------------------------------------------------------------------------- #

def print_ezreader_params(model):
    """Print the learned differentiable EZ Reader parameters."""
    ezr = model.ezreader
    print("  Learned EZ Reader parameters:")
    print(f"    saccade_time     = {ezr.saccade_time.item():.1f}ms (init=150)")
    print(f"    attention_shift  = {ezr.attention_shift.item():.1f}ms (init=25)")
    print(f"    eccentricity     = {ezr.eccentricity.item():.4f} (init=0.1)")
    print(f"    integration_cost = {ezr.integration_cost.item():.4f} (init=0.08)")
    print(f"    l1_scale         = {model.l1_scale.item():.1f} (init=8)")
    print(f"    l2_scale         = {model.l2_scale.item():.1f} (init=8)")


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #

def train(
    data_dir="../data",
    num_epochs=100,
    lr=1e-3,
    save_dir="../checkpoints_v1/provo_lstm",
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load ALL data ----
    et_path = os.path.join(data_dir, "Provo_Corpus-Eyetracking_Data.csv")
    print("Loading Provo Corpus...")
    raw_dataset = load_provo(et_path)
    print(f"  Raw per-participant observations: {len(raw_dataset):,}")

    # Split raw data by text for training (per-participant = more data)
    train_raw, val_raw, test_raw = split_dataset(raw_dataset)
    print(f"  Train: {len(train_raw):,} | Val: {len(val_raw):,} | Test: {len(test_raw):,}")

    # Also aggregate for clean evaluation metrics
    aggregated = aggregate_by_sentence(raw_dataset, min_participants=10)
    _, val_agg, test_agg = split_aggregated(aggregated)
    # Also create train aggregated for comparison
    train_text_ids = set(sd.text_id for sd in train_raw)
    train_agg = [a for a in aggregated if a.text_id in train_text_ids]
    print(f"  Aggregated: {len(train_agg)} train | {len(val_agg)} val | {len(test_agg)} test sentences")

    # ---- Vocabulary ----
    vocab = Vocabulary()
    vocab.build_from_sentences([sd.tokens for sd in raw_dataset])
    vocab.freeze()
    print(f"  Vocab: {len(vocab)} words")

    # ---- Model ----
    model = NeuralEZReader(vocab_size=len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} parameters")

    os.makedirs(save_dir, exist_ok=True)
    best_val_corr = -1.0

    print("\n" + "=" * 90)
    print("Training (Differentiable EZ Reader) - ALL per-participant data")
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

        for sd in epoch_data:
            tokens = sd.tokens
            token_ids = vocab.encode_sentence(tokens).unsqueeze(0).to(device)
            pred_vals = torch.tensor(
                [w.predictability for w in sd.words], dtype=torch.float32
            ).unsqueeze(0).to(device)
            wlens = torch.tensor(
                [len(t) for t in tokens], dtype=torch.float32
            ).unsqueeze(0).to(device)

            # Per-participant targets
            h_trt = torch.tensor(sd.total_reading_times, dtype=torch.float32).unsqueeze(0).to(device)
            h_ffd = torch.tensor(sd.first_fixation_durations, dtype=torch.float32).unsqueeze(0).to(device)
            h_gaze = torch.tensor(sd.gaze_durations, dtype=torch.float32).unsqueeze(0).to(device)
            h_skip = torch.tensor(
                [1.0 if s else 0.0 for s in sd.skip_flags], dtype=torch.float32
            ).unsqueeze(0).to(device)

            pred = model(token_ids, pred_vals, wlens)
            loss, parts = compute_loss(pred, h_trt, h_ffd, h_gaze, h_skip)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += parts['total']
            epoch_trt += parts['trt']
            epoch_ffd += parts['ffd']
            epoch_skip += parts['skip']
            n_samples += 1

        epoch_loss /= n_samples
        epoch_trt /= n_samples
        epoch_ffd /= n_samples
        epoch_skip /= n_samples
        elapsed = time.time() - t0

        # ---- Detailed validation (on aggregated data) ----
        val_metrics = evaluate_detailed(model, val_agg, vocab, device)
        scheduler.step(val_metrics['loss'])
        lr_now = optimizer.param_groups[0]['lr']

        is_best = val_metrics['r_trt'] > best_val_corr
        show = epoch <= 5 or epoch % 10 == 0 or is_best

        if show:
            print(f"\n[Epoch {epoch:3d}/{num_epochs}] {elapsed:.1f}s | lr={lr_now:.6f}")
            print(f"  Train: loss={epoch_loss:.1f} (trt={epoch_trt:.0f} ffd={epoch_ffd:.0f} skip={epoch_skip:.3f}) "
                  f"| {n_samples:,} samples")
            print(f"  Val:   loss={val_metrics['loss']:.1f} | "
                  f"r_TRT={val_metrics['r_trt']:.3f}  r_FFD={val_metrics['r_ffd']:.3f}  r_skip={val_metrics['r_skip']:.3f}")
            print(f"  Val:   MAE_TRT={val_metrics['mae_trt']:.1f}ms  MAE_FFD={val_metrics['mae_ffd']:.1f}ms")
            print(f"  Pred:  mean_TRT={val_metrics['mean_pred_trt']:.0f}ms (human={val_metrics['mean_human_trt']:.0f}ms) | "
                  f"L1={val_metrics['mean_l1']:.0f}+/-{val_metrics['std_l1']:.0f}ms  "
                  f"L2={val_metrics['mean_l2']:.0f}+/-{val_metrics['std_l2']:.0f}ms")

            # Learned parameters
            print_ezreader_params(model)

            # Per-word predictions
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
    test_metrics = evaluate_detailed(model, test_agg, vocab, device)
    print(f"\nTest set results:")
    print(f"  r_TRT={test_metrics['r_trt']:.3f}  r_FFD={test_metrics['r_ffd']:.3f}  r_skip={test_metrics['r_skip']:.3f}")
    print(f"  MAE_TRT={test_metrics['mae_trt']:.1f}ms  MAE_FFD={test_metrics['mae_ffd']:.1f}ms")
    print(f"  mean_TRT={test_metrics['mean_pred_trt']:.0f}ms (human={test_metrics['mean_human_trt']:.0f}ms)")

    print("\nSample test predictions:")
    print_sample_predictions(model, test_agg, vocab, device, n_sentences=3, n_words=10)

    # Learned parameters
    print("Final learned parameters:")
    print_ezreader_params(model)


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    save_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints_v1/provo_lstm")

    train(
        data_dir=data_dir,
        num_epochs=100,
        lr=1e-3,
        save_dir=save_dir,
    )
