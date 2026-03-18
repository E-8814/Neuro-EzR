"""
Tune the Differentiable EZ Reader on GECO training data.

Keeps formula parameters (alpha1, alpha2, alpha3, delta, eccentricity) at
their literature defaults and only optimizes the 5 EZ Reader dynamics
parameters via gradient descent:
  - saccade_time, attention_shift, skip_sharpness, eccentricity, integration_cost

This is the fairest comparison: the neural models (LSTM/BERT) also use the
same DiffEZReader with learnable dynamics params. The only difference is the
encoder (formula vs LSTM vs BERT).

Tuning all 10 params (formula + EZ) causes the formula to degenerate
(alpha2→0, alpha3→0) because the model finds it easier to ignore
frequency/predictability. Freezing the formula prevents this.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 -u src_diff_gpu/tune_diff_ezreader.py
"""

import os
import sys
import csv
import math
import time
import random

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'archive', 'original_ezreader'))

from data_loader import aggregate_by_sentence
from geco_loader import load_geco, split_geco


# --------------------------------------------------------------------------- #
#  Differentiable L1/L2 formula
# --------------------------------------------------------------------------- #

class DifferentiableFormula(nn.Module):
    """
    Differentiable version of the EZ Reader L1/L2 formulas.

    L1 = (alpha1 - alpha2 * log(freq) - alpha3 * pred) * ecc_factor
    L2 = delta * (alpha1 - alpha2 * log(freq) - alpha3 * pred)

    All parameters are learnable via gradient descent.
    """

    def __init__(self):
        super().__init__()
        # Initialize from literature defaults
        # We store raw (unconstrained) parameters and use softplus/sigmoid
        # in forward() to enforce physical constraints:
        #   alpha1 > 0, alpha2 > 0, alpha3 > 0, 0 < delta < 1, ecc > 1
        self.alpha1 = nn.Parameter(torch.tensor(104.0))
        # Store alpha2/alpha3 in log-space so softplus keeps them positive
        self._alpha2_raw = nn.Parameter(torch.tensor(3.4))
        self._alpha3_raw = nn.Parameter(torch.tensor(39.0))
        # delta in logit-space so sigmoid keeps it in (0, 1)
        # sigmoid(-0.663) ≈ 0.34
        self._delta_raw = nn.Parameter(torch.tensor(-0.663))
        # eccentricity offset (added to 1.0, kept positive via softplus)
        self._ecc_offset = nn.Parameter(torch.tensor(0.15))  # 1.0 + 0.15 = 1.15

    @property
    def alpha2(self):
        return torch.nn.functional.softplus(self._alpha2_raw)

    @property
    def alpha3(self):
        return torch.nn.functional.softplus(self._alpha3_raw)

    @property
    def delta(self):
        return torch.sigmoid(self._delta_raw)

    @property
    def eccentricity(self):
        return 1.0 + torch.nn.functional.softplus(self._ecc_offset)

    def forward(self, log_freq, predictability, word_lengths):
        """
        Args:
            log_freq:        (batch, seq_len) - log(frequency) of each word
            predictability:  (batch, seq_len) - cloze predictability (0-1)
            word_lengths:    (batch, seq_len) - character count per word

        Returns:
            L1: (batch, seq_len) in ms
            L2: (batch, seq_len) in ms
        """
        alpha2 = self.alpha2
        alpha3 = self.alpha3
        delta = self.delta
        ecc = self.eccentricity

        # Base processing time (before eccentricity)
        base = self.alpha1 - alpha2 * log_freq - alpha3 * predictability

        # Eccentricity: scale by ecc^(distance + (wordlen-1)/2)
        # Using distance=0 (fixation at word start) for simplicity,
        # so exponent = (wordlen - 1) / 2
        exponent = (word_lengths - 1.0) / 2.0
        ecc_factor = torch.pow(ecc, exponent)

        L1 = base * ecc_factor
        L2 = delta * base

        # Clamp to positive values
        L1 = L1.clamp(min=1.0)
        L2 = L2.clamp(min=1.0)

        return L1, L2


# --------------------------------------------------------------------------- #
#  Constrained Diff EZ Reader (for tuning only — doesn't modify shared module)
# --------------------------------------------------------------------------- #

class ConstrainedDiffEZReader(nn.Module):
    """
    Differentiable EZ Reader with constrained parameters.

    All parameters are kept in physically meaningful ranges:
      - saccade_time > 0      (via softplus, init ~150ms)
      - attention_shift > 0   (via softplus, init ~25ms)
      - skip_sharpness > 0    (via softplus, init ~8)
      - eccentricity > 0      (via softplus, init ~0.1)
      - integration_cost > 0  (via softplus, init ~0.08)
    """

    def __init__(self):
        super().__init__()
        # Raw unconstrained params → softplus → positive values
        # For correct init, use softplus_inverse(target) = log(exp(target) - 1)
        # For large values (>>1), softplus_inv ≈ value. For small values it differs.
        def sp_inv(x):
            return math.log(math.exp(x) - 1.0)

        self._saccade_raw = nn.Parameter(torch.tensor(sp_inv(150.0)))   # → 150.0
        self._attn_shift_raw = nn.Parameter(torch.tensor(sp_inv(25.0))) # → 25.0
        self._skip_sharp_raw = nn.Parameter(torch.tensor(sp_inv(8.0)))  # → 8.0
        self._ecc_raw = nn.Parameter(torch.tensor(sp_inv(0.1)))         # → 0.1
        self._integ_raw = nn.Parameter(torch.tensor(sp_inv(0.08)))      # → 0.08

    @property
    def saccade_time(self):
        return torch.nn.functional.softplus(self._saccade_raw)

    @property
    def attention_shift(self):
        return torch.nn.functional.softplus(self._attn_shift_raw)

    @property
    def skip_sharpness(self):
        return torch.nn.functional.softplus(self._skip_sharp_raw)

    @property
    def eccentricity(self):
        return torch.nn.functional.softplus(self._ecc_raw)

    @property
    def integration_cost(self):
        return torch.nn.functional.softplus(self._integ_raw)

    def forward(self, L1, L2, skip_input, word_lengths):
        skip_prob = torch.sigmoid(self.skip_sharpness * (skip_input - 0.5))

        ecc_scale = 1.0 + self.eccentricity * (word_lengths - 4.0).clamp(min=0)

        L1_scaled = L1 * ecc_scale
        first_fixation = L1_scaled
        gaze_duration = L1_scaled + L2

        integration_penalty = self.integration_cost * (L1_scaled + L2)

        fixate_prob = 1.0 - skip_prob
        overhead = self.saccade_time + self.attention_shift
        total_reading_time = fixate_prob * (gaze_duration + overhead + integration_penalty)

        return {
            'total_reading_time': total_reading_time,
            'first_fixation': first_fixation,
            'gaze_duration': gaze_duration,
            'skip_prob': skip_prob,
        }


# --------------------------------------------------------------------------- #
#  Full tunable model: Formula → Constrained Diff EZ Reader
# --------------------------------------------------------------------------- #

class TunableDiffEZReader(nn.Module):
    """
    Complete tunable formula model:
        word features → DifferentiableFormula → L1, L2
        → ConstrainedDiffEZReader → TRT, FFD, skip_prob
    """

    def __init__(self):
        super().__init__()
        self.formula = DifferentiableFormula()
        self.ezreader = ConstrainedDiffEZReader()

    def forward(self, log_freq, predictability, word_lengths):
        L1, L2 = self.formula(log_freq, predictability, word_lengths)
        result = self.ezreader(L1, L2, predictability, word_lengths)
        result['L1'] = L1
        result['L2'] = L2
        return result


# --------------------------------------------------------------------------- #
#  Data preparation
# --------------------------------------------------------------------------- #

def load_subtlexus(path):
    freq = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            freq[row['Word'].lower()] = int(row['FREQcount'])
    return freq


def get_real_frequency(word, subtlex):
    w = word.lower().strip(".,;:!?\"'()[]{}").replace("\u2019", "'")
    if w in subtlex:
        return max(1, subtlex[w])
    for variant in [w.replace("'", ""), w.split("'")[0], w.split("-")[0]]:
        if variant in subtlex:
            return max(1, subtlex[variant])
    length = len(w)
    if length <= 3:   return 50000
    elif length <= 5: return 10000
    elif length <= 7: return 2000
    else:             return 500


def prepare_sentence(agg, subtlex):
    """Convert an AggregatedSentence into tensors."""
    tokens = agg.tokens
    preds = agg.predictabilities
    wlens = [len(t) for t in tokens]
    freqs = [get_real_frequency(t, subtlex) for t in tokens]
    log_freqs = [math.log(max(1, f)) for f in freqs]

    return {
        'log_freq': log_freqs,
        'predictability': preds,
        'word_lengths': [float(w) for w in wlens],
        'trt': agg.mean_trt,
        'ffd': agg.mean_ffd,
        'gaze': agg.mean_gaze,
        'skip': agg.skip_rate,
    }


def collate_batch(batch, device):
    """Pad a list of sentence dicts into batched tensors."""
    max_len = max(len(s['trt']) for s in batch)
    B = len(batch)

    log_freq = torch.zeros(B, max_len, device=device)
    pred = torch.zeros(B, max_len, device=device)
    wlen = torch.zeros(B, max_len, device=device)
    trt = torch.zeros(B, max_len, device=device)
    ffd = torch.zeros(B, max_len, device=device)
    gaze = torch.zeros(B, max_len, device=device)
    skip = torch.zeros(B, max_len, device=device)
    mask = torch.zeros(B, max_len, device=device)

    for i, s in enumerate(batch):
        n = len(s['trt'])
        log_freq[i, :n] = torch.tensor(s['log_freq'])
        pred[i, :n] = torch.tensor(s['predictability'])
        wlen[i, :n] = torch.tensor(s['word_lengths'])
        trt[i, :n] = torch.tensor(s['trt'])
        ffd[i, :n] = torch.tensor(s['ffd'])
        gaze[i, :n] = torch.tensor(s['gaze'])
        skip[i, :n] = torch.tensor(s['skip'])
        mask[i, :n] = 1.0

    return log_freq, pred, wlen, trt, ffd, gaze, skip, mask


# --------------------------------------------------------------------------- #
#  Loss function (same as neural models)
# --------------------------------------------------------------------------- #

def compute_loss(pred, trt, ffd, skip, mask):
    """MSE on TRT + FFD, BCE on skip, masked for padding."""
    n = mask.sum()
    if n == 0:
        return torch.tensor(0.0, device=mask.device)

    trt_loss = ((pred['total_reading_time'] - trt) ** 2 * mask).sum() / n
    ffd_loss = ((pred['first_fixation'] - ffd) ** 2 * mask).sum() / n

    skip_pred = pred['skip_prob'].clamp(1e-6, 1 - 1e-6)
    skip_loss = -(skip * torch.log(skip_pred) + (1 - skip) * torch.log(1 - skip_pred))
    skip_loss = (skip_loss * mask).sum() / n

    # Same weights as neural training scripts
    w_trt, w_ffd, w_skip = 1.0, 1.0, 50.0
    total = w_trt * trt_loss + w_ffd * ffd_loss + w_skip * skip_loss

    return total


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #

def evaluate(model, data, subtlex, device, batch_size=64):
    """Evaluate on aggregated data, return correlations."""
    model.eval()
    all_h_trt, all_p_trt = [], []
    all_h_ffd, all_p_ffd = [], []
    all_h_skip, all_p_skip = [], []

    prepared = [prepare_sentence(s, subtlex) for s in data]

    with torch.no_grad():
        for i in range(0, len(prepared), batch_size):
            batch = prepared[i:i+batch_size]
            log_freq, pred, wlen, trt, ffd, gaze, skip, mask = collate_batch(batch, device)

            result = model(log_freq, pred, wlen)

            for b in range(len(batch)):
                n = int(mask[b].sum().item())
                all_h_trt.extend(trt[b, :n].cpu().tolist())
                all_p_trt.extend(result['total_reading_time'][b, :n].cpu().tolist())
                all_h_ffd.extend(ffd[b, :n].cpu().tolist())
                all_p_ffd.extend(result['first_fixation'][b, :n].cpu().tolist())
                all_h_skip.extend(skip[b, :n].cpu().tolist())
                all_p_skip.extend(result['skip_prob'][b, :n].cpu().tolist())

    r_trt = np.corrcoef(all_h_trt, all_p_trt)[0, 1] if len(all_h_trt) > 2 else 0.0
    r_ffd = np.corrcoef(all_h_ffd, all_p_ffd)[0, 1] if len(all_h_ffd) > 2 else 0.0
    r_skip = np.corrcoef(all_h_skip, all_p_skip)[0, 1] if len(all_h_skip) > 2 else 0.0
    mae_trt = np.mean(np.abs(np.array(all_h_trt) - np.array(all_p_trt)))

    return {
        'r_trt': r_trt, 'r_ffd': r_ffd, 'r_skip': r_skip,
        'mae_trt': mae_trt,
        'mean_pred_trt': np.mean(all_p_trt),
        'mean_pred_ffd': np.mean(all_p_ffd),
        'mean_pred_skip': np.mean(all_p_skip),
    }


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #

def train(num_epochs=200, lr=0.01, batch_size=64):
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints_tuned_diff')
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\nLoading GECO Corpus...")
    reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
    material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
    pred_path = os.path.join(data_dir, "geco_predictability.pkl")

    geco_raw = load_geco(reading_path, material_path, pred_path)
    train_raw, val_raw, test_raw = split_geco(geco_raw)

    # Aggregate by sentence
    all_agg = aggregate_by_sentence(geco_raw, min_participants=5)
    train_ids = set(sd.text_id for sd in train_raw)
    val_ids = set(sd.text_id for sd in val_raw)
    test_ids = set(sd.text_id for sd in test_raw)

    train_agg = [a for a in all_agg if a.text_id in train_ids]
    val_agg = [a for a in all_agg if a.text_id in val_ids]
    test_agg = [a for a in all_agg if a.text_id not in train_ids and a.text_id not in val_ids]

    print(f"  Train: {len(train_agg)} sentences")
    print(f"  Val:   {len(val_agg)} sentences")
    print(f"  Test:  {len(test_agg)} sentences")

    # Load frequency data
    print("Loading SUBTLEXus...")
    subtlex = load_subtlexus(os.path.join(data_dir, 'SUBTLEXus.txt'))
    print(f"  {len(subtlex):,} entries")

    # Prepare all training data
    print("Preparing training data...")
    train_prepared = [prepare_sentence(s, subtlex) for s in train_agg]

    # Model
    model = TunableDiffEZReader().to(device)

    # Freeze formula parameters — only tune EZ Reader dynamics
    for param in model.formula.parameters():
        param.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]

    # Print initial parameters
    print("\n--- Initial Parameters ---")
    print(f"  Formula (FROZEN): alpha1={model.formula.alpha1.item():.1f}, "
          f"alpha2={model.formula.alpha2.item():.3f}, "
          f"alpha3={model.formula.alpha3.item():.1f}, "
          f"delta={model.formula.delta.item():.3f}, "
          f"ecc={model.formula.eccentricity.item():.3f}")
    print(f"  EZ Reader (tunable): saccade={model.ezreader.saccade_time.item():.1f}, "
          f"attn_shift={model.ezreader.attention_shift.item():.1f}, "
          f"skip_sharp={model.ezreader.skip_sharpness.item():.1f}, "
          f"ecc={model.ezreader.eccentricity.item():.3f}, "
          f"integ={model.ezreader.integration_cost.item():.3f}")

    # Evaluate before training
    print("\n--- Before Training ---")
    val_metrics = evaluate(model, val_agg, subtlex, device)
    print(f"  Val: r_TRT={val_metrics['r_trt']:.3f}  r_FFD={val_metrics['r_ffd']:.3f}  "
          f"r_skip={val_metrics['r_skip']:.3f}  MAE_TRT={val_metrics['mae_trt']:.1f}ms")

    # Optimizer — only EZ Reader parameters (formula is frozen)
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=15, factor=0.5
    )

    best_r_trt = -1.0
    best_epoch = 0

    print(f"\n{'='*90}")
    print(f"  Starting training: {num_epochs} epochs, lr={lr}, batch_size={batch_size}")
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable)} (EZ Reader only)")
    print(f"  Frozen parameters: {sum(p.numel() for p in model.formula.parameters())} (formula)")
    print(f"{'='*90}\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        random.shuffle(train_prepared)

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_prepared), batch_size):
            batch = train_prepared[i:i+batch_size]
            log_freq, pred, wlen, trt, ffd, gaze, skip, mask = collate_batch(batch, device)

            result = model(log_freq, pred, wlen)
            loss = compute_loss(result, trt, ffd, skip, mask)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        # Evaluate
        val_metrics = evaluate(model, val_agg, subtlex, device)
        scheduler.step(val_metrics['r_trt'])

        improved = ""
        if val_metrics['r_trt'] > best_r_trt:
            best_r_trt = val_metrics['r_trt']
            best_epoch = epoch
            improved = " *** BEST ***"

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r_trt': best_r_trt,
                'formula_params': 'literature defaults (frozen)',
                'ezreader_params': {
                    'saccade_time': model.ezreader.saccade_time.item(),
                    'attention_shift': model.ezreader.attention_shift.item(),
                    'skip_sharpness': model.ezreader.skip_sharpness.item(),
                    'eccentricity': model.ezreader.eccentricity.item(),
                    'integration_cost': model.ezreader.integration_cost.item(),
                },
            }, os.path.join(ckpt_dir, 'best_tuned_diff.pt'))

        # Print progress
        if epoch % 5 == 0 or epoch == 1 or improved:
            print(f"  Epoch {epoch:3d} | loss={avg_loss:8.1f} | "
                  f"r_TRT={val_metrics['r_trt']:.3f}  r_FFD={val_metrics['r_ffd']:.3f}  "
                  f"r_skip={val_metrics['r_skip']:.3f}  MAE={val_metrics['mae_trt']:.1f}ms | "
                  f"pred_TRT={val_metrics['mean_pred_trt']:.0f}  "
                  f"pred_skip={val_metrics['mean_pred_skip']:.2f}"
                  f"{improved}")

            if epoch % 20 == 0:
                print(f"         EZ: sac={model.ezreader.saccade_time.item():.1f} "
                      f"attn={model.ezreader.attention_shift.item():.1f} "
                      f"skip_s={model.ezreader.skip_sharpness.item():.1f} "
                      f"ecc={model.ezreader.eccentricity.item():.3f} "
                      f"integ={model.ezreader.integration_cost.item():.3f}")

        # Early stopping
        if epoch - best_epoch > 50:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for 50 epochs)")
            break

    # Final evaluation on test set
    print(f"\n{'='*90}")
    print(f"  Training complete. Best epoch: {best_epoch} (r_TRT={best_r_trt:.3f})")
    print(f"{'='*90}")

    # Load best model
    ckpt = torch.load(os.path.join(ckpt_dir, 'best_tuned_diff.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    print("\n--- Final Tuned EZ Reader Parameters ---")
    print(f"  saccade_time     = {model.ezreader.saccade_time.item():.2f}ms  (default: 150)")
    print(f"  attention_shift  = {model.ezreader.attention_shift.item():.2f}ms  (default: 25)")
    print(f"  skip_sharpness   = {model.ezreader.skip_sharpness.item():.2f}     (default: 8)")
    print(f"  eccentricity     = {model.ezreader.eccentricity.item():.4f}   (default: 0.1)")
    print(f"  integration_cost = {model.ezreader.integration_cost.item():.4f}   (default: 0.08)")
    print(f"\n  Formula params (frozen at literature defaults):")
    print(f"  alpha1=104, alpha2=3.4, alpha3=39, delta=0.34, ecc=1.15")

    # Test set evaluation
    test_metrics = evaluate(model, test_agg, subtlex, device)
    print(f"\n--- Test Set Results ---")
    print(f"  r_TRT={test_metrics['r_trt']:.3f}  r_FFD={test_metrics['r_ffd']:.3f}  "
          f"r_skip={test_metrics['r_skip']:.3f}  MAE_TRT={test_metrics['mae_trt']:.1f}ms")

    # Validation set (for comparison with neural models)
    val_metrics = evaluate(model, val_agg, subtlex, device)
    print(f"\n--- Val Set Results ---")
    print(f"  r_TRT={val_metrics['r_trt']:.3f}  r_FFD={val_metrics['r_ffd']:.3f}  "
          f"r_skip={val_metrics['r_skip']:.3f}  MAE_TRT={val_metrics['mae_trt']:.1f}ms")

    print(f"\nCheckpoint saved to: {ckpt_dir}/best_tuned_diff.pt")


if __name__ == "__main__":
    train()
