"""
Train v4c_v2 with RANDOMIZED initial values for cognitive scalars
(parameter-recovery experiment, exp02).

Mirrors train_hybrid_v4c_v2_geco.py except:
  - model class is `model_llama_hybrid_v4c_v2_randinit.NeuralEZReaderHybrid`
  - the model is constructed with `init_seed=<seed>` and `jitter=0.5`
    so the cognitive scalars are perturbed within ±50% of Reichle 2003
    values at construction time
  - the sampled initial values are saved into the checkpoint for
    later recovery analysis
  - the training-recipe seed (passed by --seed) ALSO drives the init
    randomization, so each `--seed` argument gives a unique init +
    deterministic training trajectory
  - checkpoint dir / log filename use `hybrid_v4c_v2_randinit/`

All training-recipe details are identical to v4c_v2:
  - SIGMA2_FFD = 1500
  - LAMBDA_PRIOR = 30, data-anchored skip prior bounds
  - Combined-metric early stopping
  - Mid-epoch validation (5 vals/epoch)
  - cog_lr = 3e-4, head_lr = 5e-4, lm_lr = 2e-5
  - reload best_model.pt before test eval
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

from model_llama_hybrid_v4c_v2_randinit import (
    NeuralEZReaderHybrid,
    DEFAULT_JITTER,
)
from data_loader import aggregate_by_sentence
from geco_loader import load_geco, split_geco


LAMBDA_DELTA = 5.0
LAMBDA_PRIOR = 30.0
LAMBDA_SKIP_RESIDUAL = 0.001
SKIP_BOUND_HALFWIDTH = 0.03
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


def compute_loss(pred, human_trt, human_ffd, human_gaze, human_skip, delta,
                 skip_min, skip_max):
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
        F.relu(mean_skip - skip_max) + F.relu(skip_min - mean_skip)
    )

    skip_residual_reg = LAMBDA_SKIP_RESIDUAL * (residual_skip_logit ** 2).mean()

    total = (
        1.0 * trt_loss + 1.0 * ffd_loss + 1.0 * gaze_loss + 1.0 * skip_loss
        + skip_prior + delta_reg + skip_residual_reg
    )

    return total, {
        'trt': trt_mse.item(), 'ffd': ffd_mse.item(), 'gaze': gaze_mse.item(),
        'trt_norm': trt_loss.item(), 'ffd_norm': ffd_loss.item(),
        'gaze_norm': gaze_loss.item(), 'skip': skip_loss.item(),
        'skip_prior': skip_prior.item(),
        'skip_residual_reg': skip_residual_reg.item(),
        'total': total.item(),
    }


def evaluate_detailed(model, agg_data, device, subtlex, batch_size=8):
    model.eval()
    pt_, ht_, pf_, hf_, pg_, hg_, ps_, hs_ = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(agg_data), batch_size):
            batch = agg_data[i:i + batch_size]
            word_lists, freqs, wlens, *_ = collate_aggregated(batch, device, subtlex)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(word_lists, freqs, wlens)
            for b in range(len(batch)):
                seq_len = len(batch[b].tokens)
                pt_.extend(pred['conditional_trt'][b, :seq_len].cpu().tolist())
                ht_.extend(batch[b].mean_trt)
                pf_.extend(pred['first_fixation'][b, :seq_len].cpu().tolist())
                hf_.extend(batch[b].mean_ffd)
                pg_.extend(pred['gaze_duration'][b, :seq_len].cpu().tolist())
                hg_.extend(batch[b].mean_gaze)
                ps_.extend(pred['skip_prob'][b, :seq_len].cpu().tolist())
                hs_.extend(batch[b].skip_rate)

    def corr(a, b):
        a, b = np.array(a), np.array(b)
        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            return float(np.corrcoef(a, b)[0, 1])
        return 0.0
    pt, ht = np.array(pt_), np.array(ht_)
    pf, hf = np.array(pf_), np.array(hf_)
    pg, hg = np.array(pg_), np.array(hg_)
    ps, hs = np.array(ps_), np.array(hs_)
    return {
        'r_trt': corr(pt, ht), 'r_ffd': corr(pf, hf),
        'r_gaze': corr(pg, hg), 'r_skip': corr(ps, hs),
        'mae_trt': float(np.mean(np.abs(pt - ht))),
        'mae_ffd': float(np.mean(np.abs(pf - hf))),
        'mae_gaze': float(np.mean(np.abs(pg - hg))),
        'bias_trt': float(np.mean(pt) - np.mean(ht)),
        'bias_ffd': float(np.mean(pf) - np.mean(hf)),
        'bias_gaze': float(np.mean(pg) - np.mean(hg)),
        'mean_pred_trt': float(np.mean(pt)),
        'mean_human_trt': float(np.mean(ht)),
        'mean_skip': float(np.mean(ps)),
    }


def combined_metric(val):
    return 0.25 * (val['r_trt'] + val['r_ffd'] + val['r_gaze'] + val['r_skip'])


def print_init_summary(model):
    """Print the sampled vs current cog parameter values."""
    summary = model.get_init_summary()
    print(f"  Sampled init values (jitter={model.jitter}):")
    for name, (sampled, current) in summary.items():
        print(f"    {name:<20s} sampled={sampled:>9.4f}  current={current:>9.4f}")


def save_best_checkpoint(model, save_dir, epoch, val_step, val,
                         model_name, freeze_layers, hidden_dim,
                         init_seed, jitter):
    torch.save({
        'epoch': epoch, 'val_step': val_step,
        'model_state_dict': model.state_dict(),
        'model_name': model_name, 'freeze_layers': freeze_layers,
        'hidden_dim': hidden_dim,
        'init_seed': init_seed,
        'jitter': jitter,
        'sampled_init': model.sampled_init,
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
    }, os.path.join(save_dir, "best_model.pt"))


def train(
    data_dir="../data",
    num_epochs=50,
    lm_lr=2e-5, head_lr=5e-4, cog_lr=3e-4,
    batch_size=8, accumulation_steps=8,
    save_dir="../../checkpoints/hybrid_v4c_v2_randinit/geco_tinyllama",
    log_path="../../logs/train_hybrid_v4c_v2_randinit_geco.log",
    seed=42, jitter=DEFAULT_JITTER,
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    freeze_layers=12,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Random init seed: {seed} | jitter: ±{jitter*100:.0f}%")

    subtlex_path = os.path.join(data_dir, "SUBTLEXus.txt")
    print(f"Loading SUBTLEX from {subtlex_path}...")
    subtlex = load_subtlex(subtlex_path)

    print("Loading GECO Corpus...")
    raw_dataset = load_geco(
        os.path.join(data_dir, "Geco_MonolingualReadingData.csv"),
        os.path.join(data_dir, "Geco_EnglishMaterial.csv"),
        os.path.join(data_dir, "geco_predictability.pkl"),
    )
    train_raw, val_raw, _ = split_geco(raw_dataset)
    aggregated = aggregate_by_sentence(raw_dataset, min_participants=5)
    train_text_ids = set(sd.text_id for sd in train_raw)
    val_text_ids = set(sd.text_id for sd in val_raw)
    train_agg = [a for a in aggregated if a.text_id in train_text_ids]
    val_agg = [a for a in aggregated if a.text_id in val_text_ids]
    test_agg = [a for a in aggregated if a.text_id not in train_text_ids
                and a.text_id not in val_text_ids]
    print(f"  Aggregated: {len(train_agg)} train | {len(val_agg)} val | {len(test_agg)} test")

    all_train_skips = [
        1.0 if s else 0.0 for sd in train_raw for s in sd.skip_flags
    ]
    data_mean_skip = float(np.mean(all_train_skips))
    skip_min = max(0.0, data_mean_skip - SKIP_BOUND_HALFWIDTH)
    skip_max = min(1.0, data_mean_skip + SKIP_BOUND_HALFWIDTH)
    print(f"  Empirical skip rate (train): {data_mean_skip:.4f}")
    print(f"  Skip prior bounds: [{skip_min:.4f}, {skip_max:.4f}]")

    print(f"\nLoading model: {model_name}")
    print(f"  Freezing first {freeze_layers} layers")
    model = NeuralEZReaderHybrid(
        model_name=model_name,
        freeze_layers=freeze_layers,
        hidden_dim=256,
        init_seed=seed,
        jitter=jitter,
    ).to(device)

    print_init_summary(model)

    cog_name_prefixes = (
        "_delta_raw",
        "l1_base_offset", "l1_freq_coef",
        "ezreader._epsilon_raw",
        "ezreader._M1_raw", "ezreader._M2I_raw",
        "ezreader.lambda_refix", "ezreader.refix_pivot",
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
    total_val_steps = 0
    early_stop_triggered = False

    val_every_n = max(1, n_batches_per_epoch // N_VALS_PER_EPOCH)

    print("\n" + "=" * 100)
    print(f"Training v4c_v2_randinit  (jitter=±{jitter*100:.0f}% from Reichle 2003 init)")
    print("=" * 100)

    for epoch in range(1, num_epochs + 1):
        if early_stop_triggered:
            break

        t0 = time.time()
        model.train()

        epoch_data = train_raw.copy()
        random.shuffle(epoch_data)
        n_batches = (len(epoch_data) + batch_size - 1) // batch_size

        optimizer.zero_grad()

        for step in range(n_batches):
            batch = epoch_data[step * batch_size : (step + 1) * batch_size]
            word_lists, freqs, wlens, h_trt, h_ffd, h_gaze, h_skip = \
                collate_sentences(batch, device, subtlex)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(word_lists, freqs, wlens)
            loss, _ = compute_loss(
                pred, h_trt, h_ffd, h_gaze, h_skip, model.delta,
                skip_min, skip_max,
            )

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            is_last_batch = (step + 1) == n_batches
            if (step + 1) % val_every_n == 0 or is_last_batch:
                total_val_steps += 1
                val = evaluate_detailed(model, val_agg, device, subtlex)
                combined = combined_metric(val)

                print(f"  [val {total_val_steps}] epoch {epoch} batch {step+1}/{n_batches}")
                print(f"    combined={combined:.4f} | r_TRT={val['r_trt']:.3f} "
                      f"r_FFD={val['r_ffd']:.3f} r_Gaze={val['r_gaze']:.3f} r_skip={val['r_skip']:.3f}")
                print(f"    mean_skip={val['mean_skip']:.3f} | "
                      f"a1R={model.alpha1_reichle.item():.1f} "
                      f"a2R={model.alpha2_reichle.item():.3f} "
                      f"eps={model.ezreader.epsilon.item():.3f} "
                      f"M1={model.ezreader.M1.item():.1f} "
                      f"M2=I={model.ezreader.M2.item():.1f} "
                      f"d={model.delta.item():.3f}")

                if combined > best_val_corr:
                    print(f"    ** NEW BEST (combined={combined:.4f}) **")
                    best_val_corr = combined
                    patience_counter = 0
                    save_best_checkpoint(
                        model, save_dir, epoch, total_val_steps, val,
                        model_name, freeze_layers, 256,
                        seed, jitter,
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOP_PATIENCE_VALS:
                        print(f"\n  Early stopping at val-step {total_val_steps} "
                              f"(best combined={best_val_corr:.4f}).")
                        early_stop_triggered = True
                        break

                model.train()

        elapsed = time.time() - t0
        print(f"\n[Epoch {epoch:3d}] {elapsed:.1f}s")

    print("\n" + "=" * 100)
    print(f"Training complete. Best val combined = {best_val_corr:.4f}")
    print("=" * 100)
    print_init_summary(model)

    if test_agg:
        ckpt_path = os.path.join(save_dir, "best_model.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"\nLoaded best checkpoint (epoch={ckpt['epoch']}, "
                  f"val_step={ckpt['val_step']}) for test eval.")
        test = evaluate_detailed(model, test_agg, device, subtlex)
        print(f"\nTest set results:")
        print(f"  r_TRT={test['r_trt']:.3f}  r_FFD={test['r_ffd']:.3f}  "
              f"r_Gaze={test['r_gaze']:.3f}  r_skip={test['r_skip']:.3f}")
        print(f"  combined = {combined_metric(test):.4f}")
        print(f"  Bias_TRT={test['bias_trt']:+.1f}ms  Bias_FFD={test['bias_ffd']:+.1f}ms  "
              f"mean_skip={test['mean_skip']:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--freeze", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lm_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--cog_lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jitter", type=float, default=DEFAULT_JITTER,
                        help="Multiplicative range half-width for cog scalar init.")
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
        "hybrid_v4c_v2_randinit",
        f"geco_{model_short}_seed{args.seed}",
    )
    log_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "logs",
        f"train_hybrid_v4c_v2_randinit_geco_seed{args.seed}.log",
    )

    train(
        data_dir=data_dir, num_epochs=args.epochs,
        lm_lr=args.lm_lr, head_lr=args.head_lr, cog_lr=args.cog_lr,
        batch_size=args.batch_size, accumulation_steps=args.accum,
        save_dir=save_dir, log_path=log_path,
        seed=args.seed, jitter=args.jitter,
        model_name=args.model, freeze_layers=freeze_layers,
    )
