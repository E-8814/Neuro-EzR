"""
Training script for the Sequential EZ Reader on Provo corpus.

Teacher forcing on human scanpaths: at each fixation step, the model
sees where the human actually looked (not its own prediction), and learns
to predict (a) how long to fixate and (b) where to look next.

Usage:
    python src_v3/train_sequential.py
    python src_v3/train_sequential.py --bert prajjwal1/bert-mini --epochs 30
    python src_v3/train_sequential.py --corpus geco
"""

import os
import sys
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.dirname(__file__))

from scanpath_loader import (
    load_provo_scanpaths, load_geco_scanpaths,
    split_scanpaths, aggregate_scanpaths,
)
from sequential_reader import SequentialEZReader


# --------------------------------------------------------------------------- #
#  Hyperparameters
# --------------------------------------------------------------------------- #

DURATION_WEIGHT = 1.0
SACCADE_WEIGHT = 1.0
L1_REG_WEIGHT = 0.01       # penalize L1 > 200ms
L1_REG_MAX = 200.0


# --------------------------------------------------------------------------- #
#  Collate scanpaths into batched tensors
# --------------------------------------------------------------------------- #

def collate_scanpaths(batch, device):
    """
    Pad a list of ScanpathData into batched tensors.

    Returns dict with:
        word_lists:      list of list of str
        word_lengths:    (B, T) float
        fix_positions:   (B, max_fix) long
        fix_durations:   (B, max_fix) float
        fix_mask:        (B, max_fix) float
        saccade_targets: (B, max_fix) long
        saccade_mask:    (B, max_fix) float
    """
    word_lists = [sp.tokens for sp in batch]

    # Pad word lengths
    word_lengths = pad_sequence(
        [torch.tensor(sp.word_lengths, dtype=torch.float32) for sp in batch],
        batch_first=True,
    ).to(device)

    # Pad fixation sequences
    max_fix = max(len(sp.fixations) for sp in batch)
    B = len(batch)

    fix_positions = torch.zeros(B, max_fix, dtype=torch.long, device=device)
    fix_durations = torch.zeros(B, max_fix, dtype=torch.float32, device=device)
    fix_mask = torch.zeros(B, max_fix, dtype=torch.float32, device=device)
    saccade_targets = torch.zeros(B, max_fix, dtype=torch.long, device=device)
    saccade_mask = torch.zeros(B, max_fix, dtype=torch.float32, device=device)

    for i, sp in enumerate(batch):
        n = len(sp.fixations)
        for j, fix in enumerate(sp.fixations):
            fix_positions[i, j] = fix.word_index
            fix_durations[i, j] = fix.duration
            fix_mask[i, j] = 1.0
            # Saccade target: where the NEXT fixation lands
            if j < n - 1:
                saccade_targets[i, j] = sp.fixations[j + 1].word_index
                saccade_mask[i, j] = 1.0

    return {
        'word_lists': word_lists,
        'word_lengths': word_lengths,
        'fix_positions': fix_positions,
        'fix_durations': fix_durations,
        'fix_mask': fix_mask,
        'saccade_targets': saccade_targets,
        'saccade_mask': saccade_mask,
    }


# --------------------------------------------------------------------------- #
#  Logger
# --------------------------------------------------------------------------- #

class Logger:
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
#  Evaluation
# --------------------------------------------------------------------------- #

def evaluate(model, data, device, batch_size=8):
    """
    Evaluate model in free-running mode.
    Returns per-word correlations with human aggregated data.
    """
    model.eval()

    # Aggregate human data
    agg = aggregate_scanpaths(data, min_participants=2)
    if not agg:
        return {'r_ffd': 0.0, 'r_trt': 0.0, 'n_sentences': 0}

    all_pred_ffd = []
    all_human_ffd = []
    all_pred_trt = []
    all_human_trt = []

    with torch.no_grad():
        for sent in agg:
            word_lists = [sent['tokens']]
            wl = torch.tensor(
                [sent['word_lengths']], dtype=torch.float32, device=device
            )

            result = model.forward_free(word_lists, wl)
            n_words = len(sent['tokens'])

            for i in range(n_words):
                h_ffd = sent['mean_ffd'][i]
                h_trt = sent['mean_trt'][i]
                p_ffd = result['first_fixation'][0, i].item()
                p_trt = result['total_reading_time'][0, i].item()

                if h_ffd > 0:
                    all_pred_ffd.append(p_ffd)
                    all_human_ffd.append(h_ffd)
                if h_trt > 0:
                    all_pred_trt.append(p_trt)
                    all_human_trt.append(h_trt)

    def corr(a, b):
        a, b = np.array(a), np.array(b)
        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            return np.corrcoef(a, b)[0, 1]
        return 0.0

    return {
        'r_ffd': corr(all_pred_ffd, all_human_ffd),
        'r_trt': corr(all_pred_trt, all_human_trt),
        'n_words_ffd': len(all_pred_ffd),
        'n_words_trt': len(all_pred_trt),
        'mean_pred_ffd': np.mean(all_pred_ffd) if all_pred_ffd else 0,
        'mean_human_ffd': np.mean(all_human_ffd) if all_human_ffd else 0,
        'mean_pred_trt': np.mean(all_pred_trt) if all_pred_trt else 0,
        'mean_human_trt': np.mean(all_human_trt) if all_human_trt else 0,
        'n_sentences': len(agg),
    }


def evaluate_teacher_forcing(model, data, device, batch_size=8):
    """Evaluate with teacher forcing (training-mode metrics)."""
    model.eval()
    total_dur_loss = 0.0
    total_sac_loss = 0.0
    n = 0

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            collated = collate_scanpaths(batch, device)

            result = model.forward_teacher_forcing(
                collated['word_lists'],
                collated['word_lengths'],
                collated['fix_positions'],
                collated['fix_durations'],
                collated['fix_mask'],
                collated['saccade_targets'],
                collated['saccade_mask'],
            )

            total_dur_loss += result['duration_loss'].item() * len(batch)
            total_sac_loss += result['saccade_loss'].item() * len(batch)
            n += len(batch)

    return {
        'duration_loss': total_dur_loss / max(n, 1),
        'saccade_loss': total_sac_loss / max(n, 1),
    }


# --------------------------------------------------------------------------- #
#  Print model state
# --------------------------------------------------------------------------- #

def print_model_params(model):
    sigma = model.log_sigma.exp().item()
    print(f"  EZ Reader params:")
    print(f"    l1_scale={model.l1_scale.item():.1f}  l2_scale={model.l2_scale.item():.1f}")
    print(f"    eccentricity={model.eccentricity.item():.4f}")
    print(f"    l2_contribution={model.l2_contribution.item():.4f} "
          f"(effective={torch.nn.functional.softplus(model.l2_contribution).item():.4f})")
    print(f"    visual_span_sigma={sigma:.2f} words")


def print_sample_scanpath(model, data, device):
    """Run free-running inference on one sentence and show the scanpath."""
    model.eval()
    sp = data[0]
    word_lists = [sp.tokens]
    wl = torch.tensor([sp.word_lengths], dtype=torch.float32, device=device)

    with torch.no_grad():
        result = model.forward_free(word_lists, wl)

    print(f"  Sentence: {' '.join(sp.tokens[:8])}{'...' if len(sp.tokens) > 8 else ''}")

    # Human scanpath
    print(f"  Human scanpath ({len(sp.fixations)} fixations):")
    for i, fix in enumerate(sp.fixations[:8]):
        word = sp.tokens[fix.word_index] if fix.word_index < len(sp.tokens) else "?"
        print(f"    {i+1}. [{fix.word_index}] '{word}' {fix.duration:.0f}ms")

    # Model scanpath
    if result['scanpath_positions'] is not None:
        positions = result['scanpath_positions'][0].cpu().tolist()
        durations = result['scanpath_durations'][0].cpu().tolist()
        n_steps = min(len(positions), 8)
        print(f"  Model scanpath ({len(positions)} fixations):")
        for i in range(n_steps):
            idx = int(positions[i])
            word = sp.tokens[idx] if idx < len(sp.tokens) else "?"
            print(f"    {i+1}. [{idx}] '{word}' {durations[i]:.0f}ms")
    print()


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load data ----
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    if args.corpus == "provo":
        et_path = os.path.join(data_dir, "Provo_Corpus-Eyetracking_Data.csv")
        print("Loading Provo scanpaths...")
        dataset = load_provo_scanpaths(et_path)
    elif args.corpus == "geco":
        reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
        material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
        pred_path = os.path.join(data_dir, "geco_predictability.pkl")
        print("Loading GECO scanpaths...")
        dataset = load_geco_scanpaths(reading_path, material_path, pred_path)
    else:
        raise ValueError(f"Unknown corpus: {args.corpus}")

    print(f"  Total scanpaths: {len(dataset):,}")

    train_data, val_data, test_data = split_scanpaths(dataset)
    print(f"  Split: train={len(train_data):,}  val={len(val_data):,}  test={len(test_data):,}")

    # Scanpath stats
    fix_counts = [len(sp.fixations) for sp in dataset]
    print(f"  Fixations per scanpath: mean={np.mean(fix_counts):.1f}, max={max(fix_counts)}")

    # ---- Model ----
    print(f"\nLoading model: {args.bert}")

    # Auto-determine freeze layers
    if args.freeze is not None:
        freeze_layers = args.freeze
    else:
        from transformers import BertConfig
        cfg = BertConfig.from_pretrained(args.bert)
        freeze_layers = cfg.num_hidden_layers // 2

    model = SequentialEZReader(
        bert_model_name=args.bert,
        hidden_dim=args.hidden_dim,
        sigma_init=args.sigma,
        freeze_bert_layers=freeze_layers,
    ).to(device)

    # Parameter counts
    bert_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("bert."):
            bert_params.append(param)
        else:
            head_params.append(param)

    n_total = sum(p.numel() for p in model.parameters())
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_bert = sum(p.numel() for p in bert_params)
    n_head = sum(p.numel() for p in head_params)

    print(f"  Total params: {n_total:,}")
    print(f"  Frozen: {n_frozen:,}")
    print(f"  Trainable BERT: {n_bert:,} (lr={args.bert_lr})")
    print(f"  Trainable heads: {n_head:,} (lr={args.head_lr})")
    print(f"  Freeze layers: {freeze_layers}")

    # ---- Optimizer ----
    optimizer = optim.AdamW([
        {"params": bert_params, "lr": args.bert_lr, "weight_decay": 0.01},
        {"params": head_params, "lr": args.head_lr, "weight_decay": 0.0},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-7
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- Save dir ----
    model_short = args.bert.replace("/", "_")
    save_dir = os.path.join(
        os.path.dirname(__file__), "..",
        f"checkpoints_v3/{args.corpus}_{model_short}"
    )
    os.makedirs(save_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_dir, "training_log.txt"))

    # ---- Training loop ----
    best_val_metric = -1.0

    print("\n" + "=" * 90)
    print(f"Training Sequential EZ Reader on {args.corpus.upper()}")
    print(f"  Batch size: {args.batch_size} | Accum steps: {args.accum}")
    print(f"  Effective batch: {args.batch_size * args.accum}")
    print(f"  Loss weights: duration={DURATION_WEIGHT}, saccade={SACCADE_WEIGHT}")
    print(f"  Visual span sigma init: {args.sigma}")
    print("=" * 90)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        epoch_data = train_data.copy()
        random.shuffle(epoch_data)

        epoch_dur_loss = 0.0
        epoch_sac_loss = 0.0
        epoch_total_loss = 0.0
        n_samples = 0

        optimizer.zero_grad()
        n_batches = (len(epoch_data) + args.batch_size - 1) // args.batch_size

        for step in range(n_batches):
            batch = epoch_data[step * args.batch_size : (step + 1) * args.batch_size]
            collated = collate_scanpaths(batch, device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model.forward_teacher_forcing(
                    collated['word_lists'],
                    collated['word_lengths'],
                    collated['fix_positions'],
                    collated['fix_durations'],
                    collated['fix_mask'],
                    collated['saccade_targets'],
                    collated['saccade_mask'],
                )

            dur_loss = result['duration_loss']
            sac_loss = result['saccade_loss']

            # L1 regularization
            l1_vals = result['L1']
            l1_mask = collated['fix_mask']
            l1_excess = torch.nn.functional.relu(l1_vals - L1_REG_MAX) * l1_mask
            l1_reg = L1_REG_WEIGHT * l1_excess.sum() / l1_mask.sum().clamp(min=1)

            loss = DURATION_WEIGHT * dur_loss + SACCADE_WEIGHT * sac_loss + l1_reg
            loss = loss / args.accum

            scaler.scale(loss).backward()

            if (step + 1) % args.accum == 0 or (step + 1) == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_dur_loss += dur_loss.item()
            epoch_sac_loss += sac_loss.item()
            epoch_total_loss += loss.item() * args.accum
            n_samples += len(batch)

        n_b = max(n_batches, 1)
        epoch_dur_loss /= n_b
        epoch_sac_loss /= n_b
        epoch_total_loss /= n_b
        elapsed = time.time() - t0

        # ---- Validation ----
        val_tf = evaluate_teacher_forcing(model, val_data, device, batch_size=args.batch_size)
        val_free = evaluate(model, val_data, device, batch_size=1)

        val_metric = val_free['r_ffd']  # track FFD correlation
        scheduler.step(val_tf['duration_loss'])

        is_best = val_metric > best_val_metric

        print(f"\n[Epoch {epoch:3d}/{args.epochs}] {elapsed:.1f}s")
        print(f"  Train: dur_loss={epoch_dur_loss:.1f}  sac_loss={epoch_sac_loss:.3f}  "
              f"total={epoch_total_loss:.2f}  ({n_samples:,} scanpaths)")
        print(f"  Val TF: dur_loss={val_tf['duration_loss']:.1f}  sac_loss={val_tf['saccade_loss']:.3f}")
        print(f"  Val Free: r_FFD={val_free['r_ffd']:.3f}  r_TRT={val_free['r_trt']:.3f}  "
              f"({val_free['n_sentences']} sentences, {val_free['n_words_ffd']} words)")
        print(f"  Val Free: pred_FFD={val_free['mean_pred_ffd']:.0f}ms "
              f"(human={val_free['mean_human_ffd']:.0f}ms)  "
              f"pred_TRT={val_free['mean_pred_trt']:.0f}ms "
              f"(human={val_free['mean_human_trt']:.0f}ms)")

        print_model_params(model)

        if epoch % 5 == 1 or is_best:
            print_sample_scanpath(model, val_data, device)

        if is_best:
            best_val_metric = val_metric
            print(f"  ** NEW BEST (r_FFD={val_metric:.3f}) **")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'bert_model_name': args.bert,
                'hidden_dim': args.hidden_dim,
                'freeze_layers': freeze_layers,
                'sigma_init': args.sigma,
                'val_metrics': val_free,
            }, os.path.join(save_dir, "best_model.pt"))

    # ---- Final test ----
    print("\n" + "=" * 90)
    print(f"Training complete! Best val r_FFD = {best_val_metric:.3f}")
    print("=" * 90)

    # Load best model
    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_free = evaluate(model, test_data, device, batch_size=1)
    print(f"\nTest results (free-running):")
    print(f"  r_FFD={test_free['r_ffd']:.3f}  r_TRT={test_free['r_trt']:.3f}")
    print(f"  pred_FFD={test_free['mean_pred_ffd']:.0f}ms (human={test_free['mean_human_ffd']:.0f}ms)")
    print(f"  pred_TRT={test_free['mean_pred_trt']:.0f}ms (human={test_free['mean_human_trt']:.0f}ms)")

    print("\nFinal parameters:")
    print_model_params(model)

    print("\nSample test scanpath:")
    print_sample_scanpath(model, test_data, device)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sequential EZ Reader")
    parser.add_argument("--corpus", type=str, default="provo", choices=["provo", "geco"])
    parser.add_argument("--bert", type=str, default="prajjwal1/bert-small",
                        help="BERT model name")
    parser.add_argument("--freeze", type=int, default=None,
                        help="BERT layers to freeze (default: half)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=4.0,
                        help="Initial visual span sigma (in words)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--bert_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)
