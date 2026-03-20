"""
Training script for the Sequential EZ Reader (LLaMA backbone) on Provo/GECO.

Same training loop as train_sequential.py but adapted for LLaMA:
  - Differential LR for llama.* vs head params
  - Lower LM learning rate (LLaMA is bigger, needs gentler fine-tuning)
  - Smaller batch size default (LLaMA uses more VRAM)

Usage:
    python src_v3/train_sequential_llama.py --corpus provo
    python src_v3/train_sequential_llama.py --corpus geco --batch_size 2 --accum 8
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
from sequential_reader_llama import SequentialEZReaderLLaMA


# --------------------------------------------------------------------------- #
#  Hyperparameters
# --------------------------------------------------------------------------- #

DURATION_WEIGHT = 1.0
SACCADE_WEIGHT = 1.0
STOP_WEIGHT = 0.5
L1_REG_WEIGHT = 0.01
L1_REG_MAX = 200.0


# --------------------------------------------------------------------------- #
#  Word frequency lookup (SUBTLEXus)
# --------------------------------------------------------------------------- #

_SUBTLEX = None

def load_subtlexus(path):
    """Load SUBTLEXus into a dict: word -> Lg10WF."""
    global _SUBTLEX
    if _SUBTLEX is not None:
        return _SUBTLEX
    _SUBTLEX = {}
    with open(path, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 7:
                word = parts[0].lower()
                lg10wf = float(parts[6])  # Lg10WF column
                _SUBTLEX[word] = lg10wf
    print(f"  SUBTLEXus: {len(_SUBTLEX):,} entries")
    return _SUBTLEX


def get_log_freq(word, subtlex):
    """Look up log10 word frequency. Returns 0.0 for unknown words (very rare)."""
    w = word.lower().strip(".,!?;:\"'()-")
    if w in subtlex:
        return subtlex[w]
    return 0.0


def get_word_log_freqs(tokens, subtlex):
    """Get log10 frequencies for a list of tokens."""
    return [get_log_freq(t, subtlex) for t in tokens]


# --------------------------------------------------------------------------- #
#  Collate scanpaths into batched tensors
# --------------------------------------------------------------------------- #

def collate_scanpaths(batch, device, subtlex=None):
    word_lists = [sp.tokens for sp in batch]

    word_lengths = pad_sequence(
        [torch.tensor(sp.word_lengths, dtype=torch.float32) for sp in batch],
        batch_first=True,
    ).to(device)

    # Word log frequencies from SUBTLEXus
    if subtlex is not None:
        word_log_freqs = pad_sequence(
            [torch.tensor(get_word_log_freqs(sp.tokens, subtlex), dtype=torch.float32) for sp in batch],
            batch_first=True,
        ).to(device)
    else:
        word_log_freqs = torch.zeros_like(word_lengths)

    max_fix = max(len(sp.fixations) for sp in batch)
    B = len(batch)

    fix_positions = torch.zeros(B, max_fix, dtype=torch.long, device=device)
    fix_durations = torch.zeros(B, max_fix, dtype=torch.float32, device=device)
    fix_mask = torch.zeros(B, max_fix, dtype=torch.float32, device=device)
    saccade_targets = torch.zeros(B, max_fix, dtype=torch.long, device=device)
    saccade_mask = torch.zeros(B, max_fix, dtype=torch.float32, device=device)

    stop_targets = torch.zeros(B, max_fix, dtype=torch.float32, device=device)

    for i, sp in enumerate(batch):
        n = len(sp.fixations)
        for j, fix in enumerate(sp.fixations):
            fix_positions[i, j] = fix.word_index
            fix_durations[i, j] = fix.duration
            fix_mask[i, j] = 1.0
            if j < n - 1:
                saccade_targets[i, j] = sp.fixations[j + 1].word_index
                saccade_mask[i, j] = 1.0
            else:
                # Last fixation: stop target = 1
                stop_targets[i, j] = 1.0

    return {
        'word_lists': word_lists,
        'word_lengths': word_lengths,
        'word_log_freqs': word_log_freqs,
        'fix_positions': fix_positions,
        'fix_durations': fix_durations,
        'fix_mask': fix_mask,
        'saccade_targets': saccade_targets,
        'saccade_mask': saccade_mask,
        'stop_targets': stop_targets,
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

def evaluate(model, data, device, subtlex=None, batch_size=1):
    model.eval()

    agg = aggregate_scanpaths(data, min_participants=2)
    if not agg:
        return {'r_ffd': 0.0, 'r_trt': 0.0, 'n_sentences': 0}

    all_pred_ffd = []
    all_human_ffd = []
    all_pred_trt = []
    all_human_trt = []
    all_pred_skip = []
    all_human_skip = []
    all_n_fixations = []
    all_human_n_fixations = []
    all_pred_ffd_all = []  # includes zeros for skipped words
    all_saccade_lengths = []
    all_regression_count = []

    with torch.no_grad():
        for sent in agg:
            word_lists = [sent['tokens']]
            wl = torch.tensor(
                [sent['word_lengths']], dtype=torch.float32, device=device
            )
            if subtlex is not None:
                wlf = torch.tensor(
                    [get_word_log_freqs(sent['tokens'], subtlex)],
                    dtype=torch.float32, device=device
                )
            else:
                wlf = torch.zeros_like(wl)

            result = model.forward_free(word_lists, wl, wlf)
            n_words = len(sent['tokens'])

            # Scanpath analysis
            if result['scanpath_positions'] is not None:
                positions = result['scanpath_positions'][0].cpu().tolist()
                all_n_fixations.append(len(positions))
                # Saccade lengths and regressions
                n_regressions = 0
                for k in range(1, len(positions)):
                    sac_len = positions[k] - positions[k-1]
                    all_saccade_lengths.append(sac_len)
                    if sac_len < 0:
                        n_regressions += 1
                all_regression_count.append(n_regressions)

            # Human fixation count for this sentence
            all_human_n_fixations.append(sent.get('mean_n_fixations', 0))

            for i in range(n_words):
                h_ffd = sent['mean_ffd'][i]
                h_trt = sent['mean_trt'][i]
                h_skip = sent.get('skip_rate', [None] * n_words)[i]
                p_ffd = result['first_fixation'][0, i].item()
                p_trt = result['total_reading_time'][0, i].item()
                p_skip = result['skip_prob'][0, i].item()

                all_pred_ffd_all.append(p_ffd)

                if h_ffd > 0:
                    all_pred_ffd.append(p_ffd)
                    all_human_ffd.append(h_ffd)
                if h_trt > 0:
                    all_pred_trt.append(p_trt)
                    all_human_trt.append(h_trt)
                if h_skip is not None:
                    all_pred_skip.append(p_skip)
                    all_human_skip.append(h_skip)

    def corr(a, b):
        a, b = np.array(a), np.array(b)
        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            return np.corrcoef(a, b)[0, 1]
        return 0.0

    r_ffd = corr(all_pred_ffd, all_human_ffd)
    r_trt = corr(all_pred_trt, all_human_trt)
    r_skip = corr(all_pred_skip, all_human_skip)

    pred_ffd_arr = np.array(all_pred_ffd) if all_pred_ffd else np.array([0])
    human_ffd_arr = np.array(all_human_ffd) if all_human_ffd else np.array([0])
    pred_skip_arr = np.array(all_pred_skip) if all_pred_skip else np.array([0])
    human_skip_arr = np.array(all_human_skip) if all_human_skip else np.array([0])
    sac_arr = np.array(all_saccade_lengths) if all_saccade_lengths else np.array([0])

    return {
        'r_ffd': r_ffd,
        'r_trt': r_trt,
        'r_skip': r_skip,
        'n_words_ffd': len(all_pred_ffd),
        'n_words_trt': len(all_pred_trt),
        'mean_pred_ffd': float(np.mean(pred_ffd_arr)),
        'mean_human_ffd': float(np.mean(human_ffd_arr)),
        'std_pred_ffd': float(np.std(pred_ffd_arr)),
        'std_human_ffd': float(np.std(human_ffd_arr)),
        'mean_pred_trt': float(np.mean(all_pred_trt)) if all_pred_trt else 0,
        'mean_human_trt': float(np.mean(all_human_trt)) if all_human_trt else 0,
        'mean_n_fixations': float(np.mean(all_n_fixations)) if all_n_fixations else 0,
        'pred_skip_rate': float(np.mean(pred_skip_arr)),
        'human_skip_rate': float(np.mean(human_skip_arr)),
        'mean_saccade_length': float(np.mean(sac_arr)),
        'std_saccade_length': float(np.std(sac_arr)),
        'regression_rate': float(np.mean([r / max(n-1, 1) for r, n in zip(all_regression_count, all_n_fixations)])) if all_regression_count else 0,
        'n_sentences': len(agg),
    }


def evaluate_teacher_forcing(model, data, device, subtlex=None, batch_size=4):
    model.eval()
    total_dur_loss = 0.0
    total_sac_loss = 0.0
    n = 0

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            collated = collate_scanpaths(batch, device, subtlex=subtlex)

            result = model.forward_teacher_forcing(
                collated['word_lists'],
                collated['word_lengths'],
                collated['word_log_freqs'],
                collated['fix_positions'],
                collated['fix_durations'],
                collated['fix_mask'],
                collated['saccade_targets'],
                collated['saccade_mask'],
                collated['stop_targets'],
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


def print_sample_scanpath(model, data, device, subtlex=None):
    model.eval()
    sp = data[0]
    word_lists = [sp.tokens]
    wl = torch.tensor([sp.word_lengths], dtype=torch.float32, device=device)
    if subtlex is not None:
        wlf = torch.tensor(
            [get_word_log_freqs(sp.tokens, subtlex)],
            dtype=torch.float32, device=device
        )
    else:
        wlf = torch.zeros_like(wl)

    with torch.no_grad():
        result = model.forward_free(word_lists, wl, wlf)

    n_words = len(sp.tokens)
    print(f"  Sentence ({n_words} words): {' '.join(sp.tokens[:12])}{'...' if n_words > 12 else ''}")

    # Human scanpath
    print(f"  Human scanpath ({len(sp.fixations)} fixations):")
    for i, fix in enumerate(sp.fixations[:10]):
        word = sp.tokens[fix.word_index] if fix.word_index < n_words else "?"
        freq = get_log_freq(word, subtlex) if subtlex else 0
        print(f"    {i+1}. [{fix.word_index:2d}] '{word:15s}' {fix.duration:4.0f}ms  (len={len(word)}, freq={freq:.1f})")

    # Model scanpath
    if result['scanpath_positions'] is not None:
        positions = result['scanpath_positions'][0].cpu().tolist()
        durations = result['scanpath_durations'][0].cpu().tolist()
        n_steps = len(positions)
        n_show = min(n_steps, 10)
        # Count regressions
        regressions = sum(1 for k in range(1, n_steps) if positions[k] < positions[k-1])
        sac_lengths = [positions[k] - positions[k-1] for k in range(1, n_steps)]
        mean_sac = np.mean(sac_lengths) if sac_lengths else 0

        print(f"  Model scanpath ({n_steps} fixations, {regressions} regressions, "
              f"mean saccade={mean_sac:.1f} words):")
        for i in range(n_show):
            idx = int(positions[i])
            word = sp.tokens[idx] if idx < n_words else "?"
            freq = get_log_freq(word, subtlex) if subtlex else 0
            sac_str = ""
            if i > 0:
                sac_len = positions[i] - positions[i-1]
                sac_str = f"  sac={sac_len:+d}"
            print(f"    {i+1}. [{idx:2d}] '{word:15s}' {durations[i]:4.0f}ms  (len={len(word)}, freq={freq:.1f}){sac_str}")

    # Per-word comparison table
    print(f"  Per-word comparison (first 12 words):")
    print(f"    {'word':15s}  {'len':>3s} {'freq':>4s}  {'h_FFD':>5s} {'p_FFD':>5s}  {'h_TRT':>5s} {'p_TRT':>5s}  {'h_skip':>6s} {'p_skip':>6s}")
    for i in range(min(n_words, 12)):
        word = sp.tokens[i]
        wlen = len(word)
        freq = get_log_freq(word, subtlex) if subtlex else 0
        h_ffd = sp.word_ffd[i] if sp.word_ffd[i] > 0 else 0
        p_ffd = result['first_fixation'][0, i].item()
        h_trt = sp.word_trt[i] if sp.word_trt[i] > 0 else 0
        p_trt = result['total_reading_time'][0, i].item()
        h_skip = 1 if sp.word_skipped[i] else 0
        p_skip = result['skip_prob'][0, i].item()
        print(f"    {word:15s}  {wlen:3d} {freq:4.1f}  {h_ffd:5.0f} {p_ffd:5.0f}  {h_trt:5.0f} {p_trt:5.0f}  {h_skip:6.0f} {p_skip:6.2f}")

    # Duration stats
    pred_durs = [durations[i] for i in range(len(positions))] if result['scanpath_positions'] is not None else []
    if pred_durs:
        print(f"  Duration stats: min={min(pred_durs):.0f} max={max(pred_durs):.0f} "
              f"std={np.std(pred_durs):.0f} range={max(pred_durs)-min(pred_durs):.0f}ms")
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

    fix_counts = [len(sp.fixations) for sp in dataset]
    print(f"  Fixations per scanpath: mean={np.mean(fix_counts):.1f}, max={max(fix_counts)}")

    # ---- Load SUBTLEXus ----
    subtlex_path = os.path.join(data_dir, "SUBTLEXus.txt")
    print("Loading SUBTLEXus...")
    subtlex = load_subtlexus(subtlex_path)

    # ---- Model ----
    print(f"\nLoading model: {args.llama}")

    model = SequentialEZReaderLLaMA(
        model_name=args.llama,
        hidden_dim=args.hidden_dim,
        sigma_init=args.sigma,
        freeze_layers=args.freeze,
    ).to(device)

    # Parameter groups: llama, scale params (need higher LR), everything else
    scale_param_names = {'l1_scale', 'l2_scale', 'eccentricity', 'l2_contribution', 'log_sigma'}
    llama_params = []
    scale_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("llama."):
            llama_params.append(param)
        elif name in scale_param_names:
            scale_params.append((name, param))
        else:
            head_params.append(param)

    n_total = sum(p.numel() for p in model.parameters())
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_llama = sum(p.numel() for p in llama_params)
    n_head = sum(p.numel() for p in head_params)

    scale_lr = args.head_lr * 10  # 5e-3: much higher LR for scalar EZR params
    print(f"  Total params: {n_total:,}")
    print(f"  Frozen: {n_frozen:,}")
    print(f"  Trainable LLaMA: {n_llama:,} (lr={args.llama_lr})")
    print(f"  Trainable heads: {n_head:,} (lr={args.head_lr})")
    print(f"  Scale params: {[n for n, _ in scale_params]} (lr={scale_lr})")
    print(f"  Freeze layers: {args.freeze}")

    # ---- Optimizer ----
    optimizer = optim.AdamW([
        {"params": llama_params, "lr": args.llama_lr, "weight_decay": 0.01},
        {"params": head_params, "lr": args.head_lr, "weight_decay": 0.0},
        {"params": [p for _, p in scale_params], "lr": scale_lr, "weight_decay": 0.0},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-7
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- Save dir ----
    model_short = args.llama.replace("/", "_")
    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "checkpoints", "v3", f"{args.corpus}_llama_{model_short}"
    )
    os.makedirs(save_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_dir, "training_log.txt"))

    # ---- Training loop ----
    best_val_metric = -1.0

    print("\n" + "=" * 90)
    print(f"Training Sequential EZ Reader (LLaMA) on {args.corpus.upper()}")
    print(f"  Batch size: {args.batch_size} | Accum steps: {args.accum}")
    print(f"  Effective batch: {args.batch_size * args.accum}")
    print(f"  Loss weights: dur={DURATION_WEIGHT}, sac={SACCADE_WEIGHT}, stop={STOP_WEIGHT}")
    print(f"  Visual span sigma init: {args.sigma}")
    print("=" * 90)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        epoch_data = train_data.copy()
        random.shuffle(epoch_data)

        epoch_dur_loss = 0.0
        epoch_sac_loss = 0.0
        epoch_stop_loss = 0.0
        epoch_total_loss = 0.0
        n_samples = 0

        optimizer.zero_grad()
        n_batches = (len(epoch_data) + args.batch_size - 1) // args.batch_size

        for step in range(n_batches):
            batch = epoch_data[step * args.batch_size : (step + 1) * args.batch_size]
            collated = collate_scanpaths(batch, device, subtlex=subtlex)

            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model.forward_teacher_forcing(
                    collated['word_lists'],
                    collated['word_lengths'],
                    collated['word_log_freqs'],
                    collated['fix_positions'],
                    collated['fix_durations'],
                    collated['fix_mask'],
                    collated['saccade_targets'],
                    collated['saccade_mask'],
                    collated['stop_targets'],
                )

            dur_loss = result['duration_loss']
            sac_loss = result['saccade_loss']
            stop_loss = result['stop_loss']

            # L1 regularization
            l1_vals = result['L1']
            l1_mask = collated['fix_mask']
            l1_excess = torch.nn.functional.relu(l1_vals - L1_REG_MAX) * l1_mask
            l1_reg = L1_REG_WEIGHT * l1_excess.sum() / l1_mask.sum().clamp(min=1)

            loss = (DURATION_WEIGHT * dur_loss
                    + SACCADE_WEIGHT * sac_loss + STOP_WEIGHT * stop_loss + l1_reg)
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
            epoch_stop_loss += stop_loss.item()
            epoch_total_loss += loss.item() * args.accum
            n_samples += len(batch)

        n_b = max(n_batches, 1)
        epoch_dur_loss /= n_b
        epoch_sac_loss /= n_b
        epoch_stop_loss /= n_b
        epoch_total_loss /= n_b
        elapsed = time.time() - t0

        # ---- Validation ----
        val_tf = evaluate_teacher_forcing(model, val_data, device, subtlex=subtlex, batch_size=args.batch_size)
        val_free = evaluate(model, val_data, device, subtlex=subtlex, batch_size=1)

        val_metric = (val_free['r_ffd'] + val_free['r_trt']) / 2.0
        scheduler.step(val_tf['duration_loss'])

        is_best = val_metric > best_val_metric

        print(f"\n[Epoch {epoch:3d}/{args.epochs}] {elapsed:.1f}s")
        print(f"  Train: dur={epoch_dur_loss:.3f}  sac={epoch_sac_loss:.3f}  "
              f"stop={epoch_stop_loss:.3f}  total={epoch_total_loss:.2f}  ({n_samples:,} scanpaths)")
        print(f"  Val TF: dur_loss={val_tf['duration_loss']:.1f}  sac_loss={val_tf['saccade_loss']:.3f}")
        print(f"  Val Free: r_FFD={val_free['r_ffd']:.3f}  r_TRT={val_free['r_trt']:.3f}  "
              f"r_Skip={val_free['r_skip']:.3f}  "
              f"({val_free['n_sentences']} sentences, {val_free['n_words_ffd']} words)")
        print(f"    FFD:  pred={val_free['mean_pred_ffd']:.0f}ms (std={val_free['std_pred_ffd']:.0f})  "
              f"human={val_free['mean_human_ffd']:.0f}ms (std={val_free['std_human_ffd']:.0f})")
        print(f"    TRT:  pred={val_free['mean_pred_trt']:.0f}ms  human={val_free['mean_human_trt']:.0f}ms")
        print(f"    Skip: pred={val_free['pred_skip_rate']:.1%}  human={val_free['human_skip_rate']:.1%}")
        print(f"    Fixations: {val_free['mean_n_fixations']:.1f}/sentence  "
              f"saccade={val_free['mean_saccade_length']:.1f} words (std={val_free['std_saccade_length']:.1f})  "
              f"regressions={val_free['regression_rate']:.1%}")

        print_model_params(model)

        if epoch % 5 == 1 or is_best:
            print_sample_scanpath(model, val_data, device, subtlex=subtlex)

        if is_best:
            best_val_metric = val_metric
            print(f"  ** NEW BEST (avg={val_metric:.3f}, r_FFD={val_free['r_ffd']:.3f}, r_TRT={val_free['r_trt']:.3f}) **")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'llama_model_name': args.llama,
                'hidden_dim': args.hidden_dim,
                'freeze_layers': args.freeze,
                'sigma_init': args.sigma,
                'val_metrics': val_free,
            }, os.path.join(save_dir, "best_model.pt"))

    # ---- Final test ----
    print("\n" + "=" * 90)
    print(f"Training complete! Best val r_FFD = {best_val_metric:.3f}")
    print("=" * 90)

    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_free = evaluate(model, test_data, device, subtlex=subtlex, batch_size=1)
    print(f"\nTest results (free-running):")
    print(f"  r_FFD={test_free['r_ffd']:.3f}  r_TRT={test_free['r_trt']:.3f}  r_Skip={test_free['r_skip']:.3f}")
    print(f"    FFD:  pred={test_free['mean_pred_ffd']:.0f}ms (std={test_free['std_pred_ffd']:.0f})  "
          f"human={test_free['mean_human_ffd']:.0f}ms (std={test_free['std_human_ffd']:.0f})")
    print(f"    TRT:  pred={test_free['mean_pred_trt']:.0f}ms  human={test_free['mean_human_trt']:.0f}ms")
    print(f"    Skip: pred={test_free['pred_skip_rate']:.1%}  human={test_free['human_skip_rate']:.1%}")
    print(f"    Fixations: {test_free['mean_n_fixations']:.1f}/sentence  "
          f"saccade={test_free['mean_saccade_length']:.1f} words (std={test_free['std_saccade_length']:.1f})  "
          f"regressions={test_free['regression_rate']:.1%}")

    print("\nFinal parameters:")
    print_model_params(model)

    print("\nSample test scanpath:")
    print_sample_scanpath(model, test_data, device, subtlex=subtlex)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sequential EZ Reader (LLaMA)")
    parser.add_argument("--corpus", type=str, default="provo", choices=["provo", "geco"])
    parser.add_argument("--llama", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="LLaMA model name")
    parser.add_argument("--freeze", type=int, default=14,
                        help="LLaMA layers to freeze (default: 14 of 22)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=4.0,
                        help="Initial visual span sigma (in words)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--llama_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)
