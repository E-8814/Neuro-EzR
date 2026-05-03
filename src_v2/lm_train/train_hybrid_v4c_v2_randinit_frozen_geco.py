"""
Frozen-backbone parameter-recovery experiment.

Loads a pretrained dualctx model (specified by --pretrained_seed), freezes
TinyLlama + projection + ctx_head_FFD + ctx_head_skip, re-initializes the
9 cog scalars to ±jitter around Reichle 2003, and trains ONLY those scalars.

This is a sharper version of the v2_randinit experiment. The original
experiment let the LM compensate for bad cog inits (the LM has 1.1B
parameters of capacity), so cog scalars never moved. Here, with the LM
frozen, gradient descent on cog scalars alone has no escape route, and
must converge to whatever values fit the trained features best.

Usage:
    python train_hybrid_v4c_v2_randinit_frozen_geco.py \
        --seed 1 --pretrained_seed 1 --epochs 5 --jitter 0.5
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "lm_model"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "archive", "original_ezreader"))

from model_llama_hybrid_v4c_v2_dualctx import NeuralEZReaderHybrid
from model_llama_hybrid_v4c_v2_randinit import sample_init, REICHLE_INIT, DEFAULT_JITTER

# Reuse the heavy lifting from the original randinit script.
from train_hybrid_v4c_v2_randinit_geco import (
    load_subtlex, collate_sentences, compute_loss,
    evaluate_detailed, combined_metric, Logger,
    SKIP_BOUND_HALFWIDTH,
    EARLY_STOP_PATIENCE_VALS, WARMUP_EPOCHS,
)
from data_loader import aggregate_by_sentence
from geco_loader import load_geco, split_geco


# Override the imported value: we want finer-grained validation here.
N_VALS_PER_EPOCH = 10


def _reichle_targets():
    """Reichle 2003 reference values, including derived alphas."""
    base   = REICHLE_INIT["l1_base_offset"]
    freq   = REICHLE_INIT["l1_freq_coef"]
    a1R    = base - 2.0 * freq
    a2R    = -freq / 5.0
    eps    = 1.0 + REICHLE_INIT["epsilon_minus_1"]
    return {
        "a1R":   a1R,
        "a2R":   a2R,
        "eps":   eps,
        "M1":    REICHLE_INIT["M1"],
        "M2":    REICHLE_INIT["M2I"],
        "d":     REICHLE_INIT["delta"],
        "lref":  REICHLE_INIT["lambda_refix"],
        "rpiv":  REICHLE_INIT["refix_pivot"],
    }


def _diff_to_reichle(model):
    """Return signed % distance of each cog param from Reichle 2003."""
    R = _reichle_targets()
    cur = {
        "a1R":   model.alpha1_reichle.item(),
        "a2R":   model.alpha2_reichle.item(),
        "eps":   model.ezreader.epsilon.item(),
        "M1":    model.ezreader.M1.item(),
        "M2":    model.ezreader.M2.item(),
        "d":     model.delta.item(),
        "lref":  model.ezreader.lambda_refix.item(),
        "rpiv":  model.ezreader.refix_pivot.item(),
    }
    return {k: 100.0 * (cur[k] - R[k]) / R[k] for k in R}


def _format_diff_line(model):
    d = _diff_to_reichle(model)
    return ("Δ%toReichle: " +
            f"a1R={d['a1R']:+.1f} a2R={d['a2R']:+.1f} eps={d['eps']:+.1f} "
            f"M1={d['M1']:+.1f} M2={d['M2']:+.1f} d={d['d']:+.1f} "
            f"lref={d['lref']:+.1f}")


# Trainable-after-freeze parameter names (full dotted path).
COG_PARAM_NAMES = {
    "l1_base_offset",
    "l1_freq_coef",
    "_delta_raw",
    "ezreader._epsilon_raw",
    "ezreader._M1_raw",
    "ezreader._M2I_raw",
    "ezreader.lambda_refix",
    "ezreader.refix_pivot",
    "ezreader._skip_temperature_raw",
}


def _inv_softplus(y):
    return float(np.log(np.expm1(y)))


def _logit(p):
    return float(np.log(p / (1.0 - p)))


def reinit_cog_scalars(model, init_seed, jitter):
    """Overwrite the 9 cog scalars with values jittered around Reichle."""
    sampled = sample_init(init_seed=init_seed, jitter=jitter)
    with torch.no_grad():
        model.l1_base_offset.fill_(sampled["l1_base_offset"])
        model.l1_freq_coef.fill_(sampled["l1_freq_coef"])
        model._delta_raw.fill_(_logit(sampled["delta"]))
        model.ezreader._epsilon_raw.fill_(_inv_softplus(sampled["epsilon_minus_1"]))
        model.ezreader._M1_raw.fill_(_inv_softplus(sampled["M1"]))
        model.ezreader._M2I_raw.fill_(_inv_softplus(sampled["M2I"]))
        model.ezreader.lambda_refix.fill_(sampled["lambda_refix"])
        model.ezreader.refix_pivot.fill_(sampled["refix_pivot"])
        model.ezreader._skip_temperature_raw.fill_(
            _inv_softplus(sampled["skip_temperature_minus_1"])
        )
    return sampled


def freeze_non_cog(model):
    """Set requires_grad=False on every parameter except the 9 cog scalars."""
    n_train, n_freeze = 0, 0
    for name, p in model.named_parameters():
        if name in COG_PARAM_NAMES:
            p.requires_grad = True
            n_train += 1
        else:
            p.requires_grad = False
            n_freeze += 1
    return n_train, n_freeze


def report_cog_state(model, sampled_init, label):
    print(f"  [{label}]")
    rows = [
        ("l1_base_offset", model.l1_base_offset.item(), sampled_init["l1_base_offset"], REICHLE_INIT["l1_base_offset"]),
        ("l1_freq_coef",   model.l1_freq_coef.item(),   sampled_init["l1_freq_coef"],   REICHLE_INIT["l1_freq_coef"]),
        ("delta",          model.delta.item(),          sampled_init["delta"],          REICHLE_INIT["delta"]),
        ("epsilon",        model.ezreader.epsilon.item(), 1.0 + sampled_init["epsilon_minus_1"], 1.0 + REICHLE_INIT["epsilon_minus_1"]),
        ("M1",             model.ezreader.M1.item(),    sampled_init["M1"],             REICHLE_INIT["M1"]),
        ("M2",             model.ezreader.M2.item(),    sampled_init["M2I"],            REICHLE_INIT["M2I"]),
        ("lambda_refix",   model.ezreader.lambda_refix.item(), sampled_init["lambda_refix"], REICHLE_INIT["lambda_refix"]),
        ("refix_pivot",    model.ezreader.refix_pivot.item(),  sampled_init["refix_pivot"],  REICHLE_INIT["refix_pivot"]),
    ]
    print(f"    {'param':<20s} {'current':>10s} {'init':>10s} {'reichle':>10s}")
    for name, cur, init, reichle in rows:
        print(f"    {name:<20s} {cur:>10.4f} {init:>10.4f} {reichle:>10.4f}")


def save_best_checkpoint(model, save_dir, epoch, val_step, val,
                         model_name, hidden_dim,
                         init_seed, pretrained_seed, jitter, sampled_init):
    torch.save({
        "epoch": epoch, "val_step": val_step,
        "model_state_dict": model.state_dict(),
        "model_name": model_name, "hidden_dim": hidden_dim,
        "init_seed": init_seed,
        "pretrained_seed": pretrained_seed,
        "jitter": jitter,
        "sampled_init": sampled_init,
        "val_metrics": val,
        "cog_params": {
            "l1_base_offset":  model.l1_base_offset.item(),
            "l1_freq_coef":    model.l1_freq_coef.item(),
            "alpha1_reichle":  model.alpha1_reichle.item(),
            "alpha2_reichle":  model.alpha2_reichle.item(),
            "delta":           model.delta.item(),
            "epsilon":         model.ezreader.epsilon.item(),
            "M1":              model.ezreader.M1.item(),
            "M2":              model.ezreader.M2.item(),
            "lambda_refix":    model.ezreader.lambda_refix.item(),
            "refix_pivot":     model.ezreader.refix_pivot.item(),
            "skip_temperature": model.ezreader.skip_temperature.item(),
        },
        "reichle_init": REICHLE_INIT,
    }, os.path.join(save_dir, "best_model.pt"))


def train(
    pretrained_ckpt,
    seed,
    pretrained_seed,
    jitter=DEFAULT_JITTER,
    num_epochs=5,
    cog_lr=1e-3,         # only 9 params, but high LR overshoots — keep modest
    batch_size=8, accumulation_steps=8,
    save_dir="../../checkpoints/hybrid_v4c_v2_randinit_frozen/geco_tinyllama",
    log_path="../../logs/train_hybrid_v4c_v2_randinit_frozen_geco.log",
    data_dir="../data",
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    hidden_dim=256,
    freeze_layers=12,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Frozen-backbone recovery: training_seed={seed}  pretrained_seed={pretrained_seed}  jitter=±{jitter*100:.0f}%")

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

    print(f"\nBuilding dualctx model (model_name={model_name})...")
    model = NeuralEZReaderHybrid(
        model_name=model_name,
        freeze_layers=freeze_layers,
        hidden_dim=hidden_dim,
    ).to(device)

    print(f"Loading pretrained checkpoint: {pretrained_ckpt}")
    ckpt = torch.load(pretrained_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"\nRe-initializing cog scalars with jitter=±{jitter*100:.0f}% (seed={seed})...")
    sampled_init = reinit_cog_scalars(model, init_seed=seed, jitter=jitter)
    report_cog_state(model, sampled_init, "after re-init")

    n_train, n_freeze = freeze_non_cog(model)
    print(f"\nTrainable params: {n_train}  |  Frozen params: {n_freeze}")
    cog_params = [p for n, p in model.named_parameters() if n in COG_PARAM_NAMES]
    optimizer = optim.AdamW([
        {"params": cog_params, "lr": cog_lr, "weight_decay": 0.0},
    ])

    n_batches_per_epoch = (len(train_raw) + batch_size - 1) // batch_size
    optimizer_steps_per_epoch = (n_batches_per_epoch + accumulation_steps - 1) // accumulation_steps
    total_optimizer_steps = num_epochs * optimizer_steps_per_epoch
    warmup_steps = max(1, WARMUP_EPOCHS * optimizer_steps_per_epoch // 5)

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

    print("\n" + "=" * 100)
    print(f"Frozen-backbone recovery (jitter=±{jitter*100:.0f}%, "
          f"pretrained=seed{pretrained_seed}, training_seed={seed})")
    print("=" * 100)

    # Baseline validation (val 0): randomized cog + frozen pretrained backbone,
    # NO training yet. Establishes the starting point for the recovery curve.
    val0 = evaluate_detailed(model, val_agg, device, subtlex)
    combined0 = combined_metric(val0)
    print(f"  [val 0] BEFORE TRAINING (random cog × frozen pretrained backbone)")
    print(f"    combined={combined0:.4f} | r_TRT={val0['r_trt']:.3f} "
          f"r_FFD={val0['r_ffd']:.3f} r_Gaze={val0['r_gaze']:.3f} "
          f"r_skip={val0['r_skip']:.3f}")
    print(f"    a1R={model.alpha1_reichle.item():.1f} "
          f"a2R={model.alpha2_reichle.item():.3f} "
          f"eps={model.ezreader.epsilon.item():.3f} "
          f"M1={model.ezreader.M1.item():.1f} "
          f"M2=I={model.ezreader.M2.item():.1f} "
          f"d={model.delta.item():.3f} "
          f"lref={model.ezreader.lambda_refix.item():.3f}")
    print(f"    {_format_diff_line(model)}")
    model.train()

    best_val_corr = -1.0
    patience_counter = 0
    total_val_steps = 0
    early_stop = False
    val_every_n = max(1, n_batches_per_epoch // N_VALS_PER_EPOCH)

    for epoch in range(1, num_epochs + 1):
        if early_stop:
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
                torch.nn.utils.clip_grad_norm_(cog_params, max_norm=1.0)
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
                      f"r_FFD={val['r_ffd']:.3f} r_Gaze={val['r_gaze']:.3f} "
                      f"r_skip={val['r_skip']:.3f}")
                print(f"    a1R={model.alpha1_reichle.item():.1f} "
                      f"a2R={model.alpha2_reichle.item():.3f} "
                      f"eps={model.ezreader.epsilon.item():.3f} "
                      f"M1={model.ezreader.M1.item():.1f} "
                      f"M2=I={model.ezreader.M2.item():.1f} "
                      f"d={model.delta.item():.3f} "
                      f"lref={model.ezreader.lambda_refix.item():.3f}")
                print(f"    {_format_diff_line(model)}")

                if combined > best_val_corr:
                    best_val_corr = combined
                    patience_counter = 0
                    save_best_checkpoint(
                        model, save_dir, epoch, total_val_steps, val,
                        model_name, hidden_dim,
                        seed, pretrained_seed, jitter, sampled_init,
                    )
                    print(f"    ** NEW BEST (combined={combined:.4f}) **")
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOP_PATIENCE_VALS:
                        print(f"    [early stop] no improvement for "
                              f"{EARLY_STOP_PATIENCE_VALS} validations")
                        early_stop = True
                        break
                model.train()

        print(f"[Epoch {epoch:>3d}] {time.time()-t0:.1f}s")

    print("\n" + "=" * 100)
    print(f"Training complete. Best val combined = {best_val_corr:.4f}")
    print("=" * 100)
    report_cog_state(model, sampled_init, "final")

    ckpt_best = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(ckpt_best):
        bckpt = torch.load(ckpt_best, map_location=device, weights_only=False)
        model.load_state_dict(bckpt["model_state_dict"])
        print(f"Loaded best checkpoint (epoch {bckpt['epoch']}, val_step {bckpt['val_step']}) for test eval.")

    test = evaluate_detailed(model, test_agg, device, subtlex)
    print(f"\nTest set results:")
    print(f"  r_TRT={test['r_trt']:.3f}  r_FFD={test['r_ffd']:.3f}  "
          f"r_Gaze={test['r_gaze']:.3f}  r_skip={test['r_skip']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True,
                        help="training seed (drives cog re-init randomization)")
    parser.add_argument("--pretrained_seed", type=int, required=True,
                        help="which dualctx seed checkpoint to use as feature extractor")
    parser.add_argument("--jitter", type=float, default=DEFAULT_JITTER)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--cog_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--freeze", type=int, default=12)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override save_dir (default: checkpoints/hybrid_v4c_v2_randinit_frozen/...)")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Override training log path")
    args = parser.parse_args()

    model_short = args.model.replace("/", "_")
    pretrained_ckpt = os.path.join(
        _HERE, "..", "..", "checkpoints", "hybrid_v4c_v2_dualctx",
        f"geco_{model_short}_seed{args.pretrained_seed}", "best_model.pt",
    )
    pretrained_ckpt = os.path.abspath(pretrained_ckpt)
    if not os.path.isfile(pretrained_ckpt):
        raise FileNotFoundError(
            f"Pretrained dualctx checkpoint not found: {pretrained_ckpt}\n"
            f"Train exp01b first (or pass a different --pretrained_seed)."
        )

    save_dir = args.output_dir or os.path.join(
        _HERE, "..", "..", "checkpoints", "hybrid_v4c_v2_randinit_frozen",
        f"geco_{model_short}_seed{args.seed}",
    )
    log_path = args.log_path or os.path.join(
        _HERE, "..", "..", "logs",
        f"train_hybrid_v4c_v2_randinit_frozen_geco_seed{args.seed}.log",
    )

    data_dir = os.path.join(_HERE, "..", "..", "data")

    train(
        pretrained_ckpt=pretrained_ckpt,
        seed=args.seed,
        pretrained_seed=args.pretrained_seed,
        jitter=args.jitter,
        num_epochs=args.epochs,
        cog_lr=args.cog_lr,
        batch_size=args.batch_size,
        accumulation_steps=args.accum,
        save_dir=save_dir,
        log_path=log_path,
        data_dir=data_dir,
        model_name=args.model,
        freeze_layers=args.freeze,
    )
