"""
Lesion study (exp03): apply each lesion to the paper model and report
Δr vs the full model on GECO test + Provo.

Lesions are implemented by patching tensors in-flight (no permanent
state mutation). Each lesion type is a function that takes the model's
forward output dict and returns a modified dict.

Usage:
    python run_lesions.py --seed 42
    python run_lesions.py --seed 42 --lesions zero_ctx_head zero_skip_residual
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from paper_experiments import config
from paper_experiments.utils.load_data import (
    load_geco_aggregated, load_provo_aggregated, load_subtlex,
    word_frequency,
)
from paper_experiments.utils.load_model import load_paper_model
from paper_experiments.utils.eval_metrics import corr, mae, bias


RESULTS_DIR = Path(_HERE) / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LESION_CSV = RESULTS_DIR / "lesion_results.csv"


# --------------------------------------------------------------------------- #
#  Lesion functions: each takes (model, sentence) and runs forward with a
#  patched output dict.
# --------------------------------------------------------------------------- #


def _forward_one_sentence(model, sentence, device, subtlex):
    """Run model on a single aggregated sentence; returns the output dict.
    (Kept for backward compatibility; evaluate() now uses _collate_batch
    + _forward_batch to match the training script's batched eval.)"""
    word_lists = [sentence.tokens]
    freqs = torch.tensor(
        [float(word_frequency(t, subtlex)) for t in sentence.tokens],
        dtype=torch.float32,
    ).unsqueeze(0).to(device)
    wlens = torch.tensor(
        [len(t) for t in sentence.tokens], dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            return model(word_lists, freqs, wlens)


def _collate_batch(batch, device, subtlex):
    """Collate a batch of aggregated sentences with pad_sequence — exactly
    matches the training script's collate_aggregated."""
    from torch.nn.utils.rnn import pad_sequence

    word_lists = [a.tokens for a in batch]
    freqs = pad_sequence(
        [torch.tensor([float(word_frequency(t, subtlex)) for t in a.tokens],
                      dtype=torch.float32) for a in batch],
        batch_first=True, padding_value=1.0,
    ).to(device)
    wlens = pad_sequence(
        [torch.tensor([len(t) for t in a.tokens], dtype=torch.float32)
         for a in batch],
        batch_first=True,
    ).to(device)
    return word_lists, freqs, wlens


def _forward_batch(model, batch, device, subtlex):
    """Run the model on a batch of aggregated sentences with padding —
    matches training's evaluate_detailed exactly."""
    word_lists, freqs, wlens = _collate_batch(batch, device, subtlex)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            return model(word_lists, freqs, wlens)


def lesion_const_l1(p):
    """Replace per-word L1 with mean L1 across the sentence."""
    out = dict(p)
    L1 = p['L1'].clone()
    L1_mean = L1.mean(dim=-1, keepdim=True).expand_as(L1)
    out['L1'] = L1_mean
    # Recompute downstream: FFD, Gaze, TRT
    M1 = p['M1']
    M2 = p['M2']
    refix_prob = p['refix_prob']
    L2 = p['L2']
    I = p['I']
    out['first_fixation'] = L1_mean + M1 + M2
    out['gaze_duration'] = out['first_fixation'] + refix_prob * (L2 + M1 + M2)
    out['conditional_trt'] = out['gaze_duration'] + I
    return out


def lesion_const_l2(p):
    """Replace per-word L2 with mean L2."""
    out = dict(p)
    L2 = p['L2'].clone()
    L2_mean = L2.mean(dim=-1, keepdim=True).expand_as(L2)
    out['L2'] = L2_mean
    M1 = p['M1']; M2 = p['M2']; I = p['I']
    refix_prob = p['refix_prob']
    out['gaze_duration'] = p['first_fixation'] + refix_prob * (L2_mean + M1 + M2)
    out['conditional_trt'] = out['gaze_duration'] + I
    return out


def lesion_const_skip(p):
    """Replace every skip_prob with mean skip prob."""
    out = dict(p)
    skip = p['skip_prob'].clone()
    skip_mean = skip.mean(dim=-1, keepdim=True).expand_as(skip)
    out['skip_prob'] = skip_mean
    return out


def lesion_swap_l1l2(p):
    """Swap L1 and L2 values, recompute downstream."""
    out = dict(p)
    L1_new = p['L2'].clone()
    L2_new = p['L1'].clone()
    out['L1'] = L1_new
    out['L2'] = L2_new
    M1 = p['M1']; M2 = p['M2']; I = p['I']
    refix_prob = p['refix_prob']
    out['first_fixation'] = L1_new + M1 + M2
    out['gaze_duration'] = out['first_fixation'] + refix_prob * (L2_new + M1 + M2)
    out['conditional_trt'] = out['gaze_duration'] + I
    return out


def _shuffle_along_seq(tensor, generator):
    """Shuffle the last dim (seq T) of a (B, T) tensor.
    Generator is CPU-only; perm is moved to the tensor's device for indexing."""
    perm = torch.randperm(tensor.size(-1), generator=generator).to(tensor.device)
    return tensor[..., perm]


def lesion_shuffle_l1(p):
    out = dict(p)
    g = torch.Generator().manual_seed(0)  # CPU generator (required by randperm)
    L1_shuf = _shuffle_along_seq(p['L1'], g)
    out['L1'] = L1_shuf
    M1 = p['M1']; M2 = p['M2']; I = p['I']
    refix_prob = p['refix_prob']
    L2 = p['L2']
    out['first_fixation'] = L1_shuf + M1 + M2
    out['gaze_duration'] = out['first_fixation'] + refix_prob * (L2 + M1 + M2)
    out['conditional_trt'] = out['gaze_duration'] + I
    return out


def lesion_shuffle_l2(p):
    out = dict(p)
    g = torch.Generator().manual_seed(0)  # CPU generator (required by randperm)
    L2_shuf = _shuffle_along_seq(p['L2'], g)
    out['L2'] = L2_shuf
    M1 = p['M1']; M2 = p['M2']; I = p['I']
    refix_prob = p['refix_prob']
    out['gaze_duration'] = p['first_fixation'] + refix_prob * (L2_shuf + M1 + M2)
    out['conditional_trt'] = out['gaze_duration'] + I
    return out


def lesion_zero_ecc(p):
    """Set epsilon = 1.0, which means eccentricity factor = 1 (no scaling).
    base_L1 was already computed; L1_ecc = base_L1 * 1.0 = base_L1.
    So the lesion is effectively L1 = base_L1 + soft floor."""
    out = dict(p)
    base_L1 = p.get('base_L1')
    if base_L1 is None:
        return out
    # Reuse cascade's soft floor
    L1_floor = 5.0
    import torch.nn.functional as F
    L1 = L1_floor + F.softplus(base_L1 - L1_floor)
    out['L1'] = L1
    M1 = p['M1']; M2 = p['M2']; I = p['I']
    refix_prob = p['refix_prob']
    L2 = p['L2']
    out['first_fixation'] = L1 + M1 + M2
    out['gaze_duration'] = out['first_fixation'] + refix_prob * (L2 + M1 + M2)
    out['conditional_trt'] = out['gaze_duration'] + I
    return out


def lesion_no_l2_to_ffd(p):
    """Set L2 contribution in gaze to 0 (no refixation reading time)."""
    out = dict(p)
    M1 = p['M1']; M2 = p['M2']; I = p['I']
    refix_prob = p['refix_prob']
    # Gaze = FFD + refix_prob * (0 + M1 + M2)  — drop L2 from gaze
    out['gaze_duration'] = p['first_fixation'] + refix_prob * (M1 + M2)
    out['conditional_trt'] = out['gaze_duration'] + I
    return out


def lesion_zero_ctx_head(p):
    """
    Cannot patch ctx_head from output post-hoc cleanly because it's already
    embedded in base_L1. Implemented separately with a model-level patch
    (see _eval_with_zero_ctx_head).
    """
    return p   # placeholder; actual lesion uses a hook


def lesion_zero_skip_residual(p):
    """Set residual_skip_logit to zero, recompute skip_prob."""
    out = dict(p)
    race_logit = p['race_logit']
    # skip_prob = sigmoid(race_logit + 0)
    out['skip_prob'] = torch.sigmoid(race_logit)
    return out


def lesion_no_first_word_mask(p):
    """Restore the first-word skip prob from race+residual (undo the mask).
    Since we lost the unmasked value, approximate as sigmoid(race + residual)
    at position 0."""
    out = dict(p)
    skip = p['skip_prob'].clone()
    # The cascade applies first-word floor AFTER computing race+residual.
    # We can recover by recomputing position-0 skip from race_logit + residual.
    rl = p['race_logit']
    rs = p['residual_skip_logit']
    skip_unmasked = torch.sigmoid(rl + rs)
    skip[:, 0] = skip_unmasked[:, 0]
    out['skip_prob'] = skip
    return out


LESION_FUNCS_OUTPUT_LEVEL = {
    "const_l1": lesion_const_l1,
    "const_l2": lesion_const_l2,
    "const_skip": lesion_const_skip,
    "swap_l1l2": lesion_swap_l1l2,
    "shuffle_l1": lesion_shuffle_l1,
    "shuffle_l2": lesion_shuffle_l2,
    "zero_ecc": lesion_zero_ecc,
    "no_l2_to_ffd": lesion_no_l2_to_ffd,
    "zero_skip_residual": lesion_zero_skip_residual,
    "no_first_word_mask": lesion_no_first_word_mask,
}


# --------------------------------------------------------------------------- #
#  Model-level lesion: zero_ctx_head requires patching forward.
# --------------------------------------------------------------------------- #


class _ZeroCtxHead(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)


def _patch_zero_ctx_head(model, which: str = "all"):
    """
    Replace ctx head(s) with a module that always returns zeros.

    Handles both single-ctx (v4c_v2: model.ctx_head) and dual-ctx
    (v4c_v2_dualctx: model.ctx_head_FFD + model.ctx_head_skip) models.

    Args:
        which: "all" zeros every ctx head; for dualctx specifically,
            "FFD" zeros only ctx_head_FFD; "skip" zeros only ctx_head_skip.

    Returns:
        Dict {attribute_name: original_module} for restoration.
    """
    saved = {}
    z = _ZeroCtxHead()

    if hasattr(model, "ctx_head"):
        saved["ctx_head"] = model.ctx_head
        model.ctx_head = z

    if hasattr(model, "ctx_head_FFD") and which in ("all", "FFD"):
        saved["ctx_head_FFD"] = model.ctx_head_FFD
        model.ctx_head_FFD = z
    if hasattr(model, "ctx_head_skip") and which in ("all", "skip"):
        saved["ctx_head_skip"] = model.ctx_head_skip
        model.ctx_head_skip = z

    return saved


def _restore_ctx_head(model, saved):
    """Restore patched ctx head(s) from the dict returned by _patch_zero_ctx_head."""
    for attr, original in saved.items():
        setattr(model, attr, original)


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #


def evaluate(model, agg_data, device, subtlex, lesion_name=None,
             batch_size=8):
    """Evaluate model with optional lesion using BATCHED forward —
    matches training script's evaluate_detailed exactly so numbers are
    comparable to checkpoint val_metrics.

    Lesion functions operate on the (B, T) prediction dict; all existing
    output-level lesions are batch-compatible (they use dim=-1 reductions
    and elementwise ops)."""
    model.eval()  # belt and suspenders; load_paper_model already does this

    pt, ht = [], []
    pf, hf = [], []
    pg, hg = [], []
    ps, hs = [], []

    # Model-level lesions that patch ctx head(s):
    #   "zero_ctx_head" zeros all ctx heads (single-ctx and dual-ctx)
    #   "zero_ctx_FFD"  zeros only ctx_head_FFD (dual-ctx only)
    #   "zero_ctx_skip" zeros only ctx_head_skip (dual-ctx only)
    saved_ctx = None
    if lesion_name == "zero_ctx_head":
        saved_ctx = _patch_zero_ctx_head(model, which="all")
    elif lesion_name == "zero_ctx_FFD":
        saved_ctx = _patch_zero_ctx_head(model, which="FFD")
    elif lesion_name == "zero_ctx_skip":
        saved_ctx = _patch_zero_ctx_head(model, which="skip")

    try:
        for i in range(0, len(agg_data), batch_size):
            batch = agg_data[i:i + batch_size]
            p = _forward_batch(model, batch, device, subtlex)
            if lesion_name and lesion_name in LESION_FUNCS_OUTPUT_LEVEL:
                p = LESION_FUNCS_OUTPUT_LEVEL[lesion_name](p)

            for b in range(len(batch)):
                seq_len = len(batch[b].tokens)
                pt.extend(p['conditional_trt'][b, :seq_len].cpu().tolist())
                ht.extend(batch[b].mean_trt)
                pf.extend(p['first_fixation'][b, :seq_len].cpu().tolist())
                hf.extend(batch[b].mean_ffd)
                pg.extend(p['gaze_duration'][b, :seq_len].cpu().tolist())
                hg.extend(batch[b].mean_gaze)
                ps.extend(p['skip_prob'][b, :seq_len].cpu().tolist())
                hs.extend(batch[b].skip_rate)
    finally:
        if saved_ctx:
            _restore_ctx_head(model, saved_ctx)

    return {
        "r_trt": corr(pt, ht), "r_ffd": corr(pf, hf),
        "r_gaze": corr(pg, hg), "r_skip": corr(ps, hs),
        "mae_trt": mae(pt, ht), "mae_ffd": mae(pf, hf),
        "mae_gaze": mae(pg, hg),
        "bias_trt": bias(pt, ht), "bias_ffd": bias(pf, hf),
        "n_words": len(pt),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--lesions", nargs="*", default=None,
                        help="Subset of lesion names to run.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Loading paper model (seed={args.seed})...")
    model, ckpt = load_paper_model(seed=args.seed, device=device)

    print("Loading data...")
    subtlex = load_subtlex()
    geco_test = load_geco_aggregated("test")
    provo = load_provo_aggregated()
    print(f"  GECO test: {len(geco_test)} sentences")
    print(f"  Provo:     {len(provo)} sentences")

    datasets = {"geco_test": geco_test, "provo": provo}

    # Model-level (patch ctx_head module(s) before forward).
    # For dual-ctx model, also test FFD-only and skip-only ablations to
    # measure specialization directly.
    model_level_lesions = ["zero_ctx_head"]
    if hasattr(model, "ctx_head_FFD") and hasattr(model, "ctx_head_skip"):
        model_level_lesions += ["zero_ctx_FFD", "zero_ctx_skip"]

    all_lesions = ["full"] + list(LESION_FUNCS_OUTPUT_LEVEL.keys()) + model_level_lesions
    if args.lesions:
        lesions_to_run = ["full"] + args.lesions
    else:
        lesions_to_run = all_lesions

    rows = []
    full_metrics = {}

    for lesion in lesions_to_run:
        print(f"\n>> lesion: {lesion}")
        for ds_name, ds_data in datasets.items():
            metrics = evaluate(
                model, ds_data, device, subtlex,
                lesion_name=None if lesion == "full" else lesion,
            )
            print(f"   {ds_name:<10s} r_TRT={metrics['r_trt']:.3f} "
                  f"r_FFD={metrics['r_ffd']:.3f} r_Gaze={metrics['r_gaze']:.3f} "
                  f"r_skip={metrics['r_skip']:.3f}")

            if lesion == "full":
                full_metrics[ds_name] = metrics

            for metric_name, value in metrics.items():
                if metric_name == "n_words":
                    continue
                delta = (value - full_metrics[ds_name][metric_name]) \
                    if lesion != "full" and ds_name in full_metrics else 0.0
                rows.append({
                    "lesion": lesion,
                    "dataset": ds_name,
                    "metric": metric_name,
                    "value": value,
                    "delta_vs_full": delta,
                })

    with open(LESION_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lesion", "dataset", "metric", "value", "delta_vs_full"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {len(rows)} rows to {LESION_CSV}")


if __name__ == "__main__":
    main()
