"""
Shared helpers for the cached-feature cog-scalar fit.

Two functions:

    cog_only_forward(model, batch)   -- recompute the cog cascade outputs
                                        from cached frozen-neural features

    fit_group_cached(model, group_batches, ...)
                                     -- per-group SGD on cog scalars only,
                                        using cached features

Plus split-enumeration utilities:

    enumerate_balanced_splits(participants)  -> List[Tuple[Tuple[str], Tuple[str]]]
    split_index(splits, group_a, group_b)    -> int (or raises)

The cog scalars trained here are the same set as exp09's existing
fit_per_participant code path (l1_base_offset, l1_freq_coef, _delta_raw,
plus everything on model.ezreader). Tested in 02_sanity_check.py to match
the live code-path output for the actual fast/slow split.
"""

from __future__ import annotations

import os
import random
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_HERE, ".."))

from paper_experiments.utils.load_model import (
    freeze_neural_layers, get_cog_param_list, collect_cog_params,
)
from fit_per_participant import loss_fn  # reuse the same loss


# --------------------------------------------------------------------------- #
#  Split enumeration
# --------------------------------------------------------------------------- #


def enumerate_balanced_splits(
    participants: Sequence[str],
) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """
    Return all unique balanced 7/7 splits of the 14 participants.

    Two partitions {A,B} and {B,A} are the same. We pick a canonical
    ordering: the alphabetically-smallest participant always lives in
    group A. With 14 participants this gives exactly C(14,7)/2 = 1716
    unique splits.
    """
    participants = sorted(participants)
    n = len(participants)
    assert n % 2 == 0, f"need even count of participants, got {n}"
    half = n // 2
    pivot = participants[0]

    splits: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []
    for combo in combinations(participants, half):
        if pivot not in combo:
            continue
        group_a = tuple(combo)
        group_b = tuple(p for p in participants if p not in combo)
        splits.append((group_a, group_b))

    return splits


def split_index(
    splits: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    group_a: Sequence[str],
    group_b: Sequence[str],
) -> int:
    """Find which canonical split index corresponds to {group_a, group_b}."""
    target_a = frozenset(group_a)
    target_b = frozenset(group_b)
    for i, (a, b) in enumerate(splits):
        if frozenset(a) == target_a and frozenset(b) == target_b:
            return i
        if frozenset(a) == target_b and frozenset(b) == target_a:
            return i
    raise ValueError(f"Split not found in canonical enumeration.")


# --------------------------------------------------------------------------- #
#  Cached forward + fit loop
# --------------------------------------------------------------------------- #


def cog_only_forward(model, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Reproduce model.forward's post-LM path using cached features.

    Inputs (cached, frozen): log_freq_norm, word_lengths, ctx_FFD, ctx_skip,
                             residual_skip_logit
    Trainable: l1_base_offset, l1_freq_coef, delta, and everything on
               model.ezreader (epsilon, M1, M2=I, lambda_refix, refix_pivot,
               skip_temperature)

    Mirrors lines 328-357 of model_llama_hybrid_v4c_v2_dualctx.py.forward.
    """
    log_freq_norm        = batch["log_freq_norm"]
    word_lengths         = batch["word_lengths"]
    ctx_FFD              = batch["ctx_FFD"]
    ctx_skip             = batch["ctx_skip"]
    residual_skip_logit  = batch["residual_skip_logit"]

    base_L1_formula = model.l1_base_offset + model.l1_freq_coef * log_freq_norm
    l1_FFD_raw  = base_L1_formula + ctx_FFD
    l1_skip_raw = base_L1_formula + ctx_skip
    base_L1_FFD  = 5.0 + F.softplus(l1_FFD_raw  - 5.0)
    base_L1_skip = 5.0 + F.softplus(l1_skip_raw - 5.0)

    L2 = model.delta * base_L1_FFD

    return model.ezreader(
        base_L1_FFD=base_L1_FFD,
        base_L1_skip=base_L1_skip,
        L2=L2,
        residual_skip_logit=residual_skip_logit,
        word_lengths=word_lengths,
    )


def fit_group_cached(
    model,
    group_batches: List[Dict[str, torch.Tensor]],
    device: torch.device,
    epochs: int,
    lr: float,
    *,
    rng_seed: int = 0,
) -> Tuple[Dict[str, float], float]:
    """
    Frozen-neural cog-scalar fit on a list of cached batches.

    Each batch is a dict of cpu fp32 tensors (as produced by
    01_cache_features.py). We move each batch to `device` per step.

    Returns (collected_cog_params_dict, final_epoch_avg_loss).
    """
    freeze_neural_layers(model)
    cog_params = get_cog_param_list(model)
    optimizer = torch.optim.AdamW(cog_params, lr=lr)

    rng = random.Random(rng_seed)

    losses: List[float] = []
    for epoch in range(epochs):
        order = list(range(len(group_batches)))
        rng.shuffle(order)

        epoch_loss = 0.0
        n_seen = 0
        for step, idx in enumerate(order):
            cb = group_batches[idx]
            B = cb["log_freq_norm"].shape[0]
            batch = {k: v.to(device, non_blocking=True) for k, v in cb.items()}
            pred = cog_only_forward(model, batch)
            loss = loss_fn(
                pred, batch["h_trt"], batch["h_ffd"],
                batch["h_gaze"], batch["h_skip"],
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cog_params, 1.0)
            optimizer.step()
            epoch_loss += loss.item() * B
            n_seen += B
        losses.append(epoch_loss / max(1, n_seen))

    return collect_cog_params(model), (losses[-1] if losses else 0.0)


# --------------------------------------------------------------------------- #
#  Test statistic for the H3 dissociation
# --------------------------------------------------------------------------- #

LEXICAL_PARAMS = ("delta", "lambda_refix", "epsilon")
MOTOR_PARAMS   = ("M1", "M2_eq_I", "skip_temperature")


def dissociation_T(
    fast_cog: Dict[str, float], slow_cog: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute the dissociation statistic and its components.

    Per the H3 prediction:
      lexical-stage parameters change with reading speed,
      motor / decision parameters do not.

    %Δ = 100 * (slow - fast) / |fast|, abs-valued, then averaged within
    {lexical} and {motor} clusters. T = mean|%Δ_lex| - mean|%Δ_mot|.
    """
    def pct_change(p: str) -> float:
        f, s = float(fast_cog[p]), float(slow_cog[p])
        denom = abs(f) if abs(f) > 1e-9 else 1.0
        return 100.0 * (s - f) / denom

    lex = {p: abs(pct_change(p)) for p in LEXICAL_PARAMS}
    mot = {p: abs(pct_change(p)) for p in MOTOR_PARAMS}

    mean_lex = sum(lex.values()) / len(lex)
    mean_mot = sum(mot.values()) / len(mot)
    T = mean_lex - mean_mot

    return {
        "T": T,
        "mean_abs_pct_lexical": mean_lex,
        "mean_abs_pct_motor": mean_mot,
        **{f"abs_pct_{p}": v for p, v in lex.items()},
        **{f"abs_pct_{p}": v for p, v in mot.items()},
    }


# --------------------------------------------------------------------------- #
#  Cached-feature loading
# --------------------------------------------------------------------------- #


def load_cached_participants(cache_dir: Path) -> Dict[str, List[Dict[str, torch.Tensor]]]:
    """Load all per-participant caches into memory.

    Returns: {pid: [batch_dict, ...]}. cpu fp32 tensors.
    """
    out: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    for path in sorted(cache_dir.glob("*.pt")):
        if path.name.startswith("_"):
            continue
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        out[d["participant_id"]] = d["batches"]
    return out


def pool_group_batches(
    cache: Dict[str, List[Dict[str, torch.Tensor]]],
    readers: Sequence[str],
) -> List[Dict[str, torch.Tensor]]:
    """Concatenate the batch lists of a group of readers in reader order."""
    pooled: List[Dict[str, torch.Tensor]] = []
    for pid in readers:
        if pid not in cache:
            raise KeyError(f"Cached features missing for {pid}")
        pooled.extend(cache[pid])
    return pooled
