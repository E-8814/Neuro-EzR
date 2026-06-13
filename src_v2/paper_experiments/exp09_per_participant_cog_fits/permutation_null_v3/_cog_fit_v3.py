"""
v3 adaptation of the cached cog-scalar fit (see ../permutation_null/_cog_fit.py).

Differences from the v2 machinery, mirroring train_hybrid_v4c_v3_dualctx_geco.py:
  1. The model is v4c_v3_dualctx (no first-word skip clamp) loaded from the
     hybrid_v4c_v3_dualctx_next checkpoints.
  2. The refit loss supervises skip with the race-faithful alignment
     (row i scored against word i+1) and EXCLUDES sentence-initial words
     and pads (valid = word_length > 0.5).

Reused unchanged from ../permutation_null/_cog_fit.py:
  enumerate_balanced_splits, split_index, cog_only_forward,
  dissociation_T, load_cached_participants, pool_group_batches
(cog_only_forward calls model.ezreader, so the v3 cascade — without the
clamp — is used automatically when a v3 model is passed in.)
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP09 = os.path.abspath(os.path.join(_HERE, ".."))
PERM_V2 = os.path.join(EXP09, "permutation_null")
SRC_V2 = os.path.abspath(os.path.join(EXP09, "..", ".."))
REPO_ROOT = os.path.dirname(SRC_V2)
LM_MODEL = os.path.join(SRC_V2, "lm_model")

for p in (SRC_V2, EXP09, PERM_V2, LM_MODEL, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from paper_experiments.utils.load_model import (  # noqa: E402
    freeze_neural_layers, get_cog_param_list, collect_cog_params,
)
from _cog_fit import cog_only_forward  # noqa: E402  (reused as-is)


SIGMA2_TRT = 10000.0
SIGMA2_FFD = 1500.0
SIGMA2_GAZE = 4500.0

V3_CKPT_TMPL = os.path.join(
    REPO_ROOT, "checkpoints", "hybrid_v4c_v3_dualctx_next",
    "geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed{seed}", "best_model.pt",
)


def load_v3_model(seed: int, device: torch.device):
    """Load a v4c_v3_dualctx checkpoint (skip_align=next)."""
    from model_llama_hybrid_v4c_v3_dualctx import NeuralEZReaderHybrid
    path = V3_CKPT_TMPL.format(seed=seed)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing v4c_v3 checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = NeuralEZReaderHybrid(
        model_name=ckpt["model_name"],
        freeze_layers=ckpt["freeze_layers"],
        hidden_dim=ckpt.get("hidden_dim", 256),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def loss_fn_v3(pred, h_trt, h_ffd, h_gaze, h_skip, word_lengths):
    """
    v3 refit loss: time losses identical to fit_per_participant.loss_fn;
    skip BCE next-aligned (pred row i vs word i+1) over valid non-initial
    words only.
    """
    pred_trt = pred['conditional_trt'].float()
    pred_ffd = pred['first_fixation'].float()
    pred_gaze = pred['gaze_duration'].float()
    pred_skip = pred['skip_prob'].float()

    fixated = (h_skip < 0.5)
    if fixated.sum() > 0:
        trt_mse = F.mse_loss(pred_trt[fixated], h_trt[fixated])
        ffd_mse = F.mse_loss(pred_ffd[fixated], h_ffd[fixated])
        gaze_mse = F.mse_loss(pred_gaze[fixated], h_gaze[fixated])
    else:
        zero = torch.tensor(0.0, device=pred_trt.device)
        trt_mse = ffd_mse = gaze_mse = zero

    valid = word_lengths > 0.5
    sp = pred_skip[:, :-1].clamp(1e-6, 1 - 1e-6)
    st = h_skip[:, 1:]
    sm = valid[:, 1:]
    if sm.sum() > 0:
        skip_loss = F.binary_cross_entropy(sp[sm], st[sm])
    else:
        skip_loss = torch.tensor(0.0, device=pred_trt.device)

    return (
        trt_mse / SIGMA2_TRT + ffd_mse / SIGMA2_FFD
        + gaze_mse / SIGMA2_GAZE + skip_loss
    )


def fit_group_cached_v3(
    model,
    group_batches: List[Dict[str, torch.Tensor]],
    device: torch.device,
    epochs: int,
    lr: float,
    *,
    rng_seed: int = 0,
) -> Tuple[Dict[str, float], float]:
    """Same loop as _cog_fit.fit_group_cached, with the v3 loss."""
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
        for idx in order:
            cb = group_batches[idx]
            B = cb["log_freq_norm"].shape[0]
            batch = {k: v.to(device, non_blocking=True) for k, v in cb.items()}
            pred = cog_only_forward(model, batch)
            loss = loss_fn_v3(
                pred, batch["h_trt"], batch["h_ffd"],
                batch["h_gaze"], batch["h_skip"], batch["word_lengths"],
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cog_params, 1.0)
            optimizer.step()
            epoch_loss += loss.item() * B
            n_seen += B
        losses.append(epoch_loss / max(1, n_seen))

    return collect_cog_params(model), (losses[-1] if losses else 0.0)
