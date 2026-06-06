"""
Evaluate the v4c_v2 cascade with both AI correction heads zeroed out.

Tests whether v4c_v2's cog cascade (with its trained scalars: α₁, α₂, ε,
δ, M₁, M₂, λ_refix, ...) reproduces classical-E-Z-Reader-style performance
when the neural correction is removed.

What gets zeroed:
  - ctx_head: the MLP producing the per-word scalar correction added to
              base_L1. Without it, base_L1 = α₁ + α₂·log_freq (Reichle's
              original frequency-only formula).
  - skip_residual_head: the MLP producing the per-word neural correction
              added inside the skip sigmoid. Without it, the skip race is
              pure cog math: P(skip) = σ((M₁ − L₁_next_para) / τ).

What stays:
  - All trained cog scalars (l1_base_offset, l1_freq_coef, _delta_raw,
    epsilon, M1, M2/I, lambda_refix, refix_pivot, skip_temperature).
  - The LLaMA backbone and projection layer (run, but their outputs are
    discarded because both consumer heads are zeroed).

Output:
  complete_metrics/results/raw/v4c_v2_no_ai_seed42.json

Caveat:
  Only seed 42 was trained for v4c_v2 (single-head). The other seeds
  trained the dualctx variant. So this is a single-seed result.

Usage:
  python -u .../06_eval_v4c_v2_no_ai.py
  python -u .../06_eval_v4c_v2_no_ai.py --seeds 42 100
  python -u .../06_eval_v4c_v2_no_ai.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC_V2 = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, SRC_V2)
sys.path.insert(0, _HERE)

from paper_experiments import config  # noqa: E402
from paper_experiments.utils.load_data import (  # noqa: E402
    load_geco_aggregated, load_provo_aggregated, load_subtlex,
)
from paper_experiments.utils.load_model import load_paper_model  # noqa: E402
from paper_experiments.utils.eval_metrics import eval_predictions_on_aggregated  # noqa: E402

from metrics import metrics_summary_complete  # local


OUT_DIR = Path(_HERE) / "results" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Head-zeroing module
# --------------------------------------------------------------------------- #


class _ZeroScalarHead(nn.Module):
    """
    Drop-in replacement for ctx_head and skip_residual_head.

    Both originals consume a [B, T, hidden_dim] tensor and return [B, T, 1],
    which is then `.squeeze(-1)`'d to [B, T] in the forward path. We return
    zeros of the same shape so the downstream `.squeeze(-1)` works, and the
    cascade sees a zero contribution from this head.
    """

    def forward(self, x):  # x: [B, T, hidden_dim]
        return torch.zeros(
            x.shape[:-1] + (1,),
            device=x.device,
            dtype=x.dtype,
        )


def _zero_out_ai_heads(model: nn.Module) -> dict:
    """
    Replace ctx_head and skip_residual_head with zero-returning modules.
    Returns a dict listing which heads were replaced (for the output JSON).

    Works for v4c_v2 single-head models. For dualctx models, would also need
    to handle ctx_head_FFD and ctx_head_skip; but this script targets v4c_v2.
    """
    replaced = []
    if hasattr(model, "ctx_head"):
        model.ctx_head = _ZeroScalarHead()
        replaced.append("ctx_head")
    # dualctx variant exposes split heads; cover them too for completeness
    if hasattr(model, "ctx_head_FFD"):
        model.ctx_head_FFD = _ZeroScalarHead()
        replaced.append("ctx_head_FFD")
    if hasattr(model, "ctx_head_skip"):
        model.ctx_head_skip = _ZeroScalarHead()
        replaced.append("ctx_head_skip")
    if hasattr(model, "skip_residual_head"):
        model.skip_residual_head = _ZeroScalarHead()
        replaced.append("skip_residual_head")
    return {"zeroed_heads": replaced}


# --------------------------------------------------------------------------- #
#  Per-seed eval
# --------------------------------------------------------------------------- #


def _eval_one_seed(seed: int, device: torch.device, subtlex) -> dict:
    print(f"\n>> seed={seed}: loading v4c_v2 checkpoint")
    model, ckpt_meta = load_paper_model(seed=seed, device=device, recipe="v4c_v2")
    print(f"   checkpoint loaded from "
          f"{config.paper_model_ckpt_path(seed=seed, recipe='v4c_v2')}")

    print(f"   zeroing AI heads (ctx_head, skip_residual_head)...")
    zeroed = _zero_out_ai_heads(model)
    print(f"   replaced: {zeroed['zeroed_heads']}")

    # Verify the cog scalars are still the trained ones (not Reichle defaults)
    print(f"   trained cog scalars retained:")
    for name in (
        "l1_base_offset", "l1_freq_coef",
        "alpha1_reichle", "alpha2_reichle", "delta",
    ):
        if hasattr(model, name):
            val = getattr(model, name)
            if callable(val) or hasattr(val, "item"):
                val = val.item() if hasattr(val, "item") else val()
            print(f"     {name:<22s} = {float(val):+.4f}")
    ezr = model.ezreader
    for name in (
        "epsilon", "M1", "M2", "I", "skip_temperature",
        "lambda_refix", "refix_pivot",
    ):
        if hasattr(ezr, name):
            val = getattr(ezr, name)
            if callable(val) or hasattr(val, "item"):
                val = val.item() if hasattr(val, "item") else val()
            print(f"     {name:<22s} = {float(val):+.4f}")

    model.eval()

    print(f"   GECO test: predicting...")
    geco_test = load_geco_aggregated("test")
    arr_geco, _ = eval_predictions_on_aggregated(model, geco_test, device, subtlex)

    print(f"   Provo: predicting...")
    provo = load_provo_aggregated()
    arr_provo, _ = eval_predictions_on_aggregated(model, provo, device, subtlex)

    out = {
        "model": "v4c_v2_no_ai",
        "seed": seed,
        "is_lesion": True,
        "lesion_description": (
            "v4c_v2 single-head checkpoint with both neural correction heads "
            "(ctx_head, skip_residual_head) replaced by zero-returning modules. "
            "Trained cog scalars are retained; LLaMA + projection still run but "
            "their outputs are discarded by the zeroed heads."
        ),
        "zeroed_heads": zeroed["zeroed_heads"],
        "datasets": {
            "geco_test": metrics_summary_complete(
                arr_geco["pred_trt"], arr_geco["pred_ffd"],
                arr_geco["pred_gaze"], arr_geco["pred_skip"],
                arr_geco["human_trt"], arr_geco["human_ffd"],
                arr_geco["human_gaze"], arr_geco["human_skip"],
            ),
            "provo": metrics_summary_complete(
                arr_provo["pred_trt"], arr_provo["pred_ffd"],
                arr_provo["pred_gaze"], arr_provo["pred_skip"],
                arr_provo["human_trt"], arr_provo["human_ffd"],
                arr_provo["human_gaze"], arr_provo["human_skip"],
            ),
        },
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=[42],
        help="Seeds to evaluate. Only seed 42 trained for v4c_v2 single-head; "
             "other seeds will fail to load.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if the output JSON already exists.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading SUBTLEX...")
    subtlex = load_subtlex()

    for seed in args.seeds:
        out_path = OUT_DIR / f"v4c_v2_no_ai_seed{seed}.json"
        if out_path.exists() and not args.force:
            print(f">> seed={seed}: {out_path.name} exists, skipping "
                  f"(use --force to redo).")
            continue
        t0 = time.time()
        try:
            payload = _eval_one_seed(seed, device, subtlex)
        except FileNotFoundError as exc:
            print(f">> seed={seed}: checkpoint not found -> {exc}")
            continue
        out_path.write_text(json.dumps(payload, indent=2, default=float))
        print(f">> seed={seed}: wrote {out_path}  "
              f"({time.time() - t0:.1f}s)")

        # Compact summary
        for ds_name, ds in payload["datasets"].items():
            print(f"\n   {ds_name}:")
            for m in ("trt", "ffd", "gaze", "skip"):
                unit = "" if m == "skip" else " ms"
                print(f"     {m.upper():<5s}  r={ds[f'r_{m}']:+.3f}  "
                      f"MAE={ds[f'mae_{m}']:.3f}{unit}")

    print("\nDone.")


if __name__ == "__main__":
    main()
