"""
Evaluate the v4c_v2 cascade with cog scalars set to the SAME fitted-classical
parameters used in 04_eval_ez_classical.py, with both AI heads zeroed.

What this isolates: the contribution of the deterministic differentiable
cascade (vs. the stochastic discrete-event simulator), holding cog
parameter values constant. Same eight parameters in both runs — only the
cascade implementation differs.

Specifically:
  - Cog scalars are LOADED from
    `ez_classical/fitted_params.json` and applied to v4c_v2's parameter
    storage via inverse transformations.
  - ctx_head and skip_residual_head are REPLACED with zero-returning
    modules at inference time.
  - The LLaMA backbone and projection still run (their outputs are
    consumed by the zeroed heads and discarded), but contribute nothing.

Parameter mapping:
  fitted_classical                v4c_v2 internal storage
  ──────────────────              ──────────────────────────────────
  alpha1                          alpha1_reichle property
                                  (set via l1_base_offset, l1_freq_coef)
  alpha2                          alpha2_reichle property (same)
  eccentricity                    _epsilon_raw = inv_softplus(ε − 1)
  delta                           _delta_raw = logit(δ)
  lambda                          lambda_refix (direct)
  saccade_programming (M1)        _M1_raw = inv_softplus(M1)
  saccade_finishing  (M2 = I)     _M2I_raw = inv_softplus(M2)
  alpha3                          NOT APPLICABLE (v4c_v2 has no α3 /
                                  predictability term; pred would enter
                                  via ctx_head which is zeroed)

  Not in fitted_classical:        v4c_v2 default
  refix_pivot                     8.0
  skip_temperature                ≈31.0

Output:
  complete_metrics/results/raw/v4c_v2_classical_params_seed1.json
  (seed=1 by convention; the simulation here is deterministic, so the
   "seed" is a label only and identical across reruns.)

Usage:
  python -u .../07_eval_v4c_v2_classical_params.py
  python -u .../07_eval_v4c_v2_classical_params.py --force
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC_V2 = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))   # /repo/src_v2
LM_MODEL = os.path.join(SRC_V2, "lm_model")                      # /repo/src_v2/lm_model
sys.path.insert(0, SRC_V2)
sys.path.insert(0, LM_MODEL)
sys.path.insert(0, _HERE)

from paper_experiments import config  # noqa: E402
from paper_experiments.utils.load_data import (  # noqa: E402
    load_geco_aggregated, load_provo_aggregated, load_subtlex,
)
from paper_experiments.utils.eval_metrics import eval_predictions_on_aggregated  # noqa: E402

from model_llama_hybrid_v4c_v2 import NeuralEZReaderHybrid  # noqa: E402

from metrics import metrics_summary_complete  # local


FITTED_JSON = Path(_HERE) / "ez_classical" / "fitted_params.json"
OUT_DIR = Path(_HERE) / "results" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Inverse transformations
# --------------------------------------------------------------------------- #


def _inv_softplus(y: float) -> float:
    """Inverse of softplus: log(exp(y) - 1).  Numerically stable for y > 0."""
    if y <= 0:
        raise ValueError(f"softplus(_) > 0, got target y={y}")
    if y > 20:
        # softplus(x) ~ x for large x; use approximation to avoid overflow
        return y - math.log1p(-math.exp(-y))
    return math.log(math.expm1(y))


def _logit(p: float) -> float:
    if not (0 < p < 1):
        raise ValueError(f"sigmoid output in (0, 1), got target p={p}")
    return math.log(p / (1.0 - p))


# --------------------------------------------------------------------------- #
#  Head-zeroing module
# --------------------------------------------------------------------------- #


class _ZeroScalarHead(nn.Module):
    """Drop-in replacement for ctx_head / skip_residual_head: returns zeros."""

    def forward(self, x):
        return torch.zeros(
            x.shape[:-1] + (1,),
            device=x.device,
            dtype=x.dtype,
        )


# --------------------------------------------------------------------------- #
#  Cog-scalar override
# --------------------------------------------------------------------------- #


def _apply_fitted_classical_params(model: NeuralEZReaderHybrid,
                                   fitted: dict) -> dict:
    """
    Override v4c_v2's cog scalars to match the fitted classical-EZ-Reader
    parameters. Returns a dict of (key, before, after, what_v4c_stores)
    for the output JSON.
    """
    log = []

    def _set(name: str, raw_attr_path, new_value: float, transform_note: str = ""):
        """Set a raw parameter and log before/after. raw_attr_path can be
        a (module, attr_name) tuple or a chain like
        (model.ezreader, "_M1_raw"). new_value is in *raw* space."""
        module, attr = raw_attr_path
        current = float(getattr(module, attr).item())
        with torch.no_grad():
            getattr(module, attr).data.fill_(float(new_value))
        log.append({
            "param": name,
            "raw_attr": f"{module.__class__.__name__}.{attr}",
            "raw_before": current,
            "raw_after": float(new_value),
            "transform": transform_note,
        })

    # ---- alpha1 + alpha2 (frequency formula intercept + slope) -------- #
    # In v4c_v2:
    #   base_L1_formula = l1_base_offset + l1_freq_coef * log_freq_norm
    #   where log_freq_norm = (log(freq) - 10) / 5
    # Reichle-unit aliases:
    #   alpha1_reichle = l1_base_offset - 2 * l1_freq_coef
    #   alpha2_reichle = -l1_freq_coef / 5
    # Given target alpha1, alpha2:
    #   l1_freq_coef = -5 * alpha2
    #   l1_base_offset = alpha1 + 2 * l1_freq_coef
    target_alpha1 = fitted["alpha1"]
    target_alpha2 = fitted["alpha2"]
    new_l1_freq_coef = -5.0 * target_alpha2
    new_l1_base_offset = target_alpha1 + 2.0 * new_l1_freq_coef
    _set("l1_freq_coef",  (model, "l1_freq_coef"),  new_l1_freq_coef,
         "= -5 * alpha2")
    _set("l1_base_offset", (model, "l1_base_offset"), new_l1_base_offset,
         "= alpha1 + 2 * l1_freq_coef")

    # ---- epsilon (eccentricity factor) -------------------------------- #
    target_eps = fitted["eccentricity"]
    new_eps_raw = _inv_softplus(target_eps - 1.0)
    _set("_epsilon_raw", (model.ezreader, "_epsilon_raw"), new_eps_raw,
         f"epsilon = 1 + softplus(raw); set raw = inv_softplus({target_eps - 1:.4f})")

    # ---- M1 (saccade programming) ------------------------------------- #
    target_M1 = fitted["saccade_programming"]
    new_M1_raw = _inv_softplus(target_M1)
    _set("_M1_raw", (model.ezreader, "_M1_raw"), new_M1_raw,
         f"M1 = softplus(raw); set raw = inv_softplus({target_M1:.4f})")

    # ---- M2 = I (saccade finishing / integration time, tied) ---------- #
    target_M2 = fitted["saccade_finishing"]
    new_M2_raw = _inv_softplus(target_M2)
    _set("_M2I_raw", (model.ezreader, "_M2I_raw"), new_M2_raw,
         f"M2 = I = softplus(raw); set raw = inv_softplus({target_M2:.4f})")

    # ---- delta (L2 / L1 ratio) ---------------------------------------- #
    target_delta = fitted["delta"]
    new_delta_raw = _logit(target_delta)
    _set("_delta_raw", (model, "_delta_raw"), new_delta_raw,
         f"delta = sigmoid(raw); set raw = logit({target_delta:.4f})")

    # ---- lambda_refix (refixation length sensitivity) ----------------- #
    # NB: lambda_refix and refix_pivot live on model.ezreader, not on model.
    target_lambda = fitted["lambda"]
    _set("lambda_refix", (model.ezreader, "lambda_refix"), target_lambda,
         "(direct nn.Parameter on model.ezreader; no transform)")

    # ---- NOT in fitted classical: refix_pivot, skip_temperature ------- #
    # Keep v4c_v2 defaults. Logged for clarity.
    log.append({
        "param": "refix_pivot",
        "note": "not in fitted_classical; kept at v4c_v2 default 8.0",
        "raw_after": float(model.ezreader.refix_pivot.item()),
    })
    log.append({
        "param": "_skip_temperature_raw",
        "note": "not in fitted_classical; kept at v4c_v2 default raw "
                "(skip_temperature ~31.0)",
        "raw_after": float(model.ezreader._skip_temperature_raw.item()),
    })

    # ---- alpha3 / predictability: NOT APPLICABLE --------------------- #
    log.append({
        "param": "alpha3",
        "note": (
            "v4c_v2 has no alpha3 / predictability term in its L1 formula. "
            "Predictability would normally enter via ctx_head, which is "
            f"zeroed in this evaluation. fitted value {fitted['alpha3']:.4f} "
            "is therefore ignored."
        ),
    })

    return log


def _zero_out_ai_heads(model: nn.Module) -> list:
    """Replace ctx_head and skip_residual_head with zero-returning modules."""
    replaced = []
    if hasattr(model, "ctx_head"):
        model.ctx_head = _ZeroScalarHead()
        replaced.append("ctx_head")
    if hasattr(model, "skip_residual_head"):
        model.skip_residual_head = _ZeroScalarHead()
        replaced.append("skip_residual_head")
    return replaced


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_path = OUT_DIR / "v4c_v2_classical_params_seed1.json"
    if out_path.exists() and not args.force:
        print(f">> {out_path.name} exists, skipping (--force to redo).")
        return

    # ---- Load fitted classical params ---- #
    if not FITTED_JSON.exists():
        print(f"ERROR: {FITTED_JSON} not found. "
              f"Run run_fit_and_eval.sh first to produce it.")
        sys.exit(1)
    fitted_payload = json.loads(FITTED_JSON.read_text())
    fitted = fitted_payload["fitted_params"]
    print(f"Loaded fitted classical parameters from {FITTED_JSON}:")
    for k, v in fitted.items():
        print(f"  {k:<22s} = {v:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ---- Instantiate v4c_v2 fresh (no checkpoint load) ---- #
    print("\nInstantiating fresh v4c_v2 model "
          "(loads TinyLlama backbone; cog scalars at v4c_v2 defaults)...")
    model = NeuralEZReaderHybrid(
        model_name=config.BACKBONE_MODEL,
        freeze_layers=config.FREEZE_LAYERS,
        hidden_dim=256,
    ).to(device)

    # ---- Override cog scalars to match fitted classical ---- #
    print("\nApplying fitted-classical cog scalars to v4c_v2 internals...")
    override_log = _apply_fitted_classical_params(model, fitted)

    # ---- Verify by printing the human-readable values that are now active ---- #
    print("\nv4c_v2 cog scalars after override (in Reichle units where applicable):")
    print(f"  alpha1_reichle     = {model.alpha1_reichle.item():+.4f}  "
          f"(target {fitted['alpha1']:.4f})")
    print(f"  alpha2_reichle     = {model.alpha2_reichle.item():+.4f}  "
          f"(target {fitted['alpha2']:.4f})")
    print(f"  epsilon            = {model.ezreader.epsilon.item():+.4f}  "
          f"(target {fitted['eccentricity']:.4f})")
    print(f"  delta              = {model.delta.item():+.4f}  "
          f"(target {fitted['delta']:.4f})")
    print(f"  M1                 = {model.ezreader.M1.item():+.4f}  "
          f"(target {fitted['saccade_programming']:.4f})")
    print(f"  M2 = I             = {model.ezreader.M2.item():+.4f}  "
          f"(target {fitted['saccade_finishing']:.4f})")
    print(f"  lambda_refix       = {model.ezreader.lambda_refix.item():+.4f}  "
          f"(target {fitted['lambda']:.4f})")
    print(f"  refix_pivot        = {model.ezreader.refix_pivot.item():+.4f}  "
          f"(default; not in fitted_classical)")
    print(f"  skip_temperature   = {model.ezreader.skip_temperature.item():+.4f}  "
          f"(default; not in fitted_classical)")

    # ---- Zero AI heads ---- #
    print("\nZeroing AI heads (ctx_head, skip_residual_head)...")
    zeroed = _zero_out_ai_heads(model)
    print(f"  replaced: {zeroed}")

    model.eval()

    # ---- Eval ---- #
    print("\nLoading SUBTLEX...")
    subtlex = load_subtlex()

    t0 = time.time()
    print("\nGECO test: predicting...")
    geco_test = load_geco_aggregated("test")
    arr_geco, _ = eval_predictions_on_aggregated(model, geco_test, device, subtlex)

    print("Provo: predicting...")
    provo = load_provo_aggregated()
    arr_provo, _ = eval_predictions_on_aggregated(model, provo, device, subtlex)

    geco_summary = metrics_summary_complete(
        arr_geco["pred_trt"], arr_geco["pred_ffd"],
        arr_geco["pred_gaze"], arr_geco["pred_skip"],
        arr_geco["human_trt"], arr_geco["human_ffd"],
        arr_geco["human_gaze"], arr_geco["human_skip"],
    )
    provo_summary = metrics_summary_complete(
        arr_provo["pred_trt"], arr_provo["pred_ffd"],
        arr_provo["pred_gaze"], arr_provo["pred_skip"],
        arr_provo["human_trt"], arr_provo["human_ffd"],
        arr_provo["human_gaze"], arr_provo["human_skip"],
    )

    payload = {
        "model": "v4c_v2_classical_params",
        "seed": 1,
        "is_lesion": True,
        "description": (
            "v4c_v2 cascade (deterministic differentiable) with cog scalars "
            "set to the fitted-classical parameters from fitted_params.json, "
            "and both ctx_head and skip_residual_head zeroed. "
            "Isolates the cascade-implementation effect (deterministic "
            "expected-value computation vs Monte Carlo simulation) at "
            "matched parameter values. Predictability is unavailable in "
            "v4c_v2's base_L1 formula and would normally enter via "
            "ctx_head (which is zeroed)."
        ),
        "fitted_classical_source": str(FITTED_JSON),
        "applied_params": fitted,
        "param_override_log": override_log,
        "zeroed_heads": zeroed,
        "datasets": {
            "geco_test": geco_summary,
            "provo": provo_summary,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nWrote {out_path}  (total {time.time() - t0:.1f}s)")

    # Compact summary
    print("\n========== Summary ==========")
    for label, s in (("GECO test", geco_summary), ("Provo", provo_summary)):
        print(f"\n{label}:")
        for m in ("trt", "ffd", "gaze", "skip"):
            unit = "" if m == "skip" else " ms"
            print(f"  {m.upper():<5s}  r={s[f'r_{m}']:+.3f}  "
                  f"MAE={s[f'mae_{m}']:.3f}{unit}  "
                  f"bias={s[f'bias_{m}']:+.3f}")


if __name__ == "__main__":
    main()
