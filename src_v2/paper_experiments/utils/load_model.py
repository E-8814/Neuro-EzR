"""
Model loading utilities for the paper-experiments pipeline.

Knows how to load any v4c-family checkpoint:
    - v4c_v2 / v4c_v2_wide_prior  (paper-model recipes)
    - v4c_v2_randinit              (random-init recovery experiment)
    - v4c_v2_surp                  (ctx_head replaced by surprisal)

Always loads with `weights_only=False` since we save numpy scalars
in val_metrics.

Usage:
    from utils.load_model import load_paper_model
    model, ckpt_meta = load_paper_model(seed=42, device="cuda")
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import torch

# Add lm_model and lm_train to sys.path so we can import their modules.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src_v2", "lm_model"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "archive", "original_ezreader"))

from .. import config  # noqa: E402  (intentionally relative)


def _resolve_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def load_paper_model(
    seed: int = config.DEFAULT_SEED,
    device=None,
    recipe: str = None,
):
    """
    Load the paper model from its trained checkpoint. Selects the correct
    model class based on the recipe:
        - "v4c_v2" / "v4c_v2_wide_prior" → model_llama_hybrid_v4c_v2
        - "v4c_v2_dualctx" → model_llama_hybrid_v4c_v2_dualctx

    Returns:
        (model, ckpt_meta) where ckpt_meta is the dict saved alongside
        the state_dict (epoch, val_step, val_metrics, cog_params, ...).
    """
    recipe = recipe or config.PAPER_MODEL_RECIPE
    device = _resolve_device(device)

    # Dispatch to correct model class based on recipe
    if recipe == "v4c_v2_dualctx":
        from model_llama_hybrid_v4c_v2_dualctx import NeuralEZReaderHybrid  # noqa: E402
    else:
        # v4c_v2 and v4c_v2_wide_prior share the same model class
        from model_llama_hybrid_v4c_v2 import NeuralEZReaderHybrid  # noqa: E402

    ckpt_path = config.paper_model_ckpt_path(seed=seed, recipe=recipe)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Paper model checkpoint not found: {ckpt_path}\n"
            f"Run training first (exp01b)."
        )

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    model = NeuralEZReaderHybrid(
        model_name=ckpt.get("model_name", config.BACKBONE_MODEL),
        freeze_layers=ckpt.get("freeze_layers", config.FREEZE_LAYERS),
        hidden_dim=ckpt.get("hidden_dim", 256),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_randinit_model(seed: int, device=None):
    """Load a v4c_v2_randinit model checkpoint for the given seed."""
    from model_llama_hybrid_v4c_v2_randinit import NeuralEZReaderHybrid  # noqa: E402

    device = _resolve_device(device)
    ckpt_path = config.randinit_ckpt_path(seed=seed)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Randinit checkpoint not found: {ckpt_path}\n"
            f"Run training first (exp02)."
        )

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # NB: at this point we want to recreate the model with the SAME
    # init_seed and jitter as during training, so that sampled_init
    # is reproducible. The training script saves these into ckpt.
    init_seed = ckpt.get("init_seed", seed)
    jitter = ckpt.get("jitter", config.RANDINIT_JITTER)

    model = NeuralEZReaderHybrid(
        model_name=ckpt.get("model_name", config.BACKBONE_MODEL),
        freeze_layers=ckpt.get("freeze_layers", config.FREEZE_LAYERS),
        hidden_dim=ckpt.get("hidden_dim", 256),
        init_seed=init_seed,
        jitter=jitter,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_surp_model(seed: int, device=None):
    """Load a v4c_v2_surp (TinyLlama-surprisal-replacing-ctx_head) model."""
    from model_llama_hybrid_v4c_v2_surp import NeuralEZReaderHybrid  # noqa: E402

    device = _resolve_device(device)
    ckpt_path = config.surp_ckpt_path(seed=seed)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Surp checkpoint not found: {ckpt_path}\n"
            f"Run training first (exp07)."
        )

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    model = NeuralEZReaderHybrid(
        model_name=ckpt.get("model_name", config.BACKBONE_MODEL),
        freeze_layers=ckpt.get("freeze_layers", config.FREEZE_LAYERS),
        hidden_dim=ckpt.get("hidden_dim", 256),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def collect_cog_params(model) -> dict:
    """
    Return a flat dict of cognitive scalar names -> current learned values.

    Works for v4c_v2 and v4c_v2_randinit (same param set; surp may differ
    slightly due to the alpha3 parameter, see surp variant).
    """
    ezr = model.ezreader
    out = {
        'l1_base_offset': model.l1_base_offset.item(),
        'l1_freq_coef': model.l1_freq_coef.item(),
        'alpha1_reichle': model.alpha1_reichle.item(),
        'alpha2_reichle': model.alpha2_reichle.item(),
        'delta': model.delta.item(),
        'epsilon': ezr.epsilon.item(),
        'M1': ezr.M1.item(),
        'M2_eq_I': ezr.M2.item(),
        'lambda_refix': ezr.lambda_refix.item(),
        'refix_pivot': ezr.refix_pivot.item(),
        'skip_temperature': ezr.skip_temperature.item(),
    }
    # surp variant has alpha3
    if hasattr(model, 'alpha3'):
        out['alpha3'] = model.alpha3.item()
    return out


def freeze_neural_layers(model):
    """
    Freeze all neural-net components (LLaMA, projection, ctx_head(s),
    skip_residual_head) so that only cognitive scalars receive gradients.

    Handles all v4c-family models including dualctx (which has
    ctx_head_FFD and ctx_head_skip instead of a single ctx_head).

    Used by the per-participant cog-fit experiment (exp09).
    """
    # Freeze LLaMA
    for p in model.llama.parameters():
        p.requires_grad = False
    # Freeze projection
    for p in model.projection.parameters():
        p.requires_grad = False
    # Freeze ctx head(s) — single (v4c_v2) or dual (v4c_v2_dualctx)
    if hasattr(model, "ctx_head"):
        for p in model.ctx_head.parameters():
            p.requires_grad = False
    if hasattr(model, "ctx_head_FFD"):
        for p in model.ctx_head_FFD.parameters():
            p.requires_grad = False
    if hasattr(model, "ctx_head_skip"):
        for p in model.ctx_head_skip.parameters():
            p.requires_grad = False
    # Freeze skip-residual head
    if hasattr(model, "skip_residual_head"):
        for p in model.skip_residual_head.parameters():
            p.requires_grad = False


def get_cog_param_list(model):
    """Return list of cognitive Parameter objects (for optimizer)."""
    cog_params = []
    # Direct on the model
    cog_params.append(model.l1_base_offset)
    cog_params.append(model.l1_freq_coef)
    cog_params.append(model._delta_raw)
    if hasattr(model, "alpha3"):
        cog_params.append(model.alpha3)
    # On the cascade
    ezr = model.ezreader
    for name in [
        "_epsilon_raw",
        "_M1_raw",
        "_M2I_raw",
        "lambda_refix",
        "refix_pivot",
        "_skip_temperature_raw",
    ]:
        if hasattr(ezr, name):
            cog_params.append(getattr(ezr, name))
    return cog_params
