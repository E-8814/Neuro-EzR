"""
Reproducible seeding for torch / numpy / random / cudnn.

Usage:
    from utils.seed_utils import set_all_seeds
    set_all_seeds(42)

This is called at the top of every Python entry-point script that
involves randomness (training, fitting, evaluation with stochastic
components).
"""

import os
import random as _random

import numpy as np
import torch


def set_all_seeds(seed: int, deterministic_cudnn: bool = True) -> None:
    """
    Set seed across all relevant libraries.

    Args:
        seed: integer seed for all RNGs.
        deterministic_cudnn: if True, sets cuDNN to deterministic mode
            (slower but reproducible). For training experiments where
            reproducibility matters, leave this True.
    """
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Also set the env var that some PyTorch ops respect.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def make_seeded_rng(seed: int) -> _random.Random:
    """Return a local random.Random seeded with `seed`. Useful when you
    need a separate RNG that doesn't disturb the global state."""
    return _random.Random(seed)
