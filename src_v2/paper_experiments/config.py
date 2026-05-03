"""
Single source of truth for the paper-experiments pipeline.

All paths, seeds, and hyperparameters live here. Each experiment script
imports from this module rather than hard-coding values.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Repository layout
# --------------------------------------------------------------------------- #

# Resolve to /home/u384661/Neuro_EZR/ regardless of where scripts run from.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_V2 = REPO_ROOT / "src_v2"
LM_MODEL_DIR = SRC_V2 / "lm_model"
LM_TRAIN_DIR = SRC_V2 / "lm_train"
EVAL_DIR = SRC_V2 / "evaluation"
ARCHIVE_EZREADER = REPO_ROOT / "archive" / "original_ezreader"
BASELINES_DIR = REPO_ROOT / "archive" / "baselines"
DATA_DIR = REPO_ROOT / "data"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
LOGS_DIR = REPO_ROOT / "logs"

PAPER_EXPERIMENTS_DIR = SRC_V2 / "paper_experiments"
PAPER_FINAL_RESULTS = PAPER_EXPERIMENTS_DIR / "results"
PAPER_FINAL_TABLES = PAPER_FINAL_RESULTS / "tables"
PAPER_FINAL_FIGURES = PAPER_FINAL_RESULTS / "figures"

# --------------------------------------------------------------------------- #
#  Paper model selection
# --------------------------------------------------------------------------- #

# Paper model: v4c_v2_dualctx (two specialized ctx heads). Decided after
# the architectural ablation found this the best balance of all metrics.
# Options: "v4c_v2", "v4c_v2_wide_prior", "v4c_v2_dualctx"
PAPER_MODEL_RECIPE = "v4c_v2_dualctx"

# Maps recipe name -> (model module, training script, checkpoint dir name)
RECIPE_TO_PATHS = {
    "v4c_v2": {
        "model_module": "model_llama_hybrid_v4c_v2",
        "train_script": "train_hybrid_v4c_v2_geco.py",
        "ckpt_dir": "hybrid_v4c_v2",
    },
    "v4c_v2_wide_prior": {
        "model_module": "model_llama_hybrid_v4c_v2",  # same model class
        "train_script": "train_hybrid_v4c_v2_wide_prior_geco.py",
        "ckpt_dir": "hybrid_v4c_v2_wide_prior",
    },
    "v4c_v2_dualctx": {
        "model_module": "model_llama_hybrid_v4c_v2_dualctx",
        "train_script": "train_hybrid_v4c_v2_dualctx_geco.py",
        "ckpt_dir": "hybrid_v4c_v2_dualctx",
    },
}

# --------------------------------------------------------------------------- #
#  Backbone / data
# --------------------------------------------------------------------------- #

BACKBONE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BACKBONE_MODEL_SHORT = BACKBONE_MODEL.replace("/", "_")
FREEZE_LAYERS = 16   # 75% of TinyLlama's 22 layers (auto-set by training scripts)

GECO_READING_FILE = DATA_DIR / "Geco_MonolingualReadingData.csv"
GECO_MATERIAL_FILE = DATA_DIR / "Geco_EnglishMaterial.csv"
GECO_PRED_FILE = DATA_DIR / "geco_predictability.pkl"
PROVO_FILE = DATA_DIR / "Provo_Corpus-Eyetracking_Data.csv"
SUBTLEX_FILE = DATA_DIR / "SUBTLEXus.txt"

# --------------------------------------------------------------------------- #
#  Seed configuration
# --------------------------------------------------------------------------- #

# Used for: paper model multi-seed (#1b), randinit recovery (#2), surp ablation (#7)
SEEDS = [1, 2, 3, 42, 100]
N_SEEDS = len(SEEDS)
DEFAULT_SEED = 42

# --------------------------------------------------------------------------- #
#  Random-init recovery (exp02)
# --------------------------------------------------------------------------- #

# ±50% jitter around Reichle 2003 values for cog scalars.
RANDINIT_JITTER = 0.5

# --------------------------------------------------------------------------- #
#  Per-participant cog fits (exp09)
# --------------------------------------------------------------------------- #

# Lower LR for fine-tuning a few scalars on small data per reader.
PER_PARTICIPANT_COG_LR = 3e-5      # = cog_lr / 10
PER_PARTICIPANT_EPOCHS = 3
PER_PARTICIPANT_BATCH_SIZE = 8

# GECO has 14 monolingual participants. Filled in dynamically by load_data
# in case some have been dropped.
N_PARTICIPANTS_EXPECTED = 14

# --------------------------------------------------------------------------- #
#  Paper-model training defaults (read by training scripts via CLI args)
# --------------------------------------------------------------------------- #

LM_LR = 2e-5
HEAD_LR = 5e-4
COG_LR = 3e-4
NUM_EPOCHS = 5
BATCH_SIZE = 8
ACCUMULATION_STEPS = 8

# --------------------------------------------------------------------------- #
#  Reichle 2003 reference values for recovery + plotting
#  Only the parameters with a published Reichle value (used in exp02).
# --------------------------------------------------------------------------- #

REICHLE_TARGETS = {
    'alpha1_reichle': 104.0,
    'alpha2_reichle': 3.4,
    'epsilon': 1.15,
    'M1': 125.0,
    'M2_eq_I': 25.0,
    'delta': 0.34,
    'lambda_refix': 0.16,
}

# Default initial values used by v4c_v2 (Reichle-aligned but not exact for alpha1).
REICHLE_INITS = {
    'alpha1_reichle': 94.0,   # = 60 - 2*(-17)
    'alpha2_reichle': 3.4,    # = -(-17)/5
    'epsilon': 1.15,
    'M1': 125.0,
    'M2_eq_I': 25.0,
    'delta': 0.34,
    'lambda_refix': 0.4,
}

# --------------------------------------------------------------------------- #
#  Baseline definitions for exp01
# --------------------------------------------------------------------------- #

BASELINES = [
    # (name, training_script_relative_to_BASELINES_DIR, supports_seed_arg)
    ("linear_regression", "linear_regression.py", True),
    ("lightgbm",          "lightgbm_baseline.py",  True),
    ("gpt2_surprisal",    "gpt2_surprisal.py",     True),
    ("bert_regression",   "bert_regression.py",    True),
    ("ohio_state_roberta", "run_ohio_state_on_geco.py", True),
    ("toronto_cl_roberta", "run_toronto_on_geco.py",    True),
]

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def paper_model_ckpt_path(seed: int = DEFAULT_SEED, recipe: str = None) -> Path:
    """Return Path to best_model.pt for the paper model at the given seed."""
    recipe = recipe or PAPER_MODEL_RECIPE
    info = RECIPE_TO_PATHS[recipe]
    return (
        CHECKPOINTS_DIR
        / info["ckpt_dir"]
        / f"geco_{BACKBONE_MODEL_SHORT}_seed{seed}"
        / "best_model.pt"
    )


def randinit_ckpt_path(seed: int) -> Path:
    """Return Path to best_model.pt for randinit at the given seed."""
    return (
        CHECKPOINTS_DIR
        / "hybrid_v4c_v2_randinit"
        / f"geco_{BACKBONE_MODEL_SHORT}_seed{seed}"
        / "best_model.pt"
    )


def surp_ckpt_path(seed: int) -> Path:
    """Return Path to best_model.pt for v4c_v2_surp at the given seed."""
    return (
        CHECKPOINTS_DIR
        / "hybrid_v4c_v2_surp"
        / f"geco_{BACKBONE_MODEL_SHORT}_seed{seed}"
        / "best_model.pt"
    )


def baseline_ckpt_path(name: str, seed: int) -> Path:
    """Return Path for a baseline model checkpoint at the given seed."""
    return BASELINES_DIR / f"checkpoints_{name}" / f"seed{seed}" / "best_model.pt"


def ensure_dir(path) -> Path:
    """Create directory if missing; return Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
