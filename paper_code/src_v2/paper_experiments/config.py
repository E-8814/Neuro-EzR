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

# Resolve to the repo root regardless of where scripts run from.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_V2 = REPO_ROOT / "src_v2"
LM_MODEL_DIR = SRC_V2 / "lm_model"
LM_TRAIN_DIR = SRC_V2 / "lm_train"
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

PAPER_MODEL_RECIPE = "v4c_v2_dualctx"

RECIPE_TO_PATHS = {
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

# Used for: paper model multi-seed (exp01b) and surp ablation (exp07).
SEEDS = [1, 2, 3, 42, 100]
N_SEEDS = len(SEEDS)
DEFAULT_SEED = 42

# --------------------------------------------------------------------------- #
#  Per-group cog fits (exp09)
# --------------------------------------------------------------------------- #

PER_PARTICIPANT_COG_LR = 3e-5      # = COG_LR / 10
PER_PARTICIPANT_EPOCHS = 3
PER_PARTICIPANT_BATCH_SIZE = 8

N_PARTICIPANTS_EXPECTED = 14

# --------------------------------------------------------------------------- #
#  Paper-model training defaults
# --------------------------------------------------------------------------- #

LM_LR = 2e-5
HEAD_LR = 5e-4
COG_LR = 3e-4
NUM_EPOCHS = 5
BATCH_SIZE = 8
ACCUMULATION_STEPS = 8

# --------------------------------------------------------------------------- #
#  Reichle 2003 reference values (used by exp09 plotting)
# --------------------------------------------------------------------------- #

REICHLE_INITS = {
    'alpha1_reichle': 94.0,
    'alpha2_reichle': 3.4,
    'epsilon': 1.15,
    'M1': 125.0,
    'M2_eq_I': 25.0,
    'delta': 0.34,
    'lambda_refix': 0.4,
}

# --------------------------------------------------------------------------- #
#  Baseline definitions for exp01 (Table 1)
# --------------------------------------------------------------------------- #

BASELINES = [
    # (name, training_script_relative_to_BASELINES_DIR, supports_seed_arg)
    ("linear_regression", "linear_regression.py", True),
    ("lightgbm",          "lightgbm_baseline.py",  True),
    ("gpt2_surprisal",    "gpt2_surprisal.py",     True),
    ("bert_regression",   "bert_regression.py",    True),
    ("ohio_state_roberta", "run_ohio_state_on_geco.py", True),
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


def surp_ckpt_path(seed: int) -> Path:
    """Return Path to best_model.pt for v4c_v2_surp at the given seed (exp07)."""
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
