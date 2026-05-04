#!/bin/bash
#
# Train the paper model (v4c_v2 or v4c_v2_wide_prior) with 5 seeds.
# Skips seeds whose best_model.pt already exists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# Determine which training script to use based on config.PAPER_MODEL_RECIPE.
RECIPE=$(python -c "
import sys
sys.path.insert(0, 'src_v2/paper_experiments')
from config import PAPER_MODEL_RECIPE, RECIPE_TO_PATHS
print(RECIPE_TO_PATHS[PAPER_MODEL_RECIPE]['train_script'])
")
CKPT_DIR=$(python -c "
import sys
sys.path.insert(0, 'src_v2/paper_experiments')
from config import PAPER_MODEL_RECIPE, RECIPE_TO_PATHS
print(RECIPE_TO_PATHS[PAPER_MODEL_RECIPE]['ckpt_dir'])
")

TRAIN_SCRIPT="src_v2/lm_train/$RECIPE"
echo "Training script: $TRAIN_SCRIPT"
echo "Checkpoint dir base: checkpoints/$CKPT_DIR"

SEEDS=(1 2 3 42 100)
EPOCHS=5

for seed in "${SEEDS[@]}"; do
    ckpt="checkpoints/$CKPT_DIR/geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed${seed}/best_model.pt"
    if [ -f "$ckpt" ]; then
        echo "  seed=$seed: $ckpt exists, skipping."
        continue
    fi
    echo "  seed=$seed: training..."
    python -u "$TRAIN_SCRIPT" --epochs "$EPOCHS" --seed "$seed" \
        > "logs/exp01_paper_model_seed${seed}.out" 2>&1
    echo "  seed=$seed: done."
done

echo "All paper-model seeds complete."
