#!/bin/bash
#
# Train v4c_v2_randinit with 5 seeds, each with a different random
# perturbation (±50% jitter) of cognitive scalars from Reichle 2003.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

SEEDS=(1 2 3 42 100)
EPOCHS=5
JITTER=0.5

for seed in "${SEEDS[@]}"; do
    ckpt="checkpoints/hybrid_v4c_v2_randinit/geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed${seed}/best_model.pt"
    if [ -f "$ckpt" ]; then
        echo "  seed=$seed: $ckpt exists, skipping."
        continue
    fi
    echo "  seed=$seed: training (jitter=±${JITTER})..."
    python -u src_v2/lm_train/train_hybrid_v4c_v2_randinit_geco.py \
        --epochs "$EPOCHS" --seed "$seed" --jitter "$JITTER" \
        > "logs/exp02_randinit_seed${seed}.out" 2>&1
    echo "  seed=$seed: done."
done

echo "All randinit seeds complete."
