#!/bin/bash
#
# Train v4c_v2_surp (ctx_head replaced by α3·surprisal) with 5 seeds.
# Idempotent: skips seeds whose checkpoints exist.
#
# Requires: data/cache/tinyllama_surprisal_geco_*.pt produced by
#           precompute_surprisal.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# Verify surprisal caches exist
for split in train val test; do
    cache="data/cache/tinyllama_surprisal_geco_${split}.pt"
    if [ ! -f "$cache" ]; then
        echo "ERROR: missing surprisal cache: $cache"
        echo "Run: python src_v2/paper_experiments/exp07_ctx_vs_surprisal/precompute_surprisal.py"
        exit 1
    fi
done

SEEDS=(1 2 3 42 100)
EPOCHS=5

for seed in "${SEEDS[@]}"; do
    ckpt="checkpoints/hybrid_v4c_v2_surp/geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed${seed}/best_model.pt"
    if [ -f "$ckpt" ]; then
        echo "  seed=$seed: $ckpt exists, skipping."
        continue
    fi
    echo "  seed=$seed: training v4c_v2_surp..."
    python -u src_v2/lm_train/train_hybrid_v4c_v2_surp_geco.py \
        --epochs "$EPOCHS" --seed "$seed" \
        > "logs/exp07_surp_seed${seed}.out" 2>&1
    echo "  seed=$seed: done."
done

echo "All v4c_v2_surp seeds complete."
