#!/bin/bash
#
# Frozen-backbone parameter-recovery experiment (exp02 redesign).
#
# For each seed N in {1, 2, 3, 42, 100}:
#   - load dualctx pretrained checkpoint (seed=N) as the frozen feature extractor
#   - re-randomize the 9 cog scalars to ±50% jitter from Reichle 2003
#   - train ONLY cog scalars
#   - save to checkpoints/hybrid_v4c_v2_randinit_frozen/seed{N}/
#
# This is a sharper test than the original exp02: with the LM frozen, the
# only way to fit the data is by moving the cog scalars to whatever values
# match the LM's already-learned features. If those values cluster near
# Reichle 2003 across seeds, that's a strong cognitive-plausibility claim.
#
# Logs: logs/randinit_frozen/<seed>.out (one wrapper file per seed).
#
# Usage (from byzantium srun shell, neuro_ezr env):
#   bash launch_randinit_frozen.sh
#   nohup bash launch_randinit_frozen.sh > logs/randinit_frozen/wrapper.out 2>&1 &

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="logs/randinit_frozen"
mkdir -p "$LOG_DIR"

SEEDS=(1 2 3 42 100)
JITTER=0.5
EPOCHS=5
COG_LR=1e-3
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_SHORT="${MODEL//\//_}"

for seed in "${SEEDS[@]}"; do
    pretrained_ckpt="checkpoints/hybrid_v4c_v2_dualctx/geco_${MODEL_SHORT}_seed${seed}/best_model.pt"
    out_ckpt="checkpoints/hybrid_v4c_v2_randinit_frozen/geco_${MODEL_SHORT}_seed${seed}/best_model.pt"
    log="${LOG_DIR}/seed${seed}.out"

    if [ ! -f "$pretrained_ckpt" ]; then
        echo "  [seed=$seed] missing dualctx checkpoint at $pretrained_ckpt — skipping"
        continue
    fi
    if [ -f "$out_ckpt" ]; then
        echo "  [seed=$seed] frozen-recovery checkpoint already exists, skipping"
        continue
    fi

    echo "  [seed=$seed] training... (log: $log)"
    if python -u src_v2/lm_train/train_hybrid_v4c_v2_randinit_frozen_geco.py \
            --seed "$seed" --pretrained_seed "$seed" \
            --jitter "$JITTER" --epochs "$EPOCHS" --cog_lr "$COG_LR" \
            --model "$MODEL" \
            > "$log" 2>&1; then
        echo "  [seed=$seed] done."
    else
        echo "  WARNING: [seed=$seed] failed (check $log)."
    fi
done

echo ""
echo "=== launch_randinit_frozen.sh complete ==="
