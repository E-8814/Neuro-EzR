#!/bin/bash
#
# Diagnostic for the frozen-backbone parameter-recovery experiment.
#
# Question: when the frozen-backbone training drifts AWAY from Reichle 2003,
# is that because (a) Reichle is a basin-of-attraction the random-init runs
# missed (multimodal optimum), or (b) Reichle is not a local minimum at all
# given our LM features (the cascade is structurally inadequate)?
#
# This script tests outcome (a) vs (b) directly: initialize cog scalars
# EXACTLY at Reichle 2003 (jitter=0), freeze the backbone, train.
#
# Read the result:
#   - If params stay close to Reichle and val improves → Reichle IS a local
#     minimum. The random-init runs ended up in a different basin → "the
#     dissociation is real" interpretation holds.
#   - If params drift away from Reichle to the same offset point as the
#     random-init runs → Reichle is NOT a local minimum for this model. The
#     cascade is structurally inadequate to represent Reichle's dynamics.
#   - If params drift somewhere else / oscillate wildly → loss landscape
#     is too flat, not enough signal in the training data.
#
# Run on ONE seed only (seed=1, paired with pretrained dualctx seed=1) —
# we don't need 5 seeds to answer the qualitative question.
#
# Outputs:
#   logs/randinit_frozen_diagnostic/seed1.out
#   checkpoints/hybrid_v4c_v2_randinit_frozen_diagnostic/geco_TinyLlama_..._seed1/

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="logs/randinit_frozen_diagnostic"
mkdir -p "$LOG_DIR"

SEED=1
EPOCHS=5
COG_LR=1e-3
JITTER=0.0    # <-- the diagnostic
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_SHORT="${MODEL//\//_}"

OUT_DIR="checkpoints/hybrid_v4c_v2_randinit_frozen_diagnostic/geco_${MODEL_SHORT}_seed${SEED}"
LOG="${LOG_DIR}/seed${SEED}.out"

pretrained_ckpt="checkpoints/hybrid_v4c_v2_dualctx/geco_${MODEL_SHORT}_seed${SEED}/best_model.pt"
if [ ! -f "$pretrained_ckpt" ]; then
    echo "ERROR: missing dualctx checkpoint at $pretrained_ckpt"
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "Diagnostic: jitter=0 (cog params start at Reichle 2003 exactly)"
echo "  pretrained: $pretrained_ckpt"
echo "  output:     $OUT_DIR"
echo "  log:        $LOG"
echo ""

python -u src_v2/lm_train/train_hybrid_v4c_v2_randinit_frozen_geco.py \
    --seed "$SEED" --pretrained_seed "$SEED" \
    --jitter "$JITTER" --epochs "$EPOCHS" --cog_lr "$COG_LR" \
    --model "$MODEL" \
    --output_dir "$OUT_DIR" \
    --log_path "${LOG_DIR}/seed${SEED}_train.log" \
    > "$LOG" 2>&1

echo "diagnostic done — see $LOG"
