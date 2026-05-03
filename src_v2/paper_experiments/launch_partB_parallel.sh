#!/bin/bash
#
# Run exp01b (paper model) + exp02 (randinit) + exp07 (surp precompute +
# surp ablation) in a SEPARATE process from the main phase B pipeline.
#
# Why this script:
#   - Phase B is already running and will eventually train these too.
#   - Running them here in parallel finishes them faster.
#   - Phase B's wrappers all check `[ -f best_model.pt ]` before starting
#     a seed, so once we write a checkpoint they will skip.
#
# What this script does NOT do (intentionally):
#   - Edit any of the existing train scripts or wrapper scripts.
#   - Fork to a different checkpoint directory — the canonical path is
#     used so downstream eval/lesion/per-participant experiments can
#     load the trained models the normal way.
#
# Race-condition note:
#   - We re-check `best_model.pt` right before each python call. If
#     phase B finished that seed in the gap between `[ -f ... ]` and
#     `python ...`, we skip. Window is short; each seed takes hours.
#   - The realistic race is on the LAST seed if phase B's queue catches
#     up. Worst case: both processes train the same seed and one
#     overwrites the other's best_model.pt. Not corruption, just one
#     valid checkpoint instead of two. Acceptable.
#
# Logs land in: logs/partB_parallel/<exp>_seed<N>.out
#
# Usage:
#   bash launch_partB_parallel.sh
#   nohup bash launch_partB_parallel.sh > logs/partB_parallel/wrapper.out 2>&1 &
#
# Recommended: launch from your byzantium srun shell, with
# CUDA_VISIBLE_DEVICES set if you want a specific GPU.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="logs/partB_parallel"
mkdir -p "$LOG_DIR"

SEEDS=(1 2 3 42 100)
EPOCHS=5
RANDINIT_JITTER=0.5

run_one_seed() {
    # $1 label, $2 train script (absolute), $3 ckpt dir base
    # ($4..) extra args appended verbatim to the python call
    local label="$1"; shift
    local train_script="$1"; shift
    local ckpt_base="$1"; shift
    local extra_args=("$@")

    for seed in "${SEEDS[@]}"; do
        local ckpt="checkpoints/${ckpt_base}/geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed${seed}/best_model.pt"
        local log="${LOG_DIR}/${label}_seed${seed}.out"

        if [ -f "$ckpt" ]; then
            echo "  [$label seed=$seed] checkpoint exists, skipping."
            continue
        fi

        # Race-window narrowing: re-check just before launch.
        echo "  [$label seed=$seed] training... (log: $log)"
        if [ -f "$ckpt" ]; then
            echo "    (raced — checkpoint appeared, skipping.)"
            continue
        fi

        python -u "$train_script" --epochs "$EPOCHS" --seed "$seed" \
            "${extra_args[@]}" > "$log" 2>&1 \
            && echo "  [$label seed=$seed] done." \
            || echo "  WARNING: [$label seed=$seed] failed (check $log)."
    done
}

echo "=== exp01b: paper model (dualctx) × 5 seeds ==="
run_one_seed \
    "exp01b_paper_model" \
    "src_v2/lm_train/train_hybrid_v4c_v2_dualctx_geco.py" \
    "hybrid_v4c_v2_dualctx"

echo ""
echo "=== exp02: randinit recovery × 5 seeds (jitter=±${RANDINIT_JITTER}) ==="
run_one_seed \
    "exp02_randinit" \
    "src_v2/lm_train/train_hybrid_v4c_v2_randinit_geco.py" \
    "hybrid_v4c_v2_randinit" \
    --jitter "$RANDINIT_JITTER"

echo ""
echo "=== exp07a: TinyLlama surprisal precompute (one-shot) ==="
SURP_LOG="${LOG_DIR}/exp07_precompute_surprisal.out"
NEED_PRECOMPUTE=0
for split in train val test; do
    if [ ! -f "data/cache/tinyllama_surprisal_geco_${split}.pt" ]; then
        NEED_PRECOMPUTE=1
        break
    fi
done
if [ $NEED_PRECOMPUTE -eq 0 ]; then
    echo "  surprisal caches present, skipping precompute."
else
    echo "  precomputing... (log: $SURP_LOG)"
    ( cd src_v2/paper_experiments/exp07_ctx_vs_surprisal && \
      python -u precompute_surprisal.py ) > "$SURP_LOG" 2>&1 \
        && echo "  precompute done." \
        || { echo "  ERROR: precompute failed (check $SURP_LOG). Aborting exp07b."; exit 1; }
fi

echo ""
echo "=== exp07b: surp ablation × 5 seeds ==="
run_one_seed \
    "exp07_surp" \
    "src_v2/lm_train/train_hybrid_v4c_v2_surp_geco.py" \
    "hybrid_v4c_v2_surp"

echo ""
echo "=== launch_partB_parallel.sh complete ==="
