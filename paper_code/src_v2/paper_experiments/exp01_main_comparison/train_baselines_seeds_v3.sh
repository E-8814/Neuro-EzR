#!/bin/bash
#
# Train all NLP baselines (v3). Honest seed handling:
#
#   - linear_regression / lightgbm / gpt2_surprisal: deterministic given
#     the data (closed-form fit / fixed feature extraction). Multi-seed
#     would produce identical results — run ONCE, sentinel at
#     checkpoints_<name>/seed1/.done.
#
#   - bert_regression / run_ohio_state_on_geco:
#     have stochastic training (random init/dropout/shuffle). Run 5 seeds each.
#     Per-seed sentinel + output dir at checkpoints_<name>/seed{N}/.
#
# Retry-friendly: failed runs leave no sentinel and will retry next time.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

SEEDS=(1 2 3 42 100)

# Single-run (deterministic) baselines — no --seed support / no benefit.
SINGLE_RUN=(
    "linear_regression:linear_regression.py"
    "lightgbm:lightgbm_baseline.py"
    "gpt2_surprisal:gpt2_surprisal.py"
)

# Multi-seed neural baselines (now seed-aware after the v3 edits).
MULTI_SEED=(
    "bert_regression:bert_regression.py"
    "ohio_state_roberta:run_ohio_state_on_geco.py"
)

# ---- Single-run pass ---------------------------------------------------- #
for entry in "${SINGLE_RUN[@]}"; do
    name="${entry%%:*}"
    script="${entry##*:}"

    sentinel="archive/baselines/checkpoints_${name}/seed1/.done"
    log="logs/exp01_baseline_${name}.out"

    if [ -f "$sentinel" ]; then
        echo "  $name: sentinel exists, skipping."
        continue
    fi

    mkdir -p "$(dirname "$sentinel")"
    echo "  $name: training (single run, deterministic)..."
    if python -u "archive/baselines/$script" > "$log" 2>&1; then
        touch "$sentinel"
        echo "  $name: done -> $sentinel"
    else
        echo "  WARNING: $name failed (check $log)."
    fi
done

# ---- Multi-seed pass ---------------------------------------------------- #
for entry in "${MULTI_SEED[@]}"; do
    name="${entry%%:*}"
    script="${entry##*:}"

    for seed in "${SEEDS[@]}"; do
        out_dir="archive/baselines/checkpoints_${name}/seed${seed}"
        sentinel="$out_dir/.done"
        log="logs/exp01_baseline_${name}_seed${seed}.out"

        if [ -f "$sentinel" ]; then
            echo "  $name seed=$seed: sentinel exists, skipping."
            continue
        fi

        mkdir -p "$out_dir"
        echo "  $name seed=$seed: training..."
        if python -u "archive/baselines/$script" \
                --seed "$seed" --output_dir "$out_dir" \
                > "$log" 2>&1; then
            touch "$sentinel"
            echo "  $name seed=$seed: done -> $sentinel"
        else
            echo "  WARNING: $name seed=$seed failed (check $log)."
        fi
    done
done

echo "All baselines attempted."
