#!/bin/bash
#
# Train all NLP baselines with 5 seeds each.
# Idempotent: skips seeds whose checkpoints already exist.
#
# NB: each baseline script in archive/baselines/ must support a --seed
# CLI argument. If a script doesn't, this loop will fail at that script
# and you'll need to add seed support.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

SEEDS=(1 2 3 42 100)

# (display_name, script_filename)
BASELINES=(
    "linear_regression:linear_regression.py"
    "lightgbm:lightgbm_baseline.py"
    "gpt2_surprisal:gpt2_surprisal.py"
    "bert_regression:bert_regression.py"
    "ohio_state_roberta:run_ohio_state_on_geco.py"
    "toronto_cl_roberta:run_toronto_on_geco.py"
)

for entry in "${BASELINES[@]}"; do
    name="${entry%%:*}"
    script="${entry##*:}"

    for seed in "${SEEDS[@]}"; do
        ckpt_dir="archive/baselines/checkpoints_${name}/seed${seed}"
        ckpt_file="$ckpt_dir/best_model.pt"

        if [ -f "$ckpt_file" ] || [ -d "$ckpt_dir" -a -n "$(ls -A "$ckpt_dir" 2>/dev/null)" ]; then
            echo "  $name seed=$seed: checkpoint exists, skipping."
            continue
        fi

        mkdir -p "$ckpt_dir"
        echo "  $name seed=$seed: training..."
        python -u "archive/baselines/$script" --seed "$seed" \
            --output_dir "$ckpt_dir" \
            > "logs/exp01_baseline_${name}_seed${seed}.out" 2>&1 \
            || echo "  WARNING: $name seed=$seed failed (check log)."
    done
done

echo "All baseline seeds attempted."
