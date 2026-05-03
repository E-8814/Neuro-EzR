#!/bin/bash
#
# Train all NLP baselines (v2). Honest about what each script supports:
#
#   - linear_regression / lightgbm / gpt2_surprisal: no argparse, silently
#     drop CLI args. Run ONCE — multi-seed isn't actually supported.
#   - bert_regression / run_ohio_state_on_geco / run_toronto_on_geco:
#     argparse rejects unknown args, no --seed support. Run ONCE with
#     each script's hardcoded seed.
#
# After a successful run, drop a sentinel `seed1/.done` so reruns skip.
# Retry-friendly: failed runs leave no sentinel and will retry next time.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# (display_name, script_filename, supports_seed_flag)
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

    sentinel="archive/baselines/checkpoints_${name}/seed1/.done"
    log="logs/exp01_baseline_${name}.out"

    if [ -f "$sentinel" ]; then
        echo "  $name: sentinel exists, skipping."
        continue
    fi

    mkdir -p "$(dirname "$sentinel")"
    echo "  $name: training (single run, no --seed)..."

    if python -u "archive/baselines/$script" > "$log" 2>&1; then
        touch "$sentinel"
        echo "  $name: done -> $sentinel"
    else
        echo "  WARNING: $name failed (check $log)."
    fi
done

echo "All baselines attempted."
