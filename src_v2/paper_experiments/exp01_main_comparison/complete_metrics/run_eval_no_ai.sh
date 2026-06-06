#!/bin/bash
#
# Evaluate v4c_v2 with both neural correction heads (ctx_head,
# skip_residual_head) zeroed out — the "no AI" lesion.
#
# Loads the trained v4c_v2 seed-42 checkpoint, replaces ctx_head and
# skip_residual_head with zero-returning modules at inference time,
# and evaluates on GECO test + Provo with the same complete metric
# set as every other row in Table 1.
#
# Writes:
#   complete_metrics/results/raw/v4c_v2_no_ai_seed42.json
# and re-aggregates the comparison CSVs.
#
# Caveat: only seed 42 was trained for the v4c_v2 single-head model.
# This script defaults to evaluating only that seed.
#
# Submit:
#   sbatch run_eval_no_ai.sh         # SLURM
#   bash   run_eval_no_ai.sh         # interactive (tmux on a GPU node)

#SBATCH --job-name=ezr_eval_no_ai
#SBATCH --partition=GPU
#SBATCH --gres=gpu:rtx:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
CM_DIR="$(dirname "$SCRIPT_PATH")"
EXP01_DIR="$(dirname "$CM_DIR")"
PE_DIR="$(dirname "$EXP01_DIR")"
SRC_V2="$(dirname "$PE_DIR")"
REPO_ROOT="$(dirname "$SRC_V2")"

cd "$REPO_ROOT"
mkdir -p logs

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neuro_ezr

echo ">> Repo root:    $REPO_ROOT"
echo ">> Date:         $(date)"
echo ">> Hostname:     $(hostname)"
echo ">> GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

PY="python -u"

# Defaults; override via env if you ever need to.
SEEDS="${SEEDS:-42}"
FORCE_FLAG=""
if [ "${FORCE:-0}" = "1" ]; then FORCE_FLAG="--force"; fi

# ---------------------------------------------------------------------------- #
# Preflight: check v4c_v2 checkpoint exists
# ---------------------------------------------------------------------------- #
echo "================================================================"
echo "Preflight: confirm v4c_v2 seed-42 checkpoint exists"
echo "================================================================"
python - <<'PYEOF'
import os, sys
sys.path.insert(0, "src_v2")
from paper_experiments import config
ckpt = config.paper_model_ckpt_path(seed=42, recipe="v4c_v2")
if not os.path.isfile(ckpt):
    print(f">> ERROR: v4c_v2 seed-42 checkpoint not found at:\n   {ckpt}")
    print("   This script requires the single-head v4c_v2 model trained with")
    print("   train_hybrid_v4c_v2_geco.py at seed 42.")
    sys.exit(1)
print(f">> Found {ckpt}")
PYEOF
echo ""

# ---------------------------------------------------------------------------- #
# Step 1: zero-AI lesion eval
# ---------------------------------------------------------------------------- #
echo "================================================================"
echo "Step 1: v4c_v2 with ctx_head + skip_residual_head zeroed"
echo "        seeds=$SEEDS"
echo "================================================================"
t0=$(date +%s)
$PY "$CM_DIR/06_eval_v4c_v2_no_ai.py" --seeds $SEEDS $FORCE_FLAG
rc=$?
echo ""
echo ">> Step 1 finished in $(( $(date +%s) - t0 ))s with exit code $rc"
if [ "$rc" -ne 0 ]; then
    echo "   ERROR: eval failed. Aborting before aggregate."
    exit "$rc"
fi
echo ""

# ---------------------------------------------------------------------------- #
# Step 2: re-aggregate so the new row appears in the comparison table
# ---------------------------------------------------------------------------- #
echo "================================================================"
echo "Step 2: aggregate -> CSV (updates comparison_results_complete.csv)"
echo "================================================================"
$PY "$CM_DIR/aggregate.py"
echo ""

echo ">> Done at $(date)"
echo ">> Outputs:"
for seed in $SEEDS; do
    echo "    $CM_DIR/results/raw/v4c_v2_no_ai_seed${seed}.json"
done
echo "    $CM_DIR/results/comparison_results_complete.csv"
echo "    $CM_DIR/results/per_seed_metrics_complete.csv"
