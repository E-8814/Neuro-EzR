#!/bin/bash
#
# Evaluate v4c_v2 cascade with cog scalars overridden to the SAME
# fitted-classical parameters used for ez_reader_classical, and both AI
# heads zeroed.
#
# This isolates the cascade-implementation effect at matched parameter
# values (deterministic expected-value vs MC simulation), holding all
# eight Reichle scalars constant.
#
# Requires:
#   complete_metrics/ez_classical/fitted_params.json
#     (produced by run_fit_and_eval.sh's Step 1)
#
# Writes:
#   complete_metrics/results/raw/v4c_v2_classical_params_seed1.json
# and re-aggregates the comparison CSVs.
#
# Submit:
#   sbatch run_eval_classical_params.sh
#   bash   run_eval_classical_params.sh

#SBATCH --job-name=ezr_eval_classical_params
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

FORCE_FLAG=""
if [ "${FORCE:-0}" = "1" ]; then FORCE_FLAG="--force"; fi

# ---------------------------------------------------------------------------- #
# Preflight: confirm fitted_params.json exists
# ---------------------------------------------------------------------------- #
FITTED_JSON="$CM_DIR/ez_classical/fitted_params.json"
echo "================================================================"
echo "Preflight: confirm fitted classical params exist"
echo "================================================================"
if [ ! -f "$FITTED_JSON" ]; then
    echo ">> ERROR: $FITTED_JSON not found."
    echo "   This script requires the fitted classical parameters."
    echo "   Run run_fit_and_eval.sh first to produce fitted_params.json."
    exit 1
fi
echo ">> Found $FITTED_JSON"
python - <<'PYEOF'
import json
with open("src_v2/paper_experiments/exp01_main_comparison/complete_metrics/"
          "ez_classical/fitted_params.json") as f:
    d = json.load(f)
print(f"   fit history: {d.get('n_evals')} evals on "
      f"{d.get('n_sentences')} GECO train sentences")
print(f"   parameters that will be applied to v4c_v2 cascade:")
for k, v in d['fitted_params'].items():
    print(f"     {k:<22s} = {v:.4f}")
PYEOF
echo ""

# ---------------------------------------------------------------------------- #
# Step 1: eval v4c_v2 cascade with classical params + zeroed AI
# ---------------------------------------------------------------------------- #
echo "================================================================"
echo "Step 1: v4c_v2 cascade with fitted-classical params, zeroed AI"
echo "================================================================"
t0=$(date +%s)
$PY "$CM_DIR/07_eval_v4c_v2_classical_params.py" $FORCE_FLAG
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
echo "Step 2: aggregate -> CSV"
echo "================================================================"
$PY "$CM_DIR/aggregate.py"
echo ""

echo ">> Done at $(date)"
echo ">> Outputs:"
echo "    $CM_DIR/results/raw/v4c_v2_classical_params_seed1.json"
echo "    $CM_DIR/results/comparison_results_complete.csv"
echo "    $CM_DIR/results/per_seed_metrics_complete.csv"
