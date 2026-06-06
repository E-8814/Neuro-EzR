#!/bin/bash
#
# SLURM submit script for the H3 permutation null.
#
# Order:
#   1. cache features (idempotent; skipped if cache already populated)
#   2. sanity check (cached fast/slow fit ≈ published numbers)
#   3. branch:
#        sanity passed -> 03a_perm_cached.py  (1,716 splits, random order)
#        sanity failed -> 03b_perm_live.py    (300 random splits, live model)
#   4. aggregate whatever is done so far
#
# All intermediates persist on disk. Re-submitting picks up where the last
# run stopped (each split has its own JSON). To run multiple jobs in
# parallel on different GPUs, just submit this file twice — both will
# pick random missing indices.
#
# Submit:
#   sbatch run_slurm.sh
#
# Tweak the SBATCH lines below for your cluster (partition, account, gres,
# time). The defaults assume a generic single-GPU partition.

#SBATCH --job-name=ezr_perm_null
#SBATCH --output=logs/exp09_perm_%j.out
#SBATCH --error=logs/exp09_perm_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

# ---- Resolve repo root regardless of where sbatch is called from ---- #
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PERM_DIR="$(dirname "$SCRIPT_PATH")"
EXP09_DIR="$(dirname "$PERM_DIR")"
PE_DIR="$(dirname "$EXP09_DIR")"
SRC_V2="$(dirname "$PE_DIR")"
REPO_ROOT="$(dirname "$SRC_V2")"

cd "$REPO_ROOT"
mkdir -p logs

# ---- Activate environment (edit this line for your setup) ---- #
# Common patterns: `conda activate neuro_ezr`, or `source venv/bin/activate`
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    conda activate neuro_ezr 2>/dev/null || true
fi

echo ">> Repo root: $REPO_ROOT"
echo ">> Date:      $(date)"
echo ">> GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no GPU')"
echo ""

PY="python -u"

# ---- Step 1: cache features (idempotent) ---- #
echo "=================================================================="
echo "Step 1: cache LM features per participant"
echo "=================================================================="
$PY "$PERM_DIR/01_cache_features.py"
echo ""

# ---- Step 2: sanity check ---- #
echo "=================================================================="
echo "Step 2: sanity check (cached fast/slow vs published)"
echo "=================================================================="
SANITY_JSON="$PERM_DIR/results/sanity_check.json"

# Run the sanity check; do NOT exit on its non-zero rc — we branch on it.
SANITY_RC=0
$PY "$PERM_DIR/02_sanity_check.py" || SANITY_RC=$?
echo ">> sanity_check.py exit code: $SANITY_RC"

# Read the JSON's "passed" field.
SANITY_PASSED="False"
if [ -f "$SANITY_JSON" ]; then
    SANITY_PASSED=$(python -c "
import json
d = json.load(open('$SANITY_JSON'))
print('True' if d.get('passed') else 'False')
")
fi
echo ">> sanity_check passed=$SANITY_PASSED"
echo ""

# ---- Step 3: branch ---- #
if [ "$SANITY_PASSED" = "True" ]; then
    echo "=================================================================="
    echo "Step 3a: cached enumeration of all 1,716 balanced splits"
    echo "         (random order, atomic JSON-per-split, resumable)"
    echo "=================================================================="
    # Stop ~30 minutes before the SLURM wall clock so step 4 has time.
    BUDGET_MIN=$((${SBATCH_TIME_MINUTES:-1170}))
    $PY "$PERM_DIR/03a_perm_cached.py" --max_runtime_minutes "$BUDGET_MIN"
else
    echo "=================================================================="
    echo "Step 3b: LIVE-model fallback, 300 random splits"
    echo "         (sanity check failed; cached path not trusted)"
    echo "=================================================================="
    BUDGET_MIN=$((${SBATCH_TIME_MINUTES:-1170}))
    $PY "$PERM_DIR/03b_perm_live.py" \
        --num_perms 300 \
        --max_runtime_minutes "$BUDGET_MIN"
fi
echo ""

# ---- Step 4: aggregate whatever is done ---- #
echo "=================================================================="
echo "Step 4: aggregate completed perms -> distribution + p-value + plot"
echo "=================================================================="
$PY "$PERM_DIR/04_aggregate.py" || echo "  [warn] aggregator failed; rerun later when more perms complete"
echo ""

echo ">> Done at $(date)"
echo ">> Re-submit this script to continue if perms are still missing."
