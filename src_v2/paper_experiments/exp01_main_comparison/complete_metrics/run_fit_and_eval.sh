#!/bin/bash
#
# Sequentially:
#   1. Fit classical-E-Z-Reader parameters on GECO train  (CPU, ~10-20 min)
#   2. Evaluate the classical model on GECO test + Provo using the fitted
#      parameters, with N=200 Monte Carlo runs per sentence              (~20-40 min)
#   3. Re-aggregate all the per-model JSONs into the final CSV
#
# Steps 1 and 2 are idempotent — re-running the script after a partial
# completion skips the steps whose outputs already exist. Pass `--force-fit`
# or `--force-eval` (env var FORCE_FIT=1 / FORCE_EVAL=1) to redo.
#
# Submit:
#   sbatch run_fit_and_eval.sh         # SLURM
#   bash   run_fit_and_eval.sh         # interactive (e.g. inside a tmux on a GPU node)
#
# Edit the SBATCH lines below if your cluster needs different syntax.

#SBATCH --job-name=ezr_fit_and_eval
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

# Default knobs (override via env if you want):
#   N_SENTENCES   sentences from GECO train used per loss eval (200)
#   N_MC_FIT      MC runs per sentence during fit          (50)
#   MAXITER       Nelder-Mead iterations                    (250)
#   N_MC_EVAL     MC runs per sentence during final eval    (200)
#   FORCE_FIT     1 = redo fit even if fitted_params.json exists
#   FORCE_EVAL    1 = redo eval even if ez_reader_classical_seed1.json exists
N_SENTENCES="${N_SENTENCES:-200}"
N_MC_FIT="${N_MC_FIT:-50}"
MAXITER="${MAXITER:-250}"
N_MC_EVAL="${N_MC_EVAL:-200}"
FORCE_FIT_FLAG=""
FORCE_EVAL_FLAG=""
if [ "${FORCE_FIT:-0}" = "1" ];  then FORCE_FIT_FLAG="--force";  fi
if [ "${FORCE_EVAL:-0}" = "1" ]; then FORCE_EVAL_FLAG="--force"; fi

FITTED_JSON="$CM_DIR/ez_classical/fitted_params.json"
EVAL_JSON="$CM_DIR/results/raw/ez_reader_classical_seed1.json"

# ---------------------------------------------------------------------------- #
# Step 1: parameter fit on GECO train
# ---------------------------------------------------------------------------- #
echo "================================================================"
echo "Step 1: fit classical-E-Z-Reader parameters on GECO train"
echo "        n_sentences=$N_SENTENCES  n_mc=$N_MC_FIT  maxiter=$MAXITER"
echo "================================================================"

if [ -f "$FITTED_JSON" ] && [ -z "$FORCE_FIT_FLAG" ]; then
    echo ">> $FITTED_JSON already exists. Skipping fit."
    echo "   (set FORCE_FIT=1 to redo.)"
else
    $PY "$CM_DIR/05_fit_ez_classical_params.py" \
        --n_sentences "$N_SENTENCES" \
        --n_mc        "$N_MC_FIT" \
        --maxiter     "$MAXITER" \
        $FORCE_FIT_FLAG
    if [ ! -f "$FITTED_JSON" ]; then
        echo ">> ERROR: fit script finished but $FITTED_JSON was not produced. Aborting."
        exit 1
    fi
    echo ">> Fit complete. $FITTED_JSON written."
fi
echo ""

# ---------------------------------------------------------------------------- #
# Step 2: evaluate classical with fitted params (N=200 MC)
# ---------------------------------------------------------------------------- #
echo "================================================================"
echo "Step 2: evaluate classical E-Z Reader with FITTED parameters"
echo "        num_mc_runs=$N_MC_EVAL"
echo "================================================================"

$PY "$CM_DIR/04_eval_ez_classical.py" \
    --num_runs "$N_MC_EVAL" \
    $FORCE_EVAL_FLAG
echo ""

# ---------------------------------------------------------------------------- #
# Step 3: re-aggregate
# ---------------------------------------------------------------------------- #
echo "================================================================"
echo "Step 3: aggregate -> CSV"
echo "================================================================"

$PY "$CM_DIR/aggregate.py"
echo ""

echo ">> Done at $(date)"
echo ">> Outputs:"
echo "    $FITTED_JSON"
echo "    $EVAL_JSON"
echo "    $CM_DIR/results/comparison_results_complete.csv"
echo "    $CM_DIR/results/per_seed_metrics_complete.csv"
