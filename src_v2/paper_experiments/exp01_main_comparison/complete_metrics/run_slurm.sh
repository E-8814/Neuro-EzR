#!/bin/bash
#
# Orchestrator for the complete-metric augmentation of Table 1.
#
# Produces fresh JSONs (with mae_skip, bias_skip, mae_gaze on flat
# baselines, and a new ez_reader_classical row) under
# complete_metrics/results/raw/{,baselines/}, then aggregates into
# complete_metrics/results/comparison_results_complete.csv.
#
# Order:
#   1. Re-evaluate paper model (5 seeds, GPU)              ~5 min
#   2. Re-evaluate BERT + Ohio State (5 seeds each, GPU)   ~30 min
#   3. Train + evaluate flat baselines (CPU + GPU for GPT-2 surprisal)
#                                                           ~15-30 min
#   4. Run classical E-Z Reader (CPU, parallel)            ~10-30 min
#   5. Aggregate to CSV
#
# All steps are idempotent — re-submission skips already-written JSONs.
#
# Submit:
#   sbatch run_slurm.sh
#   # or run interactively in a tmux session on a GPU node:
#   bash run_slurm.sh
#
# Edit the SBATCH lines for your cluster (partition, gres syntax).

#SBATCH --job-name=ezr_complete_metrics
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
echo ">> GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

PY="python -u"

echo "================================================================"
echo "Step 1: paper model (v4c_v2_dualctx, 5 seeds)"
echo "================================================================"
$PY "$CM_DIR/01_eval_paper_model.py" || \
    echo "  [warn] paper-model eval failed; continuing"
echo ""

echo "================================================================"
echo "Step 2: BERT + Ohio State (5 seeds each)"
echo "================================================================"
$PY "$CM_DIR/02_eval_bert_ohio.py" || \
    echo "  [warn] bert/ohio eval failed; continuing"
echo ""

echo "================================================================"
echo "Step 3: flat baselines (linear, lightgbm, gpt2_surprisal)"
echo "================================================================"
$PY "$CM_DIR/03_train_eval_flat_baselines.py" || \
    echo "  [warn] flat-baselines eval failed; continuing"
echo ""

echo "================================================================"
echo "Step 4: classical E-Z Reader (N=200 MC, parallel)"
echo "================================================================"
$PY "$CM_DIR/04_eval_ez_classical.py" --num_runs 200 || \
    echo "  [warn] EZ classical eval failed; continuing"
echo ""

echo "================================================================"
echo "Step 5: aggregate -> CSV"
echo "================================================================"
$PY "$CM_DIR/aggregate.py"
echo ""

echo ">> Done at $(date)"
echo ">> Final outputs:"
echo "    $CM_DIR/results/comparison_results_complete.csv"
echo "    $CM_DIR/results/per_seed_metrics_complete.csv"
