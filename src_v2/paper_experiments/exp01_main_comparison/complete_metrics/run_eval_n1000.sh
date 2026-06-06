#!/bin/bash
#
# Re-evaluate the classical E-Z Reader with FITTED parameters at
# N=1000 Monte Carlo runs per sentence — the canonical Reichle 2003
# MC budget. Assumes `ez_classical/fitted_params.json` already exists
# (produced by run_fit_and_eval.sh). If it does not, this script aborts
# loudly.
#
# Overwrites:
#   complete_metrics/results/raw/ez_reader_classical_seed1.json
# and re-aggregates the comparison CSVs.
#
# Submit:
#   sbatch run_eval_n1000.sh         # SLURM
#   bash   run_eval_n1000.sh         # interactive (tmux on GPU node)

#SBATCH --job-name=ezr_eval_n1000
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

# Override via env if you ever want a different MC budget.
N_MC_EVAL="${N_MC_EVAL:-1000}"

FITTED_JSON="$CM_DIR/ez_classical/fitted_params.json"
EVAL_JSON="$CM_DIR/results/raw/ez_reader_classical_seed1.json"

# ---------------------------------------------------------------------------- #
# Preflight: confirm fitted params exist
# ---------------------------------------------------------------------------- #
echo "================================================================"
echo "Preflight: confirming fitted parameters are available"
echo "================================================================"
if [ ! -f "$FITTED_JSON" ]; then
    echo ">> ERROR: $FITTED_JSON not found."
    echo "   Run run_fit_and_eval.sh first to produce it."
    exit 1
fi
echo ">> Found $FITTED_JSON"
python - <<'PYEOF'
import json
with open("src_v2/paper_experiments/exp01_main_comparison/complete_metrics/"
          "ez_classical/fitted_params.json") as f:
    d = json.load(f)
print(f"   fit settings: n_sentences={d.get('n_sentences')}  "
      f"n_mc={d.get('n_mc')}  n_evals={d.get('n_evals')}")
print(f"   loss: {d.get('loss_default'):.4f} -> {d.get('loss_fitted'):.4f}  "
      f"(improvement "
      f"{100.0*(d['loss_default']-d['loss_fitted'])/d['loss_default']:.2f}%)")
print(f"   fitted parameters (vs defaults):")
for k, v in d['fitted_params'].items():
    dv = d['default_params'][k]
    pct = 100*(v - dv)/abs(dv) if abs(dv) > 1e-9 else 0
    print(f"     {k:<22s} {dv:>10.4f} -> {v:>10.4f}  ({pct:+.1f}%)")
PYEOF
echo ""

# ---------------------------------------------------------------------------- #
# Step 1: re-evaluate with N=1000 MC + fitted params (overwrites old JSON)
# ---------------------------------------------------------------------------- #
echo "================================================================"
echo "Step 1: classical E-Z Reader eval with FITTED params, N=$N_MC_EVAL MC"
echo "        (overwrites $EVAL_JSON)"
echo "================================================================"
t0=$(date +%s)
$PY "$CM_DIR/04_eval_ez_classical.py" --num_runs "$N_MC_EVAL" --force
rc=$?
elapsed=$(( $(date +%s) - t0 ))
echo ""
echo ">> Step 1 finished in ${elapsed}s with exit code $rc"
if [ "$rc" -ne 0 ]; then
    echo "   ERROR: eval script failed. Aborting before aggregate."
    exit "$rc"
fi
echo ""

# ---------------------------------------------------------------------------- #
# Step 2: re-aggregate the comparison CSVs so the table reflects the new EZ row
# ---------------------------------------------------------------------------- #
echo "================================================================"
echo "Step 2: aggregate -> CSV (updates comparison_results_complete.csv)"
echo "================================================================"
$PY "$CM_DIR/aggregate.py"
echo ""

echo ">> Done at $(date)"
echo ">> Outputs:"
echo "    $EVAL_JSON           (N=$N_MC_EVAL MC, fitted params)"
echo "    $CM_DIR/results/comparison_results_complete.csv"
echo "    $CM_DIR/results/per_seed_metrics_complete.csv"
