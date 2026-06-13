#!/bin/bash
# exp11: fair skip comparison — v4c_v3 (5 seeds + no-LM) on GECO/Provo,
# baselines re-scored on the comparable population (words 1..L-1).
#
# Usage (from the repo root):
#   sbatch --job-name=ez_exp11 src_v2/paper_experiments/exp11_skip_fix_eval/run_slurm.sh
#
#SBATCH --job-name=ez_exp11
#SBATCH --partition=GPU
#SBATCH --gres=gpu:rtx:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /home/u384661/Neuro_EZR

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neuro_ezr

EXP=src_v2/paper_experiments/exp11_skip_fix_eval

echo "=== [1/3] v4c_v3 (5 seeds + no-LM variant): GECO test + Provo ==="
python -u $EXP/eval_v4c_v3_seeds.py

echo "=== [2/3] baselines re-score (flat + BERT + OSU) ==="
python -u $EXP/rescore_baselines.py

echo "=== [3/3] aggregate ==="
python -u $EXP/aggregate.py

echo "EXP11 COMPLETE"
