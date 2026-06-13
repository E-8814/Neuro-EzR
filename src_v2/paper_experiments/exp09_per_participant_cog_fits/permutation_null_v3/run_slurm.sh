#!/bin/bash
# exp09 v3: cache frozen-neural features for the v4c_v3 model, then run
# the EXACT permutation null (all 1,716 balanced 7/7 splits), then
# aggregate to the exact p-value.
#
# Usage (from the repo root):
#   sbatch --job-name=ez_perm_v3 src_v2/paper_experiments/exp09_per_participant_cog_fits/permutation_null_v3/run_slurm.sh
#
# Resumable: re-submitting continues from completed splits.
#
#SBATCH --job-name=ez_perm_v3
#SBATCH --partition=GPU
#SBATCH --gres=gpu:rtx:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /home/u384661/Neuro_EZR

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neuro_ezr

EXP=src_v2/paper_experiments/exp09_per_participant_cog_fits/permutation_null_v3

echo "=== [1/3] cache v3 frozen-neural features (idempotent) ==="
python -u $EXP/cache_features_v3.py

echo "=== [2/3] exact permutation null: 1,716 splits (resumable) ==="
python -u $EXP/perm_v3.py

echo "=== [3/3] aggregate exact p ==="
python -u $EXP/aggregate_v3.py

echo "PERM_V3 COMPLETE"
