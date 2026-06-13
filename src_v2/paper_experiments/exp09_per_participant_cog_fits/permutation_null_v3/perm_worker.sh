#!/bin/bash
# Additional perm_v3 worker — safe to run many in parallel (the runner
# picks random missing splits and writes atomically, one JSON per split).
#
# Usage (from the repo root):
#   sbatch --job-name=ez_perm_w1 src_v2/paper_experiments/exp09_per_participant_cog_fits/permutation_null_v3/perm_worker.sh
#
#SBATCH --job-name=ez_perm_w
#SBATCH --partition=GPU
#SBATCH --gres=gpu:rtx:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /home/u384661/Neuro_EZR
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neuro_ezr

python -u src_v2/paper_experiments/exp09_per_participant_cog_fits/permutation_null_v3/perm_v3.py
echo "WORKER DONE"
