#!/bin/bash
# Pilot launcher for the v4c_v3_dualctx skip-supervision experiment.
#
# Usage (from the repo root):
#   sbatch --job-name=ez_v3_same src_v2/lm_train/run_pilot_v4c_v3.sh same
#   sbatch --job-name=ez_v3_next src_v2/lm_train/run_pilot_v4c_v3.sh next
#
# Variant A (same): first-word clamp removed, word 0 excluded from skip
#                   loss/eval, legacy row alignment.
# Variant B (next): as A, plus race-faithful alignment (row i scored
#                   against word i+1's skip).
#
# Gate (GECO test, words 1..L-1, vs v4c_v2_dualctx seed 42):
#   pass if r_skip > 0.511 with r_TRT/r_FFD/r_Gaze within ~0.02 of
#   0.433 / 0.189 / 0.379.
#
#SBATCH --job-name=ez_v3_pilot
#SBATCH --partition=GPU
#SBATCH --gres=gpu:rtx:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /home/u384661/Neuro_EZR

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neuro_ezr

ALIGN="${1:?usage: sbatch run_pilot_v4c_v3.sh <same|next>}"

python -u src_v2/lm_train/train_hybrid_v4c_v3_dualctx_geco.py \
    --skip_align "$ALIGN" \
    --seed 42
