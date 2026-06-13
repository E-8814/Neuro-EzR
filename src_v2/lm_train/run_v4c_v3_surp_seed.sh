#!/bin/bash
# Multi-seed launcher for v4c_v3_surp training (H3 ablation, v3 skip fix).
#
# Usage (from the repo root):
#   sbatch --job-name=ez_v3_surp_s1 src_v2/lm_train/run_v4c_v3_surp_seed.sh next 1
#
# $1 = skip alignment (same | next)   — paper model uses 'next'
# $2 = seed
#
#SBATCH --job-name=ez_v3_surp
#SBATCH --partition=GPU
#SBATCH --gres=gpu:rtx:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /home/u384661/Neuro_EZR

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neuro_ezr

ALIGN="${1:?usage: sbatch run_v4c_v3_surp_seed.sh <same|next> <seed>}"
SEED="${2:?usage: sbatch run_v4c_v3_surp_seed.sh <same|next> <seed>}"

python -u src_v2/lm_train/train_hybrid_v4c_v3_surp_geco.py \
    --skip_align "$ALIGN" \
    --seed "$SEED"
