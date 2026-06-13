#!/bin/bash
# exp10/exp06 v3: extract per-word features from the v4c_v3 model (GPU),
# then run the dual-ctx analyses and the exp06 variance partition (CPU).
#
# Usage (from the repo root):
#   sbatch --job-name=ez_exp10_v3 src_v2/paper_experiments/exp10_dualctx_specialization/run_v3_slurm.sh
#
#SBATCH --job-name=ez_exp10_v3
#SBATCH --partition=GPU
#SBATCH --gres=gpu:rtx:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /home/u384661/Neuro_EZR

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neuro_ezr

EXP10=src_v2/paper_experiments/exp10_dualctx_specialization
EXP06=src_v2/paper_experiments/exp06_surprisal_decomp

echo "=== [1/3] extract v3 per-word features (GPU) ==="
python -u $EXP10/extract_per_word_features_v3.py

echo "=== [2/3] dual-ctx analyses: cross-prediction + divergence ==="
python -u $EXP10/analyze_dualctx_v3.py

echo "=== [3/3] exp06 variance partition from CSV ==="
python -u $EXP06/compute_decomp_from_csv_v3.py

echo "EXP10_V3 COMPLETE"
