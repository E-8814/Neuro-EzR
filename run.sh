#!/bin/bash
#sbatch run.sh
#SBATCH --job-name=train
#SBATCH --partition=GPU
#SBATCH --gres=gpu:rtx:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate neuro_ezr

#python src_v2/lm_train/train_hybrid_v4c_saccade_noise_geco.py
#python src_v2/lm_train/train_hybrid_v4c_full_geco.py
python src_v2/lm_train/train_hybrid_v4c_v2_geco.py
python src_v2/lm_train/train_hybrid_v4c_v2_saccade_noise_geco.py 
python src_v2/lm_train/train_hybrid_v4c_v2_full_geco.py