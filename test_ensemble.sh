#!/bin/sh

#### SBATCH -o gpu-job-%j.output
#SBATCH -o S7_ensemble.output
#SBATCH -p PA100q
# SBATCH --gres=gpu:4

#SBATCH -n 1
#SBATCH -c 4
#SBATCH -w node02

CUDA_VISIBLE_DEVICES=0 \
python -u S7_ensemble_thin.py

# CUDA_VISIBLE_DEVICES=2 \
# python -u S7_ensemble.py

