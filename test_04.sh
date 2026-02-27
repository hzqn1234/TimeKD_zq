#!/bin/sh

#### SBATCH -o gpu-job-%j.output
#SBATCH -o test_04.output
#SBATCH -p PA100q
# SBATCH --gres=gpu:4

#SBATCH -n 1
#SBATCH -c 24
#SBATCH -w node05

CUDA_VISIBLE_DEVICES=0 \
python -u test_04.py

