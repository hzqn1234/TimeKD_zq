#!/bin/sh

#SBATCH -o gpu-job.output

#SBATCH -p NV100q

#SBATCH --gres=gpu:0

#SBATCH -n 1
#SBATCH -c 4
#SBATCH -w node18

echo 'S1'
python S1_denoise.py

echo 'S4'
python S4_feature_combined.py

echo 'S6_0'
python S6_0_NN_PreProcess_0.py

echo 'Done'
