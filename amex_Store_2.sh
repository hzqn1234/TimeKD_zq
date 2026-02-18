#!/bin/sh

#### SBATCH -o gpu-job-%j.output
#SBATCH -o gpu-job-store-emb-2.output
#SBATCH -p PA10080q
# SBATCH --gres=gpu:1 

# SBATCH -n 1
# SBATCH -c 16
# SBATCH -w node04

CUDA_VISIBLE_DEVICES=6 \
python amex_store_emb.py \
        --num_nodes 223 \
        --data_type "original" \
        --batch_size 2 \
        --num_workers 16 \
        --model_name "Qwen/Qwen2.5-0.5B" \
        --d_model 896 \
        --sampling "10pct"



