#!/bin/sh

#### SBATCH -o gpu-job-%j.output
#SBATCH -o gpu-job-store-emb-1.output
#SBATCH -p PA10080q
# SBATCH --gres=gpu:1 

# SBATCH -n 1
#SBATCH -c 8
#SBATCH -w node04

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python amex_store_emb.py \
        --num_nodes 223 \
        --data_type "original" \
        --batch_size 16 \
        --num_workers 8 \
        --model_name "Qwen/Qwen2.5-0.5B" \
        --d_model 896 \
        --max_token_len 4096 \
        --sampling "100pct"



