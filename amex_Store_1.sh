#!/bin/bash
# !/bin/sh

#### SBATCH -o gpu-job-%j.output
#SBATCH -o gpu-job-store-emb-1.output
#SBATCH -p PA10080q
# SBATCH --gres=gpu:1 

# SBATCH -n 1
#SBATCH -c 8
#SBATCH -w node04

# Define the specific GPUs you want to use as a space-separated array.
# You can change this to GPUS=(0) to run on a single GPU without changing any other code!
GPUS=(0 1 2 3) 

# The total number of chunks is automatically the number of GPUs you listed
TOTAL_CHUNKS=${#GPUS[@]}

echo "Running $TOTAL_CHUNKS chunks across GPUs: ${GPUS[@]}"

# OS-Level Parallelism: Launch independent processes simultaneously
for i in "${!GPUS[@]}"; do
    GPU_ID=${GPUS[$i]}
    echo "Starting chunk $i on GPU $GPU_ID..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python amex_store_emb.py \
            --num_nodes 223 \
            --data_type "original" \
            --batch_size 4 \
            --num_workers 8 \
            --model_name "Qwen/Qwen2.5-0.5B" \
            --d_model 896 \
            --max_token_len 4096 \
            --sampling "1pct" \
            --chunk_id $i \
            --total_chunks $TOTAL_CHUNKS 2>&1 &
done
# 2>&1 | tee store_emb_chunk_${i}.log
# Wait for all background processes to finish
wait

echo "All $TOTAL_CHUNKS train embedding chunks finished successfully!"