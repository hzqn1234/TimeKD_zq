#!/bin/sh

#### SBATCH -o gpu-job-%j.output
#SBATCH -o gpu-job-store-emb-1.output
#SBATCH -p PA10080q
# SBATCH --gres=gpu:1 

# SBATCH -n 1
#SBATCH -c 8
#SBATCH -w node04

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python amex_store_emb.py \
#         --num_nodes 223 \
#         --data_type "original" \
#         --batch_size 16 \
#         --num_workers 8 \
#         --model_name "Qwen/Qwen2.5-0.5B" \
#         --d_model 896 \
#         --max_token_len 4096 \
#         --sampling "100pct"

# OS-Level Parallelism: Launch 4 independent processes simultaneously

# Start Chunk 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 python amex_store_emb.py \
    --num_nodes 223 \
    --data_type "original" \
    --batch_size 16 \
    --num_workers 8 \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --d_model 896 \
    --max_token_len 4096 \
    --sampling "100pct" \
    --chunk_id 0 \
    --total_chunks 4 &

# Start Chunk 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 python amex_store_emb.py \
    --num_nodes 223 \
    --data_type "original" \
    --batch_size 16 \
    --num_workers 8 \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --d_model 896 \
    --max_token_len 4096 \
    --sampling "100pct" \
    --chunk_id 1 \
    --total_chunks 4 &

# Start Chunk 2 on GPU 2
CUDA_VISIBLE_DEVICES=2 python amex_store_emb.py \
    --num_nodes 223 \
    --data_type "original" \
    --batch_size 16 \
    --num_workers 8 \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --d_model 896 \
    --max_token_len 4096 \
    --sampling "100pct" \
    --chunk_id 2 \
    --total_chunks 4 &

# Start Chunk 3 on GPU 3
CUDA_VISIBLE_DEVICES=3 python amex_store_emb.py \
    --num_nodes 223 \
    --data_type "original" \
    --batch_size 16 \
    --num_workers 8 \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --d_model 896 \
    --max_token_len 4096 \
    --sampling "100pct" \
    --chunk_id 3 \
    --total_chunks 4 &

# Wait for all background processes to finish
wait

echo "All 4 GPUs have finished generating embeddings successfully!"

