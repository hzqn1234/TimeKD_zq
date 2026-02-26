#!/bin/sh

#### SBATCH -o gpu-job-%j.output
#SBATCH -o gpu-job-store-emb-1.output
#SBATCH -p PA100q
#SBATCH --gres=gpu:3

#SBATCH -n 1
#SBATCH -c 4
#SBATCH -w node02

# Define the specific GPUs you want to use as a space-separated string (NOT an array).
# You can change this to GPUS="0" to run on a single GPU, or GPUS="0 1 2" for multiple.
GPUS="0 1 7" 

# Define parameters as variables so they can be reused for the path
DATA_TYPE="original"
SAMPLING="10pct"
EMB_DIR="../../000_data/amex/${DATA_TYPE}_${SAMPLING}/emb_06"

# === NEW CLEANUP LOGIC ===
echo "Ensuring directory exists and clearing previous embedding files in ${EMB_DIR}..."
mkdir -p "$EMB_DIR"
rm -f "$EMB_DIR"/*.h5
# =========================

# Calculate the total number of chunks by counting the items in the GPUS string
TOTAL_CHUNKS=0
for gpu in $GPUS; do
    TOTAL_CHUNKS=$((TOTAL_CHUNKS + 1))
done

echo "Running $TOTAL_CHUNKS chunks across GPUs: $GPUS"

# OS-Level Parallelism: Launch independent processes simultaneously
i=0
for GPU_ID in $GPUS; do
    echo "Starting chunk $i on GPU $GPU_ID..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python -u amex_store_emb.py \
            --num_nodes 223 \
            --data_type "$DATA_TYPE" \
            --batch_size 1 \
            --num_workers 8 \
            --model_name "Qwen/Qwen2.5-0.5B" \
            --d_model 896 \
            --max_token_len 2048 \
            --sampling "$SAMPLING" \
            --chunk_id $i \
            --total_chunks $TOTAL_CHUNKS \
            --allow_truncate 0 \
            --l_layers 8 \
            > store_emb_1_chunk_${i}.log 2>&1 &
            
    # Increment the chunk ID index
    i=$((i + 1))
done

# Wait for all background processes to finish
wait

echo "All $TOTAL_CHUNKS train embedding chunks finished successfully!"



