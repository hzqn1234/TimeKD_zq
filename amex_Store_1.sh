#!/bin/sh

#### SBATCH -o gpu-job-%j.output
#SBATCH -o gpu-job-store-emb-1.output
#SBATCH -p RTXA6Kq
# SBATCH --gres=gpu:4

#SBATCH -n 1
#SBATCH -c 24
#SBATCH -w node16

# Define the specific GPUs you want to use as a space-separated string (NOT an array).
# You can change this to GPUS="0" to run on a single GPU, or GPUS="0 1 2" for multiple.
GPUS="0 1 2 3 4" 

# Define parameters as variables so they can be reused for the path
DATA_TYPE="original"
SAMPLING="1pct"
EMB_DIR="../../000_data/amex/${DATA_TYPE}_${SAMPLING}/emb_07"
echo "Embedding output will be saved to: ${EMB_DIR}"

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
# TOTAL_CHUNKS=5

echo "Running $TOTAL_CHUNKS chunks across GPUs: $GPUS"

# OS-Level Parallelism: Launch independent processes simultaneously
i=0
PIDS="" # Variable to store the Process IDs

for GPU_ID in $GPUS; do
    echo "Starting chunk $i on GPU $GPU_ID..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python -u amex_store_emb.py \
            --num_nodes 223 \
            --data_type "$DATA_TYPE" \
            --batch_size 32 \
            --num_workers 8 \
            --model_name "Qwen/Qwen2.5-0.5B" \
            --d_model 896 \
            --max_token_len 2048 \
            --sampling "$SAMPLING" \
            --chunk_id $i \
            --total_chunks $TOTAL_CHUNKS \
            --allow_truncate 0 \
            --l_layers 16 \
            > store_emb_1_chunk_${i}.log 2>&1 &
            
    # Capture the PID of the last background command
    PID=$!
    PIDS="$PIDS $PID"
    
    # Increment the chunk ID index
    i=$((i + 1))
done

# Wait for all background processes to finish and check their exit codes
FAILED=0
for PID in $PIDS; do
    wait $PID
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Error: Process with PID $PID failed with exit code $EXIT_CODE!"
        FAILED=1
    fi
done

# Final status check
if [ $FAILED -ne 0 ]; then
    echo "FAILED: One or more embedding chunks failed. Please check the individual log files."
    exit 1
else
    echo "SUCCESS: All $TOTAL_CHUNKS train embedding chunks finished successfully!"
    exit 0
fi