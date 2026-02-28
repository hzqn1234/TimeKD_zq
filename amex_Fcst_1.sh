#!/bin/sh

#SBATCH -o gpu-job-train-1.output
#SBATCH -p HPCAIq
#SBATCH --gpus-per-node=1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -w node14

GPU_ID=0
SAMPLING="1pct"
LRs="1e-3"
EMB_version="v7"

for lr in $LRs
do
    echo "==================================================="
    echo "lr: "$lr
    echo "==================================================="
    
    for seed in 42
    do 
        echo "========================================"
        echo "   Stage 1: Teacher Pre-training (seed: $seed)"
        echo "========================================"
        
        # 第一阶段：只关注 Teacher 的拟合 (recon_w=1)，其余全部设0
        CUDA_VISIBLE_DEVICES=$GPU_ID \
        python amex_train.py \
            --stage 1 \
            --lrate $lr \
            --sampling "$SAMPLING" \
            --data_type "original" \
            --num_nodes 223 \
            --es_patience 3 \
            --seed $seed \
            --train \
            --batch_size 128 \
            --num_workers 8 \
            --feature_w 0.0 \
            --fcst_w 0.0 \
            --recon_w 1.0 \
            --att_w 0.0 \
            --distill_w 0.0 \
            --emb_version "$EMB_version" \
            --remark "Stage 1 Pretrain Teacher" \
            --epochs 20 
            
        # 动态获取最新的 Stage 1 保存目录作为 Teacher 模型路径
        # 确保自动捕获含有best_model的准确路径
        TEACHER_DIR=$(ls -td ./logs/Amex/S1_* | head -1)
        
        echo ""
        echo "=> Teacher models generated at: $TEACHER_DIR"
        echo ""

        echo "========================================"
        echo "   Stage 2: Student Distillation (seed: $seed)"
        echo "========================================"
        
        # 第二阶段：冻结 Teacher，开启蒸馏，优化 Student
        # 此时关闭 recon_w，打开 fcst_w, feature_w, att_w, distill_w
        CUDA_VISIBLE_DEVICES=$GPU_ID \
        python amex_train.py \
            --stage 2 \
            --teacher_dir "$TEACHER_DIR" \
            --lrate $lr \
            --sampling "$SAMPLING" \
            --data_type "original" \
            --num_nodes 223 \
            --es_patience 3 \
            --seed $seed \
            --train \
            --test \
            --predict \
            --submit \
            --batch_size 128 \
            --num_workers 8 \
            --feature_w 0.5 \
            --fcst_w 1.0 \
            --recon_w 0.0 \
            --att_w 0.5 \
            --distill_w 1.0 \
            --emb_version "$EMB_version" \
            --remark "Stage 2 Distillation" \
            --epochs 20 
    done
done