#!/bin/sh

#SBATCH -o gpu-job-train-2.output
#SBATCH -p PA100q
#SBATCH --gpus-per-node=1

#SBATCH -n 1
#SBATCH -c 16
#SBATCH -w node02

for lr in 1e-3 1e-4 1e-5 2e-3 2e-4 2e-5
do
    echo "lr: "$lr
    for seed in 42
    do 
        echo "seed: "$seed
        CUDA_VISIBLE_DEVICES=3 python amex_train.py \
                                        --lrate $lr \
                                        --sampling "10pct" \
                                        --data_type "original" \
                                        --num_nodes 223 \
                                        --es_patience 3 \
                                        --seed $seed \
                                        --train \
                                        --test \
                                        --predict \
                                        --submit \
                                        --batch_size 128 \
                                        --num_workers 16 \
                                        --feature_w 0.01\
                                        --fcst_w 1\
                                        --recon_w 0.5\
                                        --att_w 0.01\
                                        --emb_version "v2"\
                                        --remark "only forecast loss"\
                                        --epochs 20 
    done
done

