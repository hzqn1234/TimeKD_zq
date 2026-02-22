#!/bin/sh

# for lr in 1e-3 1e-4 1e-5 ## 2e-3 2e-4 2e-5
for lr in 1e-3
do
    echo "lr: "$lr
    for seed in 42
    do 
        echo "seed: "$seed
        CUDA_VISIBLE_DEVICES=0 python amex_train.py \
                                        --lrate $lr \
                                        --sampling "1pct" \
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
                                        --emb_version "v4"\
                                        --d_llm 896 \
                                        --remark "emb_v4"\
                                        --epochs 20 
    done
done

