#!/bin/sh

#### SBATCH -o gpu-job-%j.output
#SBATCH -o gpu-job-store-emb-3.output
#SBATCH -p NH100q
#SBATCH --gpus-per-node=2

#SBATCH -n 1
#SBATCH -c 16
#SBATCH -w node07

CUDA_VISIBLE_DEVICES=0 \
python amex_store_emb.py \
        --num_nodes 223 \
        --data_type "original" \
        --batch_size 16 \
        --num_workers 16 \
        --sampling "100pct"


# export PYTHONPATH=/path/to/project_root:$PYTHONPATH
# export CUDA_LAUNCH_BLOCKING=1

# data_paths=("ETTm1" "ETTm2" "ETTh1" "ETTh2")
# data_paths=("ETTm1")

# divides=("train" "val")

# device="cuda"
# num_nodes=7
# input_len=96
# output_len_values=(24 36 48 96 192)
# output_len_values=24
# model_name="gpt2"
# d_model=768
# l_layer=12

# for data_path in "ETTm1"; do
#   for divide in "train" "val"; do
#     for output_len in 24; do
#       # log_file="${data_path}_${output_len}_${divide}.log"
#       # nohup python store_emb.py \
#       CUDA_VISIBLE_DEVICES=7 python store_emb.py \
#         --data_path $data_path \
#         --divide $divide \
#         --device $device \
#         --num_nodes $num_nodes \
#         --input_len $input_len \
#         --output_len $output_len \
#         --model_name $model_name \
#         --d_model $d_model \
#         --batch_size 8 \
#         --l_layer $l_layer 2>&1 | tee logfile_store_2_${divide}.txt
#     done
#   done
# done


