#!/bin/sh

#### SBATCH -o gpu-job-%j.output
#SBATCH -o gpu-job.output
#SBATCH -p V100q
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1

#SBATCH -n 1
#SBATCH -c 2
#SBATCH -w node19

# export PYTHONPATH=/path/to/project_root:$PYTHONPATH
# export CUDA_LAUNCH_BLOCKING=1

seq_len=96
pred_len=24
learning_rate=1e-3
channel=64
d_llm=768
e_layer=2
dropout_n=0.5
batch_size=16
model_name="gpt2"
data_path="ETTm1"
epochs=100

log_path="./Results/Fcst/${data_path}/"
mkdir -p $log_path
log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dn${dropout_n}_bs${batch_size}_e${epochs}.log"
# nohup python train.py \
CUDA_VISIBLE_DEVICES=1 python train.py \
  --data_path $data_path \
  --device cuda \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs $epochs \
  --seed 8888 \
  --channel $channel \
  --head 4 \
  --lrate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer\
  --model_name $model_name \
  --num_workers 10 \
  --d_llm $d_llm 2>&1 | tee logfile_train.txt

# for seq_len in "${seq_lens[@]}"; do 
#   for pred_len in "${pred_lens[@]}"; do
#     for learning_rate in "${learning_rates[@]}"; do
#       for channel in "${channels[@]}"; do
#         for dropout_n in "${dropout_ns[@]}"; do
#           for e_layer in "${e_layers[@]}"; do
#             for batch_size in "${batch_sizes[@]}"; do
#               log_path="./Results/Fcst/${data_path}/"
#               mkdir -p $log_path
#               log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dn${dropout_n}_bs${batch_size}_e${epochs}.log"
#               # nohup python train.py \
#               CUDA_VISIBLE_DEVICES=1 python train.py \
#                 --data_path $data_path \
#                 --device cuda \
#                 --batch_size $batch_size \
#                 --num_nodes 7 \
#                 --seq_len $seq_len \
#                 --pred_len $pred_len \
#                 --epochs $epochs \
#                 --seed 8888 \
#                 --channel $channel \
#                 --head 4 \
#                 --lrate $learning_rate \
#                 --dropout_n $dropout_n \
#                 --e_layer $e_layer\
#                 --model_name $model_name \
#                 --num_workers 10 \
#                 --d_llm $d_llm 2>&1 | tee logfile_train.txt
#             done
#           done
#         done
#       done
#     done
#   done
# done
