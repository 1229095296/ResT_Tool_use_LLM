#!/bin/bash
set -e

# 数据参数
DATA_ARGS=" \
    --model /path/to/project/github.com/ORGANIZATION/PROJECT.git/huggingface.co/meta-llama/Llama-3.2-3B-Instruct \
    --dataset /path/to/project/ToolRL-main/dataset/rlla_4k_raw/rlla_sft.json \
    --split_dataset_ratio 0.01 \
"
# 训练参数
TRAIN_ARGS=" \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_steps 10 \
    --logging_steps 1 \
    --max_length 8192 \
    --dataloader_num_workers 4 \
    --use_liger_kernel true \
    --load_from_cache_file false \
    --attn_impl flash_attn \
    --deepspeed zero2 \
    --dataset_num_proc 8  \
    --add_version false \
    --loss_scale hermes \






"
# 学习率参数
LR_ARGS=" \
    --learning_rate 5e-6 \
    --warmup_ratio 0.05 \
"

# 检查点参数
CHECKPOINT_ARGS=" \
    --output_dir /path/to/project/github.com/ORGANIZATION/PROJECT.git/checkpoints/new/llama \
    --save_steps 500 \
    --save_total_limit 2 \
"

# 合并所有参数
ALL_ARGS=" \
    ${DATA_ARGS} \
    ${TRAIN_ARGS} \
    ${LR_ARGS} \
    ${CHECKPOINT_ARGS} \
"
LAUNCHER="swift sft "

# 环境变量
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=eth0
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=psxkf4nxr84yez59-worker-0.psxkf4nxr84yez59.hadoop-djst.svc.cluster.local
export MASTER_PORT=29501
export WORLD_SIZE=8
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 执行训练
CMD="${LAUNCHER} ${ALL_ARGS}"
echo "$CMD"
eval "$CMD"  
