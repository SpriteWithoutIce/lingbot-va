#!/usr/bin/bash

set -x
umask 007

NGPU=${NGPU:-"2"}
MASTER_PORT=${MASTER_PORT:-"29502"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_action_expert_train"}
SAVE_ROOT=${SAVE_ROOT:-"./train_out/robotwin_action_expert"}

export SWANLAB_API_KEY=DrS0mShJWfRVGsCtt4ewx
export SWANLAB_PROJECT=lingbot-va

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT}
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export TORCH_CPP_LOG_LEVEL=ERROR
export TOKENIZERS_PARALLELISM=false

mkdir -p logs
LOGFILE=logs/train_action_expert_$(date +"%Y%m%d_%H%M%S").log

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
python -m torch.distributed.run \
    --nproc_per_node=${NGPU} \
    --master_port ${MASTER_PORT} \
    -m wan_va.train_action_expert --config-name ${CONFIG_NAME} --save-root "${SAVE_ROOT}" \
    --run-name action_expert_train_$(date +"%Y%m%d_%H%M%S") \
    "$@" \
    > ${LOGFILE} 2>&1
