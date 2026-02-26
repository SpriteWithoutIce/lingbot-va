#!/usr/bin/bash
# LIBERO-goal 后训练：从 lingbot-va-base 加载 pretrain，在 libero_goal_dataset 上继续训。
# 路径在 wan_va/configs/va_libero_goal_train_cfg.py 中配置；改路径请编辑该文件。

set -x
umask 007

NGPU=${NGPU:-"2"}
MASTER_PORT=${MASTER_PORT:-"29501"}
CONFIG_NAME=${CONFIG_NAME:-"libero_goal_train"}
SAVE_ROOT=${SAVE_ROOT:-"./train_out/libero_goal_posttrain"}

export SWANLAB_API_KEY=DrS0mShJWfRVGsCtt4ewx
export SWANLAB_PROJECT=lingbot-va-libero-goal

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29502
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export TORCH_CPP_LOG_LEVEL=ERROR

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=2,3
mkdir -p logs
LOGFILE=logs/train_libero-goal_$(date +"%Y%m%d_%H%M%S").log

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
python -m torch.distributed.run \
    --nproc_per_node=${NGPU} \
    --master_port ${MASTER_PORT} \
    -m wan_va.train --config-name ${CONFIG_NAME} --save-root "${SAVE_ROOT}" \
    "$@" \
    > ${LOGFILE} 2>&1
