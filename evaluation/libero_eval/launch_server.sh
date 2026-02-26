#!/bin/bash
# Start lingbot-va policy server for LIBERO evaluation.
# Run this in one terminal, then run launch_client.sh in another.
#
# 使用训练得到的 checkpoint（仅含 transformer）时，设置 CKPT 为对应目录，例如：
#   CKPT=./train_out/libero_all/checkpoints/checkpoint_step_800 bash evaluation/libero_eval/launch_server.sh
# 此时 transformer 从 CKPT/transformer 加载，VAE/tokenizer/text_encoder 仍从 config 中的 base 路径加载。

cd "$(dirname "$0")/../.."

START_PORT=${START_PORT:-29536}
MASTER_PORT=${MASTER_PORT:-29561}
CKPT=${CKPT:-/home/jwhe/linyihan/lingbot-va/train_out/libero_spatial_posttrain/checkpoints_20260226_131554/checkpoint_step_500}
SAVE_ROOT=${SAVE_ROOT:-./evaluation/libero_eval/server_out}
mkdir -p "$SAVE_ROOT"

# EXTRA_ARGS=()
# [[ -n "${CKPT:-}" ]] && EXTRA_ARGS+=(--ckpt "$CKPT")

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "$MASTER_PORT" \
    wan_va/wan_va_server.py \
    --config-name libero_spatial \
    --port "$START_PORT" \
    --save_root "${SAVE_ROOT:-}" \
    --ckpt "$CKPT"
