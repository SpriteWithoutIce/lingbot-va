export CUDA_VISIBLE_DEVICES=2
START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
# 训练 checkpoint 目录（其下需有 transformer/ 子目录）；不设则用 config 里的 base 模型
CKPT=${CKPT:-/home/jwhe/linyihan/lingbot-va/train_out/robotwin}
save_root=${SAVE_ROOT:-visualization/}
mkdir -p $save_root

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port $START_PORT \
    --save_root $save_root \
    --ckpt "$CKPT"


