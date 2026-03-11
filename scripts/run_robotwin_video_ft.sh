#!/bin/bash
set -euo pipefail

# 1) Extract raw-video latents into <each_dataset>/latents_video_ft (4 video frames -> 1 latent frame)
# --dataset_path supports either:
#   a) a single dataset root (.../meta/info.json)
#   b) a parent directory containing multiple dataset roots
# python scripts/robotwin_data_processing/extract_robotwin_video_latents.py \
#   --dataset_path /home/jwhe/linyihan/datasets/lerobot_robotwin_eef_clean_50 \
#   --vae_dir /home/jwhe/linyihan/lingbot-va-base/vae \
#   --text_encoder_dir /home/jwhe/linyihan/lingbot-va-base/text_encoder \
#   --tokenizer_dir /home/jwhe/linyihan/lingbot-va-base/tokenizer \
#   --output_subdir latents_video_ft

# 2) Train video-only WAN finetune model (flow-matching velocity field)
torchrun --nproc_per_node=8 wan_va/train.py --config robotwin_video_train

# Note: robotwin_video_train now initializes transformer from wan_official_ckpt_path
# default: /home/jwhe/linyihan/Wan2.2-TI2V-5B
