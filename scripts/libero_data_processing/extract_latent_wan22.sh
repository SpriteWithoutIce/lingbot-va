# for name in libero_goal_dataset; do
#   python scripts/libero_data_processing/extract_latents_wan22.py \
#     --wan22_root /Wan2.2 \
#     --ckpt_dir /Wan2.2-TI2V-5B \
#     --dataset_path /datasets/libero_lingbot/$name
# done
export CUDA_VISIBLE_DEVICES=2
python scripts/libero_data_processing/extract_latents_wan22.py \
  --dataset_path /datasets/libero_lingbot/libero_10_dataset \
  --vae_dir /lingbot-va-base/vae \
  --height 256 \
  --width 256 \
  --wrist_height 256 \
  --wrist_width 256 \