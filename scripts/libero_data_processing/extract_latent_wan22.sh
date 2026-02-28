# for name in libero_goal_dataset; do
#   python scripts/libero_data_processing/extract_latents_wan22.py \
#     --wan22_root /home/jwhe/linyihan/Wan2.2 \
#     --ckpt_dir /home/jwhe/linyihan/Wan2.2-TI2V-5B \
#     --dataset_path /home/jwhe/linyihan/datasets/libero_lingbot/$name
# done
export CUDA_VISIBLE_DEVICES=2
python scripts/libero_data_processing/extract_latents_wan22_robotwin.py \
  --dataset_path /home/jwhe/linyihan/datasets/libero_lingbot/libero_10_dataset \
  --wan22_root /home/jwhe/linyihan/Wan2.2 \
  --ckpt_dir /home/jwhe/linyihan/Wan2.2-TI2V-5B \
  --vae_dir /home/jwhe/linyihan/lingbot-va-base/vae \
  --height 256 \
  --width 256 \
  --wrist_height 256 \
  --wrist_width 256 \