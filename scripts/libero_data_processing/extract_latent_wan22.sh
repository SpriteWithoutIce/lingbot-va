for name in libero_goal_dataset; do
  python scripts/libero/extract_latents_wan22.py \
    --wan22_root /home/jwhe/linyihan/Wan2.2 \
    --ckpt_dir /home/jwhe/linyihan/Wan2.2-TI2V-5B \
    --dataset_path /home/jwhe/linyihan/datasets/libero_lingbot/$name
done