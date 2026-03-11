export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/jwhe/linyihan/lingbot-va:$PYTHONPATH
python scripts/visualization/visualize_velocity_delta.py \
  --config_name robotwin_video_train \
  --train_ckpt /home/jwhe/linyihan/lingbot-va/train_out/robotwin/checkpoints_20260311_135525/checkpoint_step_100 \
  --base_ckpt_dir /home/jwhe/linyihan/Wan2.2-TI2V-5B \
  --sample_index 0 \
  --timestep_id 500 \
  --out_dir outputs/velocity_delta_viz_train