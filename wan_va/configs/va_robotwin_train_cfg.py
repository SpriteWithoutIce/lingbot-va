# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg
import os

va_robotwin_train_cfg = EasyDict(__name__='Config: VA robotwin train')
va_robotwin_train_cfg.update(va_robotwin_cfg)

# va_robotwin_train_cfg.resume_from = '/robby/share/Robotics/lilin1/code/Wan_VA_Release/train_out/checkpoints/checkpoint_step_10'

va_robotwin_train_cfg.dataset_path = '/home/jwhe/linyihan/datasets/lerobot_robotwin_eef_clean_50/adjust_bottle-demo_clean_collect_200-50'
va_robotwin_train_cfg.empty_emb_path = os.path.join(va_robotwin_train_cfg.dataset_path, 'empty_emb.pt')
va_robotwin_train_cfg.enable_wandb = False
va_robotwin_train_cfg.load_worker = 16
va_robotwin_train_cfg.save_interval = 1000
va_robotwin_train_cfg.gc_interval = 50
va_robotwin_train_cfg.cfg_prob = 0.1

# Training parameters
va_robotwin_train_cfg.learning_rate = 1e-5
va_robotwin_train_cfg.beta1 = 0.9
va_robotwin_train_cfg.beta2 = 0.95
va_robotwin_train_cfg.weight_decay = 0.1
va_robotwin_train_cfg.warmup_steps = 10
va_robotwin_train_cfg.batch_size = 1 
va_robotwin_train_cfg.gradient_accumulation_steps = 16
va_robotwin_train_cfg.num_steps = 2000


# Video-only WAN finetuning (no action branch)
va_robotwin_train_cfg.train_video_only = False
# Pre-extracted latent root under dataset_path (default keeps existing latents/)
va_robotwin_train_cfg.latent_subdir = 'latents'
# Model selector: 'wan_va' (joint video+action) or 'wan_video_finetune' (video only)
va_robotwin_train_cfg.transformer_model_name = 'wan_va'

# transformer init source: 'lingbot_va' or 'wan_official'
va_robotwin_train_cfg.transformer_source = 'lingbot_va'
# used when transformer_source='wan_official'
va_robotwin_train_cfg.wan_official_ckpt_path = '/home/jwhe/linyihan/Wan2.2-TI2V-5B'
