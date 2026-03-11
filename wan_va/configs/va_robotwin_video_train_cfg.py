# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robotwin_train_cfg import va_robotwin_train_cfg

va_robotwin_video_train_cfg = EasyDict(__name__='Config: VA robotwin video-only train')
va_robotwin_video_train_cfg.update(va_robotwin_train_cfg)

va_robotwin_video_train_cfg.train_video_only = True
va_robotwin_video_train_cfg.transformer_model_name = 'wan_video_finetune'
# use separately extracted latents from raw RobotWin videos
va_robotwin_video_train_cfg.latent_subdir = 'latents_video_ft'

va_robotwin_video_train_cfg.transformer_source = 'wan_official'
va_robotwin_video_train_cfg.wan_official_ckpt_path = '/home/jwhe/linyihan/Wan2.2-TI2V-5B'
