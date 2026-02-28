# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_libero_spatial_cfg import va_libero_spatial_cfg
import os

va_libero_spatial_train_cfg = EasyDict(__name__='Config: VA LIBERO Spatial train')
va_libero_spatial_train_cfg.update(va_libero_spatial_cfg)

# Load transformer from pretrained base (no optimizer state)
va_libero_spatial_train_cfg.wan22_pretrained_model_name_or_path = "/home/jwhe/linyihan/lingbot-va-base"

# Dataset: libero_spatial_dataset (must have meta/episodes.jsonl with action_config, latents/, data/, videos/)
va_libero_spatial_train_cfg.dataset_path = '/home/jwhe/linyihan/datasets/libero_lingbot/libero_spatial_dataset'
va_libero_spatial_train_cfg.empty_emb_path = os.path.join(
    va_libero_spatial_train_cfg.dataset_path, 'empty_emb.pt'
)

va_libero_spatial_train_cfg.enable_wandb = True
va_libero_spatial_train_cfg.load_worker = 0
va_libero_spatial_train_cfg.save_interval = 250
va_libero_spatial_train_cfg.gc_interval = 10
va_libero_spatial_train_cfg.cfg_prob = 0.1

va_libero_spatial_train_cfg.learning_rate = 1e-5
va_libero_spatial_train_cfg.beta1 = 0.9
va_libero_spatial_train_cfg.beta2 = 0.95
va_libero_spatial_train_cfg.weight_decay = 0.1
va_libero_spatial_train_cfg.warmup_steps = 10
va_libero_spatial_train_cfg.batch_size = 1
va_libero_spatial_train_cfg.gradient_accumulation_steps = 16
va_libero_spatial_train_cfg.num_steps = 500

# 开环测试
va_libero_spatial_train_cfg.eval_interval = 10           # 每 N step 做一次开环测试（0 表示关闭）
va_libero_spatial_train_cfg.eval_num_samples = 1          # 用多少个 batch 做 eval
va_libero_spatial_train_cfg.eval_action_num_inference_steps = 50  # action 去噪步数
