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

va_libero_spatial_train_cfg.enable_wandb = False
va_libero_spatial_train_cfg.load_worker = 0
va_libero_spatial_train_cfg.save_interval = 500
va_libero_spatial_train_cfg.gc_interval = 10  # 每步清缓存，缓解 80GB 单卡 OOM
# 降低 attention 窗口以省显存（默认 72 在 backward 时易 OOM）
# va_libero_spatial_train_cfg.attn_window = 36
va_libero_spatial_train_cfg.cfg_prob = 0.1

va_libero_spatial_train_cfg.learning_rate = 1e-5
va_libero_spatial_train_cfg.beta1 = 0.9
va_libero_spatial_train_cfg.beta2 = 0.95
va_libero_spatial_train_cfg.weight_decay = 0.1
va_libero_spatial_train_cfg.warmup_steps = 10
va_libero_spatial_train_cfg.batch_size = 1
va_libero_spatial_train_cfg.gradient_accumulation_steps = 1
va_libero_spatial_train_cfg.num_steps = 1000
