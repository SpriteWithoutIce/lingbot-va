# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# LIBERO 四套数据联合训练：Spatial + Object + Goal + Long-Horizon(10)
# 将 dataset_path 设为「包含四个子数据集目录」的父目录即可，train 会通过
# recursive_find_file(dataset_path, 'info.json') 自动发现所有 meta/info.json 并合并。
from easydict import EasyDict
from .va_libero_all_cfg import va_libero_all_cfg
import os

va_libero_all_train_cfg = EasyDict(__name__='Config: VA LIBERO All (4 suites) train')
va_libero_all_train_cfg.update(va_libero_all_cfg)

# 预训练 base
va_libero_all_train_cfg.wan22_pretrained_model_name_or_path = "/home/jwhe/linyihan/lingbot-va-base"

# 多数据集：指向包含四个子数据集的父目录，例如：
#   libero_lingbot/
#     libero_spatial_dataset/   (含 meta/info.json, latents/, data/, ...)
#     libero_object_dataset/
#     libero_goal_dataset/
#     libero_10_dataset/
# 每个子目录需与单数据集时的结构一致（meta/episodes.jsonl, action_config, latents 等）。
va_libero_all_train_cfg.dataset_path = '/home/jwhe/linyihan/datasets/libero_lingbot'

# empty_emb 放在父目录一份即可（可从任意子数据集复制：cp libero_spatial_dataset/empty_emb.pt .）
va_libero_all_train_cfg.empty_emb_path = os.path.join(
    va_libero_all_train_cfg.dataset_path, 'empty_emb.pt'
)

va_libero_all_train_cfg.enable_wandb = True
va_libero_all_train_cfg.load_worker = 0
va_libero_all_train_cfg.save_interval = 500
va_libero_all_train_cfg.gc_interval = 50
va_libero_all_train_cfg.cfg_prob = 0.1

va_libero_all_train_cfg.learning_rate = 1e-5
va_libero_all_train_cfg.beta1 = 0.9
va_libero_all_train_cfg.beta2 = 0.95
va_libero_all_train_cfg.weight_decay = 0.1
va_libero_all_train_cfg.warmup_steps = 0
va_libero_all_train_cfg.batch_size = 1
va_libero_all_train_cfg.gradient_accumulation_steps = 32
va_libero_all_train_cfg.num_steps = 2000

# 开环测试
va_libero_all_train_cfg.eval_interval = 50           # 每 N step 做一次开环测试（0 表示关闭）
va_libero_all_train_cfg.eval_num_samples = 1          # 用多少个 batch 做 eval
va_libero_all_train_cfg.eval_action_num_inference_steps = 20  # action 去噪步数