# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# LIBERO-Spatial: single-arm 7-dim action, 2 cameras (image + wrist_image).
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_libero_spatial_cfg = EasyDict(__name__='Config: VA LIBERO Spatial')
va_libero_spatial_cfg.update(va_shared_cfg)

va_libero_spatial_cfg.infer_mode = 'server'

# Pretrained base (default for evaluation; override with env or train config)
va_libero_spatial_cfg.wan22_pretrained_model_name_or_path = "/home/jwhe/linyihan/lingbot-va-base"

va_libero_spatial_cfg.attn_window = 72
va_libero_spatial_cfg.frame_chunk_size = 4
va_libero_spatial_cfg.env_type = 'libero'

va_libero_spatial_cfg.height = 256
va_libero_spatial_cfg.width = 256
va_libero_spatial_cfg.action_dim = 30
va_libero_spatial_cfg.action_per_frame = 4
va_libero_spatial_cfg.obs_cam_keys = [
    'observation.images.image',
    'observation.images.wrist_image',
]
va_libero_spatial_cfg.guidance_scale = 5
va_libero_spatial_cfg.action_guidance_scale = 1

va_libero_spatial_cfg.num_inference_steps = 20
va_libero_spatial_cfg.video_exec_step = -1
va_libero_spatial_cfg.action_num_inference_steps = 50

va_libero_spatial_cfg.snr_shift = 5.0
va_libero_spatial_cfg.action_snr_shift = 1.0

va_libero_spatial_cfg.used_action_channel_ids = list(range(7))
inverse_used = [7] * va_libero_spatial_cfg.action_dim
for i, j in enumerate(va_libero_spatial_cfg.used_action_channel_ids):
    inverse_used[j] = i
va_libero_spatial_cfg.inverse_used_action_channel_ids = inverse_used

# norm_stat: 30-dim (first 8 used for 7 pose + 1 gripper, rest zero-padded in data)
va_libero_spatial_cfg.action_norm_method = 'quantiles'
va_libero_spatial_cfg.norm_stat = {
    "q01": [
        -0.7044642567634583,
        -0.8008928298950195,
        -0.9375,
        -0.11464285850524902,
        -0.1639285683631897,
        -0.2239285707473755,
        -1.0
    ] + [0.0] * 23,
    "q99": [
        0.9375,
        0.8678571581840515,
        0.9375,
        0.13178572058677673,
        0.19285714626312256,
        0.335357129573822,
        1.0
    ] + [1.0] * 23,
}
