# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_libero_spatial_cfg import va_libero_spatial_cfg
from .va_libero_spatial_train_cfg import va_libero_spatial_train_cfg
from .va_libero_all_train_cfg import va_libero_all_train_cfg
from .va_libero_all_cfg import va_libero_all_cfg
from .va_libero_object_cfg import va_libero_object_cfg
from .va_libero_object_train_cfg import va_libero_object_train_cfg
from .va_libero_goal_cfg import va_libero_goal_cfg
from .va_libero_goal_train_cfg import va_libero_goal_train_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'libero_spatial': va_libero_spatial_cfg,
    'libero_spatial_train': va_libero_spatial_train_cfg,
    'libero_all_train': va_libero_all_train_cfg,
    'libero_all': va_libero_all_cfg,
    'libero_object': va_libero_object_cfg,
    'libero_object_train': va_libero_object_train_cfg,
    'libero_goal': va_libero_goal_cfg,
    'libero_goal_train': va_libero_goal_train_cfg,
}