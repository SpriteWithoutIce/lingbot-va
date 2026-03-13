from easydict import EasyDict

from .va_robotwin_video_train_cfg import va_robotwin_video_train_cfg


va_robotwin_action_expert_train_cfg = EasyDict(
    __name__='Config: VA robotwin action expert train'
)
va_robotwin_action_expert_train_cfg.update(va_robotwin_video_train_cfg)

# Stage-2 training: freeze WAN(video) backbone, train action expert only.
va_robotwin_action_expert_train_cfg.enable_wandb = True
va_robotwin_action_expert_train_cfg.train_video_only = True
va_robotwin_action_expert_train_cfg.save_root = './train_out/robotwin_action_expert'
va_robotwin_action_expert_train_cfg.save_interval = 500
va_robotwin_action_expert_train_cfg.eval_interval = 0

va_robotwin_action_expert_train_cfg.learning_rate = 2e-4
va_robotwin_action_expert_train_cfg.beta1 = 0.9
va_robotwin_action_expert_train_cfg.beta2 = 0.95
va_robotwin_action_expert_train_cfg.weight_decay = 0.05
va_robotwin_action_expert_train_cfg.warmup_steps = 100
va_robotwin_action_expert_train_cfg.batch_size = 1
va_robotwin_action_expert_train_cfg.gradient_accumulation_steps = 8
va_robotwin_action_expert_train_cfg.num_steps = 4000

va_robotwin_action_expert_train_cfg.robot_state_dim = 16
va_robotwin_action_expert_train_cfg.action_expert_velocity_dim = 48
va_robotwin_action_expert_train_cfg.action_expert_hidden_dim = 768
va_robotwin_action_expert_train_cfg.action_expert_num_heads = 12
va_robotwin_action_expert_train_cfg.action_expert_num_layers = 8
va_robotwin_action_expert_train_cfg.action_expert_dropout = 0.1
va_robotwin_action_expert_train_cfg.action_expert_ffn_mult = 4
va_robotwin_action_expert_train_cfg.action_expert_timestep_dim = 256
va_robotwin_action_expert_train_cfg.mip_t_star = 0.9
va_robotwin_action_expert_train_cfg.mip_loss_weight_step0 = 1.0
va_robotwin_action_expert_train_cfg.mip_loss_weight_step1 = 1.0

# WAN stage-1 checkpoint used to produce latent velocity condition.
va_robotwin_action_expert_train_cfg.wan_stage1_source = 'wan_official'
va_robotwin_action_expert_train_cfg.wan_stage1_model_name = 'wan_video_finetune'
va_robotwin_action_expert_train_cfg.wan_stage1_model_path = va_robotwin_action_expert_train_cfg.wan_official_ckpt_path
