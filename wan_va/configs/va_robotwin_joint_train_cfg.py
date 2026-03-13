from easydict import EasyDict

from .va_robotwin_video_train_cfg import va_robotwin_video_train_cfg


va_robotwin_joint_train_cfg = EasyDict(__name__='Config: VA robotwin joint WAN+expert train')
va_robotwin_joint_train_cfg.update(va_robotwin_video_train_cfg)

# Joint training: WAN keeps video objective, action expert is trained with detached WAN condition.
va_robotwin_joint_train_cfg.save_root = './train_out/robotwin_joint'
va_robotwin_joint_train_cfg.train_video_only = True
va_robotwin_joint_train_cfg.eval_interval = 0

# WAN optimizer
va_robotwin_joint_train_cfg.learning_rate = 1e-5
va_robotwin_joint_train_cfg.beta1 = 0.9
va_robotwin_joint_train_cfg.beta2 = 0.95
va_robotwin_joint_train_cfg.weight_decay = 0.1
va_robotwin_joint_train_cfg.warmup_steps = 10

# Action expert optimizer
va_robotwin_joint_train_cfg.action_expert_learning_rate = 2e-4
va_robotwin_joint_train_cfg.action_expert_beta1 = 0.9
va_robotwin_joint_train_cfg.action_expert_beta2 = 0.95
va_robotwin_joint_train_cfg.action_expert_weight_decay = 0.05
va_robotwin_joint_train_cfg.action_expert_warmup_steps = 100

va_robotwin_joint_train_cfg.batch_size = 1
va_robotwin_joint_train_cfg.gradient_accumulation_steps = 16
va_robotwin_joint_train_cfg.num_steps = 5000
va_robotwin_joint_train_cfg.save_interval = 1000
va_robotwin_joint_train_cfg.dataset_init_worker = 1

# observation.state in your dataset info is 16-dim
va_robotwin_joint_train_cfg.robot_state_dim = 16
va_robotwin_joint_train_cfg.action_expert_velocity_dim = 48
va_robotwin_joint_train_cfg.action_expert_hidden_dim = 768
va_robotwin_joint_train_cfg.action_expert_num_heads = 12
va_robotwin_joint_train_cfg.action_expert_num_layers = 4
va_robotwin_joint_train_cfg.action_expert_dropout = 0.1
va_robotwin_joint_train_cfg.action_expert_ffn_mult = 4
va_robotwin_joint_train_cfg.action_expert_timestep_dim = 256
va_robotwin_joint_train_cfg.mip_t_star = 0.9
va_robotwin_joint_train_cfg.mip_loss_weight_step0 = 1.0
va_robotwin_joint_train_cfg.mip_loss_weight_step1 = 1.0
