# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
from pathlib import Path
import wandb
import swanlab

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from modules.utils import (
    load_transformer,
)
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from dataset import MultiLatentLeRobotDataset
import gc
from datetime import datetime


class Trainer:
    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            # wandb.login(host=os.environ['WANDB_BASE_URL'], key=os.environ['WANDB_API_KEY'])
            # self.wandb = wandb
            # self.wandb.init(
            #     # entity=os.environ["WANDB_TEAM_NAME"],
            #     project=os.getenv("WANDB_PROJECT", "va_robotwin"),
            #     # dir=log_dir,
            #     config=config,
            #     mode="online",
            #     name='test_lln'
            #     # name=os.path.basename(os.path.normpath(job_config.job.dump_folder))
            # )
            swanlab.login(api_key=os.getenv("SWANLAB_API_KEY", None))
            self.wandb = swanlab
            self.wandb.init(
                project=os.getenv("SWANLAB_PROJECT", "va_robotwin"),
                config=dict(config),
                name="test_lln",
            )
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')

        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
            attn_mode="flex",
        )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(config=config)
        if config.rank == 0:
            print(f"[Train] Train dataset length: {len(train_dataset)}", flush=True)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42
        ) if config.world_size > 1 else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None), 
            num_workers=config.load_worker,
            sampler=train_sampler,
        )

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(config.save_root) / f"checkpoints_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        # if hasattr(config, 'resume_from') and config.resume_from:
        #     self._load_training_state(config.resume_from)
    
    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare input dict following infer code pattern from wan_va_server.py."""
        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=0.5)
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0)

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
        }
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(self.device)
        return input_dict

    def compute_loss(self,
        input_dict,
        pred
    ):
        latent_pred, action_pred = pred
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and compute mask per frame
        latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        # Frame-wise action loss calculation       
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        mask = input_dict['action_dict']['actions_mask'].float()
        raw_mse_mean = (action_loss * mask).sum() / (mask.sum() + 1e-6)
        
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps, raw_mse_mean.detach()

    def train_epoch(self):
        self.transformer.train()

        # Use manual progress bar control to only update on optimizer steps
        progress_bar = tqdm(
            total=len(self.train_loader),
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_raw_mse_losses = []

        for batch_idx, batch in enumerate(self.train_loader):
            if self.config.rank == 0:
                episode_indices = batch.get('episode_index', None)
                # print(f"[batch {batch_idx}] episode_index: {episode_indices}", flush=True)
            # transfer batch to device
            batch = self.convert_input_format(batch)

            input_dict = self._prepare_input_dict(batch)

            should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader)
            
            if not should_sync:
                self.transformer.set_requires_gradient_sync(False)
            else:
                self.transformer.set_requires_gradient_sync(True)

            output = self.transformer(input_dict, train_mode=True)
            latent_loss, action_loss, raw_mse = self.compute_loss(input_dict, output)
            loss = latent_loss + action_loss # Scale loss for accumulation

            loss.backward()

            # Accumulate losses for logging
            accumulated_latent_losses.append(latent_loss.detach())
            accumulated_action_losses.append(action_loss.detach())
            accumulated_raw_mse_losses.append(raw_mse)

            # Only update weights after accumulating gradients
            if should_sync:
                total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                lr = self.lr_scheduler.get_last_lr()[0]

                # Average accumulated losses
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                raw_mse_show = dist_mean(torch.stack(accumulated_raw_mse_losses).mean()).detach().cpu().item()

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_raw_mse_losses = []

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    # Manually increment counter, set_postfix will refresh automatically
                    progress_bar.n += self.gradient_accumulation_steps
                    progress_bar.set_postfix({
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}'
                    })
                    if self.config.enable_wandb:
                        self.wandb.log({
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'loss_metrics/action_raw_mse': raw_mse_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                        }, step=self.step)
                self.step += 1
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

                eval_interval = getattr(self.config, 'eval_interval', 0)
                eval_num_samples = getattr(self.config, 'eval_num_samples', 4)
                if eval_interval > 0 and self.step % eval_interval == 0:
                    # 从当前 loader 收集若干 batch 做开环测试
                    eval_batches = []
                    for _b in self.train_loader:
                        eval_batches.append(_b)
                        if len(eval_batches) >= eval_num_samples:
                            break
                    self.run_open_loop_eval(eval_batches)

        progress_bar.close()

    @torch.no_grad()
    def run_open_loop_eval(self, eval_batches):
        """从训练数据取若干 batch，GT latent 作为干净条件，对 action 做完整去噪，计算 action MSE。
        去噪逻辑与 wan_va_server._infer 对齐：
          - action frame 0 置零作为条件帧（timestep=0）
          - 每步后重置 frame 0 = 0，并应用 action_mask
          - 直接迭代 scheduler.timesteps（extra_one_step 保证最后一步 sigma_=0）
          - 最终结果取 noisy_actions（scheduler.step 累积结果）
        """
        self.transformer.eval()

        eval_action_scheduler = FlowMatchScheduler(
            shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        num_steps = getattr(self.config, 'eval_action_num_inference_steps', 10)
        eval_action_scheduler.set_timesteps(num_steps)
        # 直接用 scheduler.timesteps，不额外 pad 0（无 KV cache 时不需要）
        action_timesteps = eval_action_scheduler.timesteps

        total_mse = 0.0
        total_count = 0

        for batch in eval_batches:
            batch = self.convert_input_format(batch)
            latent_gt   = batch['latents']       # (B, C, F, H, W)
            action_gt   = batch['actions']       # (B, C, FA, NA, 1)
            text_emb    = batch['text_emb']
            action_mask = batch['actions_mask']  # (B, C, FA, NA, 1) bool

            B, C, LF, H, W = latent_gt.shape
            _, CA, FA, NA, _ = action_gt.shape
            patch_f, patch_h, patch_w = self.patch_size

            latent_grid_id = get_mesh_id(
                LF // patch_f, H // patch_h, W // patch_w,
                t=0, f_w=1, f_shift=0, action=False
            ).to(self.device)[None].repeat(B, 1, 1)

            action_grid_id = get_mesh_id(
                FA, NA, 1, t=1, f_w=1, f_shift=0, action=True
            ).to(self.device)[None].repeat(B, 1, 1)

            # latent 全部帧干净（timestep=0），对应 server latent_cond 逻辑
            latent_dict = dict(
                timesteps=torch.zeros(B, LF, device=self.device),
                cond_timesteps=torch.zeros(B, LF, device=self.device),
                noisy_latents=latent_gt.to(self.dtype),
                latent=latent_gt.to(self.dtype),
                text_emb=text_emb,
                grid_id=latent_grid_id,
            )

            # action 从纯噪声出发
            noisy_actions = torch.randn_like(action_gt, dtype=self.dtype)
            # frame 0 = 条件帧（置零）；应用 action_mask（未使用 channel → 0）
            noisy_actions[:, :, 0:1] = 0.0
            noisy_actions = noisy_actions * action_mask.float()

            for t in action_timesteps:
                # frame 0 的 timestep = 0（clean cond），其余帧 = t
                ts = torch.full((B, FA), t.item(), device=self.device)
                ts[:, 0] = 0.0

                action_dict = dict(
                    timesteps=ts,
                    cond_timesteps=torch.zeros(B, FA, device=self.device),
                    noisy_latents=noisy_actions,
                    latent=torch.zeros_like(noisy_actions),
                    text_emb=text_emb,
                    actions_mask=action_mask,
                    grid_id=action_grid_id,
                )
                input_dict = {
                    'latent_dict': latent_dict,
                    'action_dict': action_dict,
                    'chunk_size': 1,
                    'window_size': 64,
                }
                _, action_pred_raw = self.transformer(input_dict, train_mode=True)
                action_pred = rearrange(action_pred_raw, 'b (f n) c -> b c f n 1', f=FA)  # FA = latent action frames

                # scheduler.step：extra_one_step 保证最后一步 sigma_=0（直接到干净样本）
                noisy_actions = eval_action_scheduler.step(
                    action_pred, t, noisy_actions, return_dict=False)
                # 每步后重置 frame 0 = 0；重新应用 mask（与 server 一致）
                noisy_actions[:, :, 0:1] = 0.0
                noisy_actions = noisy_actions * action_mask.float()

            # 最终结果 = noisy_actions（所有 scheduler.step 后的累积去噪结果）
            mask = action_mask.float()
            mse = F.mse_loss(noisy_actions.float(), action_gt.float(), reduction='none')
            mse_val = (mse * mask).sum() / (mask.sum() + 1e-6)
            total_mse += mse_val.item()
            total_count += 1

        avg_mse = total_mse / max(total_count, 1)

        # 训练 loss 在 self.step += 1 之前记录（即 step N-1），
        # eval 在 step += 1 之后触发，用 self.step - 1 对齐到同一个 step 轴
        log_step = self.step - 1
        if self.config.rank == 0:
            logger.info(f"[OpenLoop Eval step {log_step}] action MSE = {avg_mse:.6f}")
            if self.config.enable_wandb:
                self.wandb.log({'eval/open_loop_action_mse': avg_mse}, step=log_step)

        self.transformer.train()

    def save_checkpoint(self,):
        """Save model checkpoint in the same format as pretrained model."""
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            # optim_state = get_optimizer_state_dict(
            #         self.transformer, self.optimizer,
            #         options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            #     )

            # Only rank 0 saves the checkpoint
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save transformer in the same format as pretrained model
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")

                # Manually save in diffusers format (outside FSDP context to avoid deadlock)
                # Save model weights
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)

                # Save config (copy from original transformer config and update _name_or_path)
                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

                # # Save optimizer state and training metadata in PyTorch format
                # training_state_path = checkpoint_dir / "training_state.pt"
                # logger.info(f"Saving training state to {training_state_path}")
                # torch.save({
                #     'step': self.step,
                #     'optimizer_state_dict': optim_state,
                #     'config': vars(self.config),
                # }, training_state_path)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            # Synchronize all processes after saving
            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            # Ensure all processes stay synchronized even on error
            if dist.is_initialized():
                dist.barrier()

    def _load_training_state(self, checkpoint_path):
        """Load training state (optimizer + step) after FSDP and optimizer creation."""
        checkpoint_dir = Path(checkpoint_path)
        training_state_path = checkpoint_dir / "training_state.pt"

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        # All ranks load the training state directly
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)

        # All ranks load optimizer state (required for FSDP)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=training_state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False)
        )
        self.step = training_state.get('step', 0)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")

        while self.step < self.config.num_steps:
            self.train_epoch()
            if dist.is_initialized():
                dist.barrier()

        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = Trainer(config)
    trainer.train()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()