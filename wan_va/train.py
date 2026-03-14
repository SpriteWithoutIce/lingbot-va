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
            #     entity=os.environ["WANDB_TEAM_NAME"],
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
                name=getattr(config, "run_name", None) or "test_lln",
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
        self.action_expert_learning_rate = float(
            getattr(config, "action_expert_learning_rate", config.learning_rate * 0.1)
        )
        self.freeze_dit_steps = int(getattr(config, "freeze_dit_steps", 0))

        # Optimizer
        action_expert = self._get_action_expert()
        action_expert_param_ids = {id(p) for p in action_expert.parameters()}
        action_expert_params = []
        backbone_params = []
        for p in self.transformer.parameters():
            if not p.requires_grad:
                continue
            if id(p) in action_expert_param_ids:
                action_expert_params.append(p)
            else:
                backbone_params.append(p)
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": config.learning_rate})
        if action_expert_params:
            param_groups.append({"params": action_expert_params, "lr": self.action_expert_learning_rate})
        if not param_groups:
            raise RuntimeError("No trainable parameters found for optimizer.")
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))
        self._apply_freeze_policy()

        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(config=config)
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
        self.mip_t_star = float(getattr(self.config, "mip_t_star", 0.9))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(config.save_root) / f"checkpoints_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.train_loader_iter = None
        # if hasattr(config, 'resume_from') and config.resume_from:
        #     self._load_training_state(config.resume_from)
    
    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        
        return batch

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

    def _text_mask(self, text_emb):
        return (text_emb.abs().sum(dim=-1) > 0)

    def _get_action_expert(self):
        if hasattr(self.transformer, "action_expert"):
            return self.transformer.action_expert
        if hasattr(self.transformer, "module") and hasattr(self.transformer.module, "action_expert"):
            return self.transformer.module.action_expert
        raise AttributeError("Cannot find action_expert on transformer/FSDP wrapper.")

    def _apply_freeze_policy(self):
        freeze_dit = self.step < self.freeze_dit_steps
        action_expert = self._get_action_expert()
        action_expert_param_ids = {id(p) for p in action_expert.parameters()}
        for p in self.transformer.parameters():
            if id(p) in action_expert_param_ids:
                p.requires_grad_(True)
            else:
                p.requires_grad_(not freeze_dit)

    @torch.no_grad()
    def _prepare_video_input(self, batch_dict):
        video_dict = self._add_noise(
            latent=batch_dict['latents'],
            train_scheduler=self.train_scheduler_latent,
            action_mask=None,
            action_mode=False,
            noisy_cond_prob=0.5,
        )
        video_dict["text_emb"] = batch_dict["text_emb"]
        video_dict["encoder_attention_mask"] = self._text_mask(batch_dict["text_emb"])
        return video_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device)#.to(self.dtype)
        return input_dict

    def _compute_video_loss(self, video_dict, video_pred_seq):
        video_pred = data_seq_to_patch(
            self.patch_size, video_pred_seq,
            video_dict['targets'].shape[-3], video_dict['targets'].shape[-2],
            video_dict['targets'].shape[-1], batch_size=video_pred_seq.shape[0]
        )
        Bn, Fn = video_dict['timesteps'].shape
        weight = self.train_scheduler_latent.training_weight(
            video_dict['timesteps'].flatten()
        ).reshape(Bn, Fn)
        loss = F.mse_loss(
            video_pred.float(),
            video_dict['targets'].float().detach(),
            reduction='none',
        )
        loss = loss * weight[:, None, :, None, None]
        loss = loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        loss = (loss.sum(dim=1) / (torch.ones_like(loss).sum(dim=1) + 1e-6)).mean()
        return loss, video_pred

    def _compute_action_mip_loss(self, batch_dict, video_velocity):
        actions = batch_dict["actions"].to(self.dtype)
        actions_mask = batch_dict["actions_mask"].to(self.dtype)
        states = batch_dict["states"].to(self.dtype)
        states_mask = batch_dict["states_mask"].to(self.dtype)
        B, _, F, N, _ = actions.shape
        block_size = int(self.config.frame_chunk_size)
        num_blocks = max((F + block_size - 1) // block_size, 1)
        # Delta-v from video velocity (latent branch), no grad back to DiT.
        delta_v = torch.zeros_like(video_velocity)
        delta_v[:, :, 1:] = video_velocity[:, :, 1:] - video_velocity[:, :, :-1]
        delta_v = delta_v.detach()

        action_loss = actions.new_tensor(0.0)
        raw_mse = actions.new_tensor(0.0)
        sigma_mean = actions.new_tensor(0.0)
        action_expert = self._get_action_expert()
        for block_id in range(num_blocks):
            start = block_id * block_size
            end = min((block_id + 1) * block_size, F)
            cur_f = end - start
            if cur_f <= 0:
                continue

            hist_action = actions[:, :, :start] * actions_mask[:, :, :start]
            hist_state = states[:, :, :start] * states_mask[:, :, :start]
            hist_delta = delta_v[:, :, :start]

            gt_action = actions[:, :, start:end] * actions_mask[:, :, start:end]
            gt_state = states[:, :, start:end] * states_mask[:, :, start:end]
            mask = actions_mask[:, :, start:end]

            t0 = torch.zeros((B, end), dtype=torch.float32, device=self.device)
            cur_zero = torch.zeros_like(gt_action)
            step1_actions = torch.cat([hist_action, cur_zero], dim=2)
            step1_states = torch.cat([hist_state, gt_state], dim=2)
            step1_delta = torch.cat([hist_delta, delta_v[:, :, start:end]], dim=2)
            pred_step1_all, sigma = action_expert(
                noisy_actions=step1_actions,
                timesteps=t0,
                delta_v=step1_delta,
                states=step1_states,
            )
            pred_step1 = pred_step1_all[:, :, start:end]

            z = torch.randn_like(gt_action, dtype=self.dtype)
            noisy_step2 = (self.mip_t_star * pred_step1.detach() + (1.0 - self.mip_t_star) * z) * mask
            t_star = torch.zeros((B, end), dtype=torch.float32, device=self.device)
            t_star[:, start:end] = self.mip_t_star
            step2_actions = torch.cat([hist_action, noisy_step2], dim=2)
            pred_step2_all, _ = action_expert(
                noisy_actions=step2_actions,
                timesteps=t_star,
                delta_v=step1_delta,
                states=step1_states,
            )
            pred_step2 = pred_step2_all[:, :, start:end]

            denom = mask.sum().float().clamp_min(1.0)
            loss_step1 = (((pred_step1.float() - gt_action.float()) ** 2) * mask).sum() / denom
            loss_step2 = (((pred_step2.float() - gt_action.float()) ** 2) * mask).sum() / denom
            action_loss = action_loss + (loss_step1 + loss_step2)
            raw_mse = raw_mse + loss_step2.detach()
            sigma_mean = sigma_mean + sigma[:, start:end].mean().detach()

        action_loss = action_loss / float(num_blocks)
        raw_mse = raw_mse / float(num_blocks)
        sigma_mean = sigma_mean / float(num_blocks)
        return action_loss, raw_mse, sigma_mean

    def _velocity_delta_metric(self, video_velocity):
        if video_velocity.shape[2] <= 1:
            return video_velocity.new_tensor(0.0)
        dv = video_velocity[:, :, 1:] - video_velocity[:, :, :-1]
        return dv.abs().mean().detach()

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        batch = self.convert_input_format(batch)
        video_dict = self._prepare_video_input(batch)
        
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        
        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        video_pred_seq = self.transformer(video_dict, action_mode=False)
        latent_loss, video_velocity = self._compute_video_loss(video_dict, video_pred_seq)
        action_loss, raw_mse, sigma_mean = self._compute_action_mip_loss(batch, video_velocity)
        vel_delta = self._velocity_delta_metric(video_velocity)

        if (not torch.isfinite(latent_loss)) or (not torch.isfinite(action_loss)):
            logger.warning(
                f"Non-finite loss detected at step={self.step}: "
                f"latent_loss={latent_loss.detach().float().cpu().item()}, "
                f"action_loss={action_loss.detach().float().cpu().item()}. "
                "Skip this micro-batch."
            )
            self.optimizer.zero_grad(set_to_none=True)
            zero = torch.zeros((), device=self.device, dtype=torch.float32)
            return {
                'latent_loss': zero,
                'action_loss': zero,
                'raw_mse': raw_mse.detach().float(),
                'vel_delta': vel_delta.detach().float(),
                'sigma_mean': sigma_mean.detach().float(),
                'should_log': False,
            }

        latent_loss = latent_loss / self.gradient_accumulation_steps
        action_loss = action_loss / self.gradient_accumulation_steps
        loss = latent_loss + action_loss

        loss.backward()

        losses = {
            'latent_loss': latent_loss.detach(),
            'action_loss': action_loss.detach(),
            'raw_mse': raw_mse.detach(),
            'vel_delta': vel_delta,
            'sigma_mean': sigma_mean,
        }
        
        # Only update weights after accumulating gradients
        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

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
        """Main training loop - train by steps instead of epochs."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_raw_mse_losses = []
        accumulated_vel_delta = []
        accumulated_sigma_mean = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            self._apply_freeze_policy()
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()
            
            losses = self._train_step(batch, step_in_accumulation)
            
            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            accumulated_raw_mse_losses.append(losses['raw_mse'])
            accumulated_vel_delta.append(losses['vel_delta'])
            accumulated_sigma_mean.append(losses['sigma_mean'])
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                # Average accumulated losses
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

                raw_mse_show = dist_mean(torch.stack(accumulated_raw_mse_losses).mean()).detach().cpu().item()
                vel_delta_show = dist_mean(torch.stack(accumulated_vel_delta).mean()).detach().cpu().item()
                sigma_mean_show = dist_mean(torch.stack(accumulated_sigma_mean).mean()).detach().cpu().item()
                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_raw_mse_losses = []
                accumulated_vel_delta = []
                accumulated_sigma_mean = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
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
                            'loss_metrics/global_avg_raw_mse': raw_mse_show,
                            'loss_metrics/global_avg_video_velocity_delta': vel_delta_show,
                            'loss_metrics/global_avg_state_gate_sigma': sigma_mean_show,
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
                    eval_batches = []
                    for _b in self.train_loader:
                        eval_batches.append(_b)
                        if len(eval_batches) >= eval_num_samples:
                            break
                    self.run_open_loop_eval(eval_batches)
            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("Training completed!")
    
    @torch.no_grad()
    def run_open_loop_eval(self, eval_batches):
        """MIP 开环评估：每个 block 两次前向，t=0 与 t=t*。"""
        self.transformer.eval()
        infer_chunk_size = int(self.config.frame_chunk_size)
        num_windows = getattr(self.config, 'eval_num_windows', 4)

        patch_f, patch_h, patch_w = self.patch_size

        total_mse = 0.0
        total_count = 0

        for batch in eval_batches:
            batch = self.convert_input_format(batch)
            latent_gt   = batch['latents']       # (B, C, LF, H, W)
            action_gt   = batch['actions']       # (B, CA, FA, NA, 1)
            state_gt    = batch['states']        # (B, CS, FS, NA, 1)
            text_emb    = batch['text_emb']
            action_mask = batch['actions_mask']  # (B, CA, FA, NA, 1) bool
            state_mask  = batch['states_mask']   # (B, CS, FS, NA, 1) bool

            action_expert = self._get_action_expert()
            B, C, LF, H, W = latent_gt.shape
            _, CA, FA, NA, _ = action_gt.shape

            if LF < infer_chunk_size:
                continue

            max_start = LF - infer_chunk_size
            starts = torch.randint(0, max_start + 1, (num_windows,)).tolist()

            latent_grid_id = get_mesh_id(
                infer_chunk_size // patch_f, H // patch_h, W // patch_w,
                t=0, f_w=1, f_shift=0, action=False
            ).to(self.device)[None].repeat(B, 1, 1)

            action_grid_id = get_mesh_id(
                infer_chunk_size, NA, 1, t=1, f_w=1, f_shift=0, action=True
            ).to(self.device)[None].repeat(B, 1, 1)

            for start in starts:
                end = start + infer_chunk_size

                lat_win  = latent_gt[:, :, start:end, :, :]
                act_win  = action_gt[:, :, start:end, :, :]
                mask_win = action_mask[:, :, start:end, :, :].to(self.dtype)

                _ = self.transformer({
                    "noisy_latents": lat_win.to(self.dtype),
                    "timesteps": torch.zeros(B, infer_chunk_size, device=self.device),
                    "grid_id": latent_grid_id,
                    "text_emb": text_emb,
                    "encoder_attention_mask": self._text_mask(text_emb),
                }, action_mode=False)
                video_velocity = data_seq_to_patch(
                    self.patch_size,
                    _,
                    infer_chunk_size,
                    H,
                    W,
                    batch_size=B,
                )
                delta_v = torch.zeros_like(video_velocity)
                delta_v[:, :, 1:] = video_velocity[:, :, 1:] - video_velocity[:, :, :-1]

                step1_input = {
                    "noisy_actions": torch.zeros_like(act_win, dtype=self.dtype),
                    "timesteps": torch.zeros(B, infer_chunk_size, dtype=torch.float32, device=self.device),
                    "delta_v": delta_v.detach(),
                    "states": (state_gt[:, :, start:end].to(self.dtype) * state_mask[:, :, start:end].to(self.dtype)),
                }
                pred0, _ = action_expert(**step1_input)

                step2_input = {
                    "noisy_actions": (self.mip_t_star * pred0 * mask_win).to(self.dtype),
                    "timesteps": torch.full(
                        (B, infer_chunk_size),
                        self.mip_t_star,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    "delta_v": delta_v.detach(),
                    "states": (state_gt[:, :, start:end].to(self.dtype) * state_mask[:, :, start:end].to(self.dtype)),
                }
                pred, _ = action_expert(**step2_input)

                mse = F.mse_loss(pred.float(), act_win.float(), reduction='none')
                mse_val = (mse * mask_win).sum() / (mask_win.sum() + 1e-6)
                total_mse += mse_val.item()
                total_count += 1

        avg_mse = total_mse / max(total_count, 1)

        # 训练 loss 在 self.step += 1 之前记录（即 step N-1），
        # eval 在 step += 1 之后触发，用 self.step - 1 对齐到同一个 step 轴
        log_step = self.step - 1
        if self.config.rank == 0:
            logger.info(f"[OpenLoop Eval step {log_step}] action MSE = {avg_mse:.6f} "
                        f"(windows={total_count})")
            if self.config.enable_wandb:
                self.wandb.log({'eval/open_loop_action_mse': avg_mse}, step=log_step)

        self.transformer.train()

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

    if args.resume_from is not None:
        config.resume_from = args.resume_from

    if getattr(args, "run_name", None) is not None:
        config.run_name = args.run_name

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
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from (e.g. .../checkpoint_step_1000)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="WandB/SwanLab run name (default: test_lln)",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()