import argparse
import contextlib
import gc
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import swanlab
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from dataset import MultiLatentLeRobotDataset
from distributed.fsdp import apply_ac, shard_model
from distributed.util import _configure_model, dist_max, dist_mean, init_distributed
from modules.action_expert_model import FlowMatchingActionExpert
from modules.utils import (
    load_transformer,
    remap_video_model_state_dict_to_wan_official,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    sample_timestep_id,
    warmup_constant_lambda,
)


class JointTrainer:
    def __init__(self, config):
        self.config = config
        self.step = 0
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = tuple(config.patch_size)
        self.mip_t_star = float(getattr(config, "mip_t_star", 0.9))
        self.mip_loss_weight_step0 = float(getattr(config, "mip_loss_weight_step0", 1.0))
        self.mip_loss_weight_step1 = float(getattr(config, "mip_loss_weight_step1", 1.0))
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.train_loader_iter = None

        if config.enable_wandb and config.rank == 0:
            swanlab.login(api_key=os.getenv("SWANLAB_API_KEY", None))
            self.wandb = swanlab
            self.wandb.init(
                project=os.getenv("SWANLAB_PROJECT", "lingbot-va"),
                config=dict(config),
                name=getattr(config, "run_name", None) or "joint_train",
            )
        else:
            self.wandb = None

        self._build_models()
        self._build_optimizers()
        self._build_data()

        self.train_scheduler_latent = FlowMatchScheduler(
            shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.train_scheduler_latent.set_timesteps(1000, training=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(config.save_root) / f"checkpoints_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _build_models(self):
        transformer_source = getattr(self.config, "transformer_source", "lingbot_va")
        if transformer_source == "wan_official":
            transformer_path = getattr(
                self.config, "wan_official_ckpt_path", self.config.wan22_pretrained_model_name_or_path
            )
        else:
            transformer_path = os.path.join(self.config.wan22_pretrained_model_name_or_path, "transformer")
        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device="cpu",
            attn_mode="flex",
            model_name=getattr(self.config, "transformer_model_name", "wan_video_finetune"),
            transformer_source=transformer_source,
        )
        apply_ac(self.transformer)
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_model,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        self.action_expert = FlowMatchingActionExpert(
            action_dim=self.config.action_dim,
            state_dim=self.config.robot_state_dim,
            velocity_dim=getattr(
                self.config, "action_expert_velocity_dim", self.config.action_dim
            ),
            hidden_dim=self.config.action_expert_hidden_dim,
            num_heads=self.config.action_expert_num_heads,
            num_layers=self.config.action_expert_num_layers,
            dropout=self.config.action_expert_dropout,
            ffn_mult=self.config.action_expert_ffn_mult,
            timestep_dim=self.config.action_expert_timestep_dim,
        ).to(self.device, dtype=self.dtype)
        if self.config.world_size > 1:
            self.action_expert = DDP(
                self.action_expert,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=False,
            )

    def _build_optimizers(self):
        self.optimizer_wan = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=1e-8,
            weight_decay=self.config.weight_decay,
            fused=True,
            foreach=False,
        )
        self.optimizer_expert = torch.optim.AdamW(
            [p for p in self.action_expert.parameters() if p.requires_grad],
            lr=self.config.action_expert_learning_rate,
            betas=(self.config.action_expert_beta1, self.config.action_expert_beta2),
            eps=1e-8,
            weight_decay=self.config.action_expert_weight_decay,
            fused=True,
            foreach=False,
        )
        self.lr_scheduler_wan = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_wan,
            lr_lambda=lambda s: warmup_constant_lambda(s, warmup_steps=self.config.warmup_steps),
        )
        self.lr_scheduler_expert = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_expert,
            lr_lambda=lambda s: warmup_constant_lambda(
                s, warmup_steps=self.config.action_expert_warmup_steps
            ),
        )

    def _build_data(self):
        dataset_init_worker = getattr(
            self.config,
            "dataset_init_worker",
            1 if self.config.world_size > 1 else 8,
        )
        if self.config.rank == 0:
            logger.info(f"Dataset init workers: {dataset_init_worker}")
        train_dataset = MultiLatentLeRobotDataset(
            config=self.config, num_init_worker=dataset_init_worker
        )
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=True,
                seed=42,
            )
            if self.config.world_size > 1
            else None
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.config.load_worker,
            sampler=train_sampler,
        )

    def _get_next_batch(self):
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        try:
            return next(self.train_loader_iter)
        except StopIteration:
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.step + 1)
            self.train_loader_iter = iter(self.train_loader)
            return next(self.train_loader_iter)

    def _to_device(self, batch):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(self.device)
        return batch

    @torch.no_grad()
    def _add_noise_latent(self, latent):
        b, _, f, h, w = latent.shape
        timestep_ids = sample_timestep_id(
            batch_size=f, num_train_timesteps=self.train_scheduler_latent.num_train_timesteps
        )
        noise = torch.randn_like(latent)
        timesteps = self.train_scheduler_latent.timesteps[timestep_ids].to(device=self.device)
        noisy_latents = self.train_scheduler_latent.add_noise(latent, noise, timesteps, t_dim=2)
        targets = self.train_scheduler_latent.training_target(latent, noise, timesteps)
        grid_id = get_mesh_id(
            f // self.patch_size[0],
            h // self.patch_size[1],
            w // self.patch_size[2],
            t=0,
            f_w=1,
            f_shift=0,
            action=False,
        ).to(self.device)[None].repeat(b, 1, 1)
        return {
            "timesteps": timesteps[None].repeat(b, 1),
            "noisy_latents": noisy_latents,
            "targets": targets,
            "latent": latent,
            "cond_timesteps": torch.zeros_like(timesteps)[None].repeat(b, 1),
            "grid_id": grid_id,
        }

    def _build_mip_inputs(self, actions, action_mask):
        t_star = self.mip_t_star
        mask = action_mask.float()
        noisy_step0 = torch.zeros_like(actions) * mask
        z = torch.randn_like(actions)
        noisy_step1 = (t_star * actions + (1.0 - t_star) * z) * mask
        t0 = torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t1 = torch.full(
            (actions.shape[0],), t_star, device=actions.device, dtype=actions.dtype
        )
        return noisy_step0, noisy_step1, t0, t1

    def _compute_video_loss(self, latent_dict, latent_pred_seq):
        latent_pred = data_seq_to_patch(
            self.patch_size,
            latent_pred_seq,
            latent_dict["targets"].shape[-3],
            latent_dict["targets"].shape[-2],
            latent_dict["targets"].shape[-1],
            batch_size=latent_pred_seq.shape[0],
        )
        bn, fn = latent_dict["timesteps"].shape
        latent_weight = self.train_scheduler_latent.training_weight(
            latent_dict["timesteps"].flatten()
        ).reshape(bn, fn)
        latent_loss = F.mse_loss(
            latent_pred.float(), latent_dict["targets"].float().detach(), reduction="none"
        )
        latent_loss = latent_loss * latent_weight[:, None, :, None, None]
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        per_frame = latent_loss.sum(dim=1)
        denom = torch.ones_like(latent_loss).sum(dim=1)
        return (per_frame / (denom + 1e-6)).mean(), latent_pred

    def _compute_action_loss(self, pred_action, gt_actions, action_mask):
        gt = gt_actions.permute(0, 2, 3, 1, 4).squeeze(-1)
        m = action_mask.permute(0, 2, 3, 1, 4).squeeze(-1).float()
        mse = (pred_action.float() - gt.float()) ** 2
        return (mse * m).sum() / (m.sum() + 1e-6)

    def _expert_no_sync_ctx(self, should_sync):
        if should_sync or not isinstance(self.action_expert, DDP):
            return contextlib.nullcontext()
        return self.action_expert.no_sync()

    def _train_step(self, batch, batch_idx):
        batch = self._to_device(batch)
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        self.transformer.set_requires_gradient_sync(should_sync)

        latent_dict = self._add_noise_latent(batch["latents"])
        latent_dict["text_emb"] = batch["text_emb"]
        latent_pred_seq = self.transformer(
            {"latent_dict": latent_dict, "window_size": int(getattr(self.config, "attn_window", 64))},
            train_mode=True,
        )
        video_loss, latent_pred_patch = self._compute_video_loss(latent_dict, latent_pred_seq)

        # Build detached velocity condition for action expert.
        latent_velocity_cond = latent_pred_patch.mean(dim=(-1, -2)).detach()
        # print(f"latent_velocity_cond shape: {latent_velocity_cond.shape}")
        clean_actions = batch["actions"].to(self.dtype)
        noisy_step0, noisy_step1, t0, t1 = self._build_mip_inputs(
            clean_actions, batch["actions_mask"]
        )
        robot_states = batch["robot_states"].to(self.dtype)
        action_token_valid = (
            batch["actions_mask"].any(dim=1).squeeze(-1).reshape(batch["actions_mask"].shape[0], -1)
        )
        action_padding_mask = ~action_token_valid

        with self._expert_no_sync_ctx(should_sync):
            pred_action_step0, gate0 = self.action_expert(
                noisy_actions=noisy_step0.to(self.dtype),
                timesteps=t0.to(self.dtype),
                latent_velocity=latent_velocity_cond.to(self.dtype),
                robot_states=robot_states,
                action_padding_mask=action_padding_mask,
                clean_actions=clean_actions,
            )
            pred_action_step1, gate1 = self.action_expert(
                noisy_actions=noisy_step1.to(self.dtype),
                timesteps=t1.to(self.dtype),
                latent_velocity=latent_velocity_cond.to(self.dtype),
                robot_states=robot_states,
                action_padding_mask=action_padding_mask,
                clean_actions=clean_actions,
            )
            action_loss_step0 = self._compute_action_loss(
                pred_action_step0, clean_actions, batch["actions_mask"]
            )
            action_loss_step1 = self._compute_action_loss(
                pred_action_step1, clean_actions, batch["actions_mask"]
            )
            action_loss = (
                self.mip_loss_weight_step0 * action_loss_step0
                + self.mip_loss_weight_step1 * action_loss_step1
            )
            total_loss = (video_loss + action_loss) / self.gradient_accumulation_steps
            total_loss.backward()

        out = {
            "video_loss": (video_loss / self.gradient_accumulation_steps).detach(),
            "action_loss": (action_loss / self.gradient_accumulation_steps).detach(),
            "action_loss_step0": (action_loss_step0 / self.gradient_accumulation_steps).detach(),
            "action_loss_step1": (action_loss_step1 / self.gradient_accumulation_steps).detach(),
            "mean_gate": ((gate0 + gate1) * 0.5).detach(),
            "should_log": should_sync,
        }
        if should_sync:
            wan_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            exp_norm = torch.nn.utils.clip_grad_norm_(self.action_expert.parameters(), 1.0)
            self.optimizer_wan.step()
            self.optimizer_expert.step()
            self.lr_scheduler_wan.step()
            self.lr_scheduler_expert.step()
            self.optimizer_wan.zero_grad()
            self.optimizer_expert.zero_grad(set_to_none=True)
            out["wan_norm"] = wan_norm.detach()
            out["exp_norm"] = exp_norm.detach()
        return out

    def save_checkpoint(self):
        try:
            wan_state = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            wan_state_bf16 = {k: v.to(torch.bfloat16) for k, v in wan_state.items()}
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)
                save_wan_official = (
                    getattr(self.config, "transformer_source", None) == "wan_official"
                    and getattr(self.config, "transformer_model_name", None) == "wan_video_finetune"
                )
                if save_wan_official:
                    in_channels = getattr(self.transformer.config, "in_channels", 48)
                    patch_size = getattr(self.transformer.config, "patch_size", [1, 2, 2])
                    patch_size = tuple(patch_size) if isinstance(patch_size, (list, tuple)) else (1, 2, 2)
                    mapped = remap_video_model_state_dict_to_wan_official(
                        wan_state_bf16, in_channels=in_channels, patch_size=patch_size
                    )
                    save_file(mapped, transformer_dir / "diffusion_pytorch_model.safetensors")
                    wan_ckpt = Path(getattr(self.config, "wan_official_ckpt_path", ""))
                    if wan_ckpt and (wan_ckpt / "config.json").exists():
                        shutil.copy2(wan_ckpt / "config.json", transformer_dir / "config.json")
                    else:
                        cfg = dict(self.transformer.config)
                        cfg.pop("_name_or_path", None)
                        with open(transformer_dir / "config.json", "w") as f:
                            json.dump(cfg, f, indent=2)
                else:
                    save_file(wan_state_bf16, transformer_dir / "diffusion_pytorch_model.safetensors")
                    cfg = dict(self.transformer.config)
                    cfg.pop("_name_or_path", None)
                    with open(transformer_dir / "config.json", "w") as f:
                        json.dump(cfg, f, indent=2)

                expert_module = self.action_expert.module if isinstance(self.action_expert, DDP) else self.action_expert
                torch.save(expert_module.state_dict(), checkpoint_dir / "action_expert.pt")
                with open(checkpoint_dir / "joint_train_config.json", "w", encoding="utf-8") as f:
                    json.dump(dict(self.config), f, indent=2, default=str)

            if dist.is_initialized():
                dist.barrier()
        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback

                logger.error(traceback.format_exc())
            if dist.is_initialized():
                dist.barrier()

    def train(self):
        logger.info(f"Starting joint training for {self.config.num_steps} steps...")
        self.transformer.train()
        self.action_expert.train()
        self.optimizer_wan.zero_grad()
        self.optimizer_expert.zero_grad(set_to_none=True)

        progress = tqdm(
            total=self.config.num_steps,
            desc="Joint Training",
            disable=(self.config.rank != 0),
            dynamic_ncols=True,
        )
        acc_video, acc_action, acc_action_step0, acc_action_step1, acc_gate = [], [], [], [], []
        step_in_acc = 0

        while self.step < self.config.num_steps:
            batch = self._get_next_batch()
            # print(batch["latents"].shape, batch["actions"].shape, batch["robot_states"].shape)
            out = self._train_step(batch, step_in_acc)
            acc_video.append(out["video_loss"])
            acc_action.append(out["action_loss"])
            acc_action_step0.append(out["action_loss_step0"])
            acc_action_step1.append(out["action_loss_step1"])
            acc_gate.append(out["mean_gate"])
            step_in_acc += 1

            if out["should_log"]:
                video_show = dist_mean(torch.stack(acc_video).sum()).detach().cpu().item()
                action_show = dist_mean(torch.stack(acc_action).sum()).detach().cpu().item()
                action_step0_show = dist_mean(torch.stack(acc_action_step0).sum()).detach().cpu().item()
                action_step1_show = dist_mean(torch.stack(acc_action_step1).sum()).detach().cpu().item()
                gate_show = dist_mean(torch.stack(acc_gate).mean()).detach().cpu().item()
                max_video = dist_max(torch.stack(acc_video).sum()).detach().cpu().item()
                max_action = dist_max(torch.stack(acc_action).sum()).detach().cpu().item()
                acc_video, acc_action, acc_action_step0, acc_action_step1, acc_gate = [], [], [], [], []
                step_in_acc = 0
                self.step += 1

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    lr_wan = self.lr_scheduler_wan.get_last_lr()[0]
                    lr_exp = self.lr_scheduler_expert.get_last_lr()[0]
                    progress.update(1)
                    progress.set_postfix(
                        {
                            "video": f"{video_show:.4f}",
                            "action": f"{action_show:.4f}",
                            "action_s0": f"{action_step0_show:.4f}",
                            "action_s1": f"{action_step1_show:.4f}",
                            "gate": f"{gate_show:.3f}",
                            "lr_wan": f"{lr_wan:.2e}",
                            "lr_exp": f"{lr_exp:.2e}",
                        }
                    )
                    if self.wandb is not None:
                        self.wandb.log(
                            {
                                "loss/video_avg": video_show,
                                "loss/action_avg": action_show,
                                "loss/action_step0_avg": action_step0_show,
                                "loss/action_step1_avg": action_step1_show,
                                "loss/video_max": max_video,
                                "loss/action_max": max_action,
                                "gate/mean": gate_show,
                                "grad/wan_norm": out["wan_norm"].item(),
                                "grad/expert_norm": out["exp_norm"].item(),
                                "lr/wan": lr_wan,
                                "lr/expert": lr_exp,
                            },
                            step=self.step,
                        )

                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

        progress.close()
        logger.info("Joint training completed.")


def run(args):
    config = VA_CONFIGS[args.config_name]
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    if args.save_root is not None:
        config.save_root = args.save_root
    if args.run_name is not None:
        config.run_name = args.run_name

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
    trainer = JointTrainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Joint train WAN(video) + action expert")
    parser.add_argument("--config-name", type=str, default="robotwin_joint_train")
    parser.add_argument("--save-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
