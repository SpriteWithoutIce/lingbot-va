import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import swanlab
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from dataset import MultiLatentLeRobotDataset
from distributed.util import dist_mean, init_distributed
from modules.action_expert_model import FlowMatchingActionExpert
from modules.utils import load_transformer
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    warmup_constant_lambda,
)


class ActionExpertTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.step = 0
        self.train_loader_iter = None
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)

        if config.enable_wandb and config.rank == 0:
            swanlab.login(api_key=os.getenv("SWANLAB_API_KEY", None))
            self.wandb = swanlab
            self.wandb.init(
                project=os.getenv("SWANLAB_PROJECT", "lingbot-va"),
                config=dict(config),
                name=getattr(config, "run_name", None) or "action_expert_train",
            )
        else:
            self.wandb = None

        self.patch_size = tuple(config.patch_size)
        self.mip_t_star = float(getattr(config, "mip_t_star", 0.9))
        self.mip_loss_weight_step0 = float(getattr(config, "mip_loss_weight_step0", 1.0))
        self.mip_loss_weight_step1 = float(getattr(config, "mip_loss_weight_step1", 1.0))
        self.latent_scheduler = FlowMatchScheduler(
            shift=config.snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.latent_scheduler.set_timesteps(1000)

        self._build_models()
        self._build_data()
        self._build_optim()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(config.save_root) / f"checkpoints_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _build_models(self):
        wan_model_path = self.config.wan_stage1_model_path
        if self.config.rank == 0:
            logger.info(f"Loading frozen WAN backbone from {wan_model_path}")
        self.wan_backbone = load_transformer(
            wan_model_path,
            torch_dtype=torch.float32,
            torch_device="cpu",
            attn_mode="flex",
            model_name=self.config.wan_stage1_model_name,
            transformer_source=self.config.wan_stage1_source,
        ).to(self.device)
        self.wan_backbone.eval()
        self.wan_backbone.requires_grad_(False)

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

    def _build_optim(self):
        params = [p for p in self.action_expert.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=1e-8,
            weight_decay=self.config.weight_decay,
            fused=True,
            foreach=False,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda s: warmup_constant_lambda(
                s, warmup_steps=self.config.warmup_steps
            ),
        )

    def _get_next_batch(self):
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.step + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        return batch

    def _to_device(self, batch):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(self.device)
        return batch

    @torch.no_grad()
    def _build_wan_velocity_cond(self, batch):
        latents = batch["latents"]
        b, _, f, h, w = latents.shape
        timestep_ids = torch.randint(
            0, self.latent_scheduler.num_train_timesteps, (b,), device=self.device
        )
        timesteps = self.latent_scheduler.timesteps[timestep_ids.cpu()].to(self.device)
        sigma = self.latent_scheduler.sigmas[timestep_ids.cpu()].to(self.device)
        sigma = sigma.view(b, 1, 1, 1, 1).to(latents.dtype)
        noise = torch.randn_like(latents)
        noisy_latents = (1.0 - sigma) * latents + sigma * noise

        grid_id = get_mesh_id(
            f // self.patch_size[0],
            h // self.patch_size[1],
            w // self.patch_size[2],
            t=0,
            f_w=1,
            f_shift=0,
            action=False,
        ).to(self.device)
        grid_id = grid_id[None].repeat(b, 1, 1)

        latent_dict = {
            "noisy_latents": noisy_latents.to(self.dtype),
            "text_emb": batch["text_emb"],
            "grid_id": grid_id,
            "timesteps": timesteps[:, None].repeat(1, f),
        }
        pred_seq = self.wan_backbone(latent_dict, train_mode=False)
        pred_patch = data_seq_to_patch(
            self.patch_size,
            pred_seq,
            latent_num_frames=f,
            latent_height=h,
            latent_width=w,
            batch_size=b,
        )
        # Convert per-token velocity to per-frame condition.
        latent_velocity = pred_patch.mean(dim=(-1, -2))  # [B, C, F]
        return latent_velocity

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

    def _compute_loss(self, pred_action, gt_actions, action_mask):
        # pred_action: [B, F, N, C], gt/mask: [B, C, F, N, 1]
        gt = gt_actions.permute(0, 2, 3, 1, 4).squeeze(-1)
        m = action_mask.permute(0, 2, 3, 1, 4).squeeze(-1).float()
        mse = (pred_action.float() - gt.float()) ** 2
        loss = (mse * m).sum() / (m.sum() + 1e-6)
        return loss

    def _train_step(self, batch, batch_idx):
        batch = self._to_device(batch)
        with torch.no_grad():
            latent_velocity = self._build_wan_velocity_cond(batch)

        clean_actions = batch["actions"].to(self.dtype)
        noisy_step0, noisy_step1, t0, t1 = self._build_mip_inputs(
            clean_actions, batch["actions_mask"]
        )

        robot_states = batch["robot_states"].to(self.dtype)
        action_token_valid = (
            batch["actions_mask"].any(dim=1).squeeze(-1).reshape(batch["actions_mask"].shape[0], -1)
        )
        action_padding_mask = ~action_token_valid

        pred_action_step0, gate0 = self.action_expert(
            noisy_actions=noisy_step0,
            timesteps=t0,
            latent_velocity=latent_velocity.to(self.dtype),
            robot_states=robot_states,
            action_padding_mask=action_padding_mask,
            clean_actions=clean_actions,
        )
        pred_action_step1, gate1 = self.action_expert(
            noisy_actions=noisy_step1,
            timesteps=t1,
            latent_velocity=latent_velocity.to(self.dtype),
            robot_states=robot_states,
            action_padding_mask=action_padding_mask,
            clean_actions=clean_actions,
        )
        loss_step0 = self._compute_loss(pred_action_step0, clean_actions, batch["actions_mask"])
        loss_step1 = self._compute_loss(pred_action_step1, clean_actions, batch["actions_mask"])
        loss = self.mip_loss_weight_step0 * loss_step0 + self.mip_loss_weight_step1 * loss_step1
        loss = loss / self.gradient_accumulation_steps
        loss.backward()

        should_step = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        total_norm = None
        if should_step:
            total_norm = torch.nn.utils.clip_grad_norm_(self.action_expert.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": loss.detach(),
            "loss_step0": (loss_step0 / self.gradient_accumulation_steps).detach(),
            "loss_step1": (loss_step1 / self.gradient_accumulation_steps).detach(),
            "gate": ((gate0 + gate1) * 0.5).detach(),
            "grad_norm": total_norm.detach() if total_norm is not None else None,
            "should_step": should_step,
        }

    def _module_state_dict(self):
        module = self.action_expert.module if isinstance(self.action_expert, DDP) else self.action_expert
        return module.state_dict()

    def save_checkpoint(self):
        if self.config.rank != 0:
            return
        ckpt_dir = self.save_dir / f"checkpoint_step_{self.step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model_path = ckpt_dir / "action_expert.pt"
        torch.save(self._module_state_dict(), model_path)
        with open(ckpt_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(dict(self.config), f, indent=2, default=str)
        logger.info(f"Saved checkpoint: {ckpt_dir}")

    def train(self):
        self.action_expert.train()
        self.optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="ActionExpert Training",
            disable=(self.config.rank != 0),
            dynamic_ncols=True,
        )

        accum_loss = []
        accum_loss_step0 = []
        accum_loss_step1 = []
        accum_gate = []
        step_in_accum = 0
        while self.step < self.config.num_steps:
            batch = self._get_next_batch()
            out = self._train_step(batch, step_in_accum)
            accum_loss.append(out["loss"])
            accum_loss_step0.append(out["loss_step0"])
            accum_loss_step1.append(out["loss_step1"])
            accum_gate.append(out["gate"])
            step_in_accum += 1

            if out["should_step"]:
                mean_loss = dist_mean(torch.stack(accum_loss).sum()).item()
                mean_loss_step0 = dist_mean(torch.stack(accum_loss_step0).sum()).item()
                mean_loss_step1 = dist_mean(torch.stack(accum_loss_step1).sum()).item()
                mean_gate = dist_mean(torch.stack(accum_gate).mean()).item()
                lr = self.lr_scheduler.get_last_lr()[0]
                accum_loss, accum_loss_step0, accum_loss_step1, accum_gate = [], [], [], []
                step_in_accum = 0
                self.step += 1

                if self.config.rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {
                            "loss": f"{mean_loss:.4f}",
                            "loss_s0": f"{mean_loss_step0:.4f}",
                            "loss_s1": f"{mean_loss_step1:.4f}",
                            "gate": f"{mean_gate:.3f}",
                            "lr": f"{lr:.2e}",
                        }
                    )
                    if self.wandb is not None:
                        self.wandb.log(
                            {
                                "train/loss": mean_loss,
                                "train/loss_step0": mean_loss_step0,
                                "train/loss_step1": mean_loss_step1,
                                "train/mean_gate": mean_gate,
                                "train/lr": lr,
                                "train/grad_norm": float(out["grad_norm"]) if out["grad_norm"] is not None else 0.0,
                            },
                            step=self.step,
                        )

                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        if self.config.rank == 0:
            logger.info("Action expert training completed.")


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
    if args.wan_stage1_model_path is not None:
        config.wan_stage1_model_path = args.wan_stage1_model_path

    trainer = ActionExpertTrainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Train flow-matching action expert")
    parser.add_argument("--config-name", type=str, default="robotwin_action_expert_train")
    parser.add_argument("--save-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wan-stage1-model-path", type=str, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
