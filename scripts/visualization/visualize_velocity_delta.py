#!/usr/bin/env python3
"""Visualize Δv on decoded z_t/z_{t-1} using training pipeline components.

核心目标：
1) 使用训练保存的 transformer（checkpoint_step_xxx/transformer）做一次前向，得到预测 v。
2) 按训练代码同样方式构造 target v=noise-latent。
3) 计算 frame 维度上的 Δv_t = v_t - v_{t-1}，并映射成热力图。
4) 将热力图叠加到可由 Wan VAE decoder 还原的 z_t / z_{t-1} 图像上。

说明：
- t 在本脚本里指视频 latent 的 frame 索引（不是 diffusion step 索引）。
- decoder 来自 base Wan ckpt（通常是 --base_ckpt_dir/vae）。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan

from wan_va.configs import VA_CONFIGS
from wan_va.dataset import MultiLatentLeRobotDataset
from wan_va.modules.utils import load_transformer, load_vae
from wan_va.utils.scheduler import FlowMatchScheduler
from wan_va.utils.utils import data_seq_to_patch, get_mesh_id


def _resolve_transformer_dir(train_ckpt: Path) -> Path:
    # 允许传 checkpoint_step_xxx 或其下 transformer 目录
    if (train_ckpt / "diffusion_pytorch_model.safetensors").exists():
        return train_ckpt
    cand = train_ckpt / "transformer"
    if (cand / "diffusion_pytorch_model.safetensors").exists():
        return cand
    raise FileNotFoundError(
        f"Cannot find transformer weights under {train_ckpt}. "
        "Expected diffusion_pytorch_model.safetensors or transformer/diffusion_pytorch_model.safetensors"
    )

def _load_vae_from_base(base_ckpt_dir: Path, device: torch.device, dtype: torch.dtype) -> AutoencoderKLWan:
    """Load Wan VAE robustly from either:
    - a root ckpt dir containing subfolder `vae/`, or
    - a direct VAE folder path.

    We force `low_cpu_mem_usage=False` to avoid meta tensors that can trigger
    `Cannot copy out of meta tensor` when moving to device.
    """
    base_ckpt_dir = Path(base_ckpt_dir)

    # 1) Most stable path: load from root + subfolder="vae"
    try:
        vae = AutoencoderKLWan.from_pretrained(
            str(base_ckpt_dir),
            subfolder="vae",
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        return vae.to(device)
    except Exception:
        pass

    # 2) Fallback: user passed direct VAE path
    vae_dir = base_ckpt_dir / "vae" if (base_ckpt_dir / "vae").exists() else base_ckpt_dir
    cfg_file = vae_dir / "config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"Cannot find VAE config at {cfg_file}. Please pass Wan base ckpt root (with vae/) "
            "or pass direct --base_ckpt_dir=/path/to/vae."
        )

    vae = AutoencoderKLWan.from_pretrained(
        str(vae_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    return vae.to(device)

def _decode_latents_to_frames(vae, latents_bcfhw: torch.Tensor) -> torch.Tensor:
    """latents_bcfhw: normalized latent [B,C,F,H,W] -> uint8 [B,F,H,W,3]"""
    latents = latents_bcfhw.to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype)

    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    latents = latents * std + mean

    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0]  # [B,3,F,H,W], [-1,1]
    video = ((video.clamp(-1, 1) + 1.0) * 0.5 * 255.0).round().to(torch.uint8)
    return video.permute(0, 2, 3, 4, 1).contiguous()  # [B,F,H,W,3]


def _to_color_heatmap(hm: torch.Tensor) -> torch.Tensor:
    """hm [H,W] in [0,1] -> uint8 [H,W,3]"""
    r = torch.clamp(1.5 * hm, 0.0, 1.0)
    g = torch.clamp(1.5 * (1 - torch.abs(hm - 0.5) * 2), 0.0, 1.0)
    b = torch.clamp(1.5 * (1 - hm), 0.0, 1.0)
    rgb = torch.stack([r, g, b], dim=-1)
    return (rgb * 255.0).round().to(torch.uint8)


def _overlay(base: torch.Tensor, heat: torch.Tensor, alpha: float) -> torch.Tensor:
    # base/heat uint8 [H,W,3]
    out = base.float() * (1.0 - alpha) + heat.float() * alpha
    return out.clamp(0, 255).to(torch.uint8)


def _save_panel(z_prev: torch.Tensor, z_cur: torch.Tensor, hm: torch.Tensor, alpha: float, out_file: Path):
    """保存三联图到 png。输入均为 CPU 张量。"""
    from PIL import Image, ImageDraw

    hm_rgb = _to_color_heatmap(hm)
    left = _overlay(z_prev, hm_rgb, alpha)
    mid = _overlay(z_cur, hm_rgb, alpha)

    h, w, _ = left.shape
    canvas = torch.zeros((h + 28, w * 3, 3), dtype=torch.uint8)
    canvas[28:, 0:w] = left
    canvas[28:, w:2 * w] = mid
    canvas[28:, 2 * w:3 * w] = hm_rgb

    img = Image.fromarray(canvas.numpy())
    draw = ImageDraw.Draw(img)
    draw.text((6, 6), "z_{t-1} + |Δv|", fill=(240, 240, 240))
    draw.text((w + 6, 6), "z_t + |Δv|", fill=(240, 240, 240))
    draw.text((2 * w + 6, 6), "|Δv|", fill=(240, 240, 240))
    img.save(out_file)


def parse_args():
    p = argparse.ArgumentParser(description="Visualize Δv on decoded z_t/z_{t-1} from training checkpoint")
    p.add_argument("--config_name", type=str, default="robotwin_video_train", choices=sorted(VA_CONFIGS.keys()))
    p.add_argument("--train_ckpt", type=str, required=True, help="checkpoint_step_xxx or .../transformer")
    p.add_argument("--base_ckpt_dir", type=str, required=True, help="Wan base ckpt dir (contains vae/) or VAE dir")

    p.add_argument("--sample_index", type=int, default=0, help="dataset sample index")
    p.add_argument("--noise_seed", type=int, default=0)
    p.add_argument("--timestep_id", type=int, default=500, help="diffusion timestep index [0,999]")

    p.add_argument("--dataset_path", type=str, default=None, help="optional override config.dataset_path")
    p.add_argument("--latent_subdir", type=str, default=None, help="optional override config.latent_subdir")

    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--max_frame_pairs", type=int, default=8)
    p.add_argument("--out_dir", type=str, default="outputs/velocity_delta_viz_train")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = deepcopy(VA_CONFIGS[args.config_name])
    cfg.rank = 0
    cfg.world_size = 1
    cfg.local_rank = 0
    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path
    if args.latent_subdir is not None:
        cfg.latent_subdir = args.latent_subdir

    dataset = MultiLatentLeRobotDataset(cfg)
    sample = dataset[args.sample_index]

    latents = sample["latents"].unsqueeze(0).to(device)  # [1,C,F,H,W]
    text_emb = sample["text_emb"].unsqueeze(0).to(device)
    _, _, Ff, Hh, Ww = latents.shape

    transformer_dir = _resolve_transformer_dir(Path(args.train_ckpt))
    model = load_transformer(
        str(transformer_dir),
        torch_dtype=torch.bfloat16,
        torch_device=device,
        attn_mode=getattr(cfg, "attn_mode", "torch"),
        model_name=getattr(cfg, "transformer_model_name", "wan_va"),
        transformer_source=getattr(cfg, "transformer_source", "lingbot_va"),
    )
    model.eval()

    scheduler = FlowMatchScheduler(shift=getattr(cfg, "snr_shift", 3.0), sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    t_id = int(max(0, min(999, args.timestep_id)))
    timestep = scheduler.timesteps[t_id].to(device)
    timesteps = timestep.repeat(Ff)

    g = torch.Generator(device=device)
    g.manual_seed(args.noise_seed)
    noise = torch.randn(latents.shape, generator=g, device=device, dtype=latents.dtype)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps, t_dim=2)
    target_v = scheduler.training_target(latents, noise, timesteps)  # [1,C,F,H,W]

    patch_f, patch_h, patch_w = model.patch_size
    grid_id = get_mesh_id(
        Ff // patch_f,
        Hh // patch_h,
        Ww // patch_w,
        t=0,
        f_w=1,
        f_shift=0,
        action=False,
    ).to(device)
    grid_id = grid_id[None].repeat(1, 1, 1)

    input_dict = {
        "noisy_latents": noisy_latents.to(torch.bfloat16),
        "latent": latents.to(torch.bfloat16),
        "timesteps": timesteps[None],
        "text_emb": text_emb.to(torch.bfloat16),
        "grid_id": grid_id,
    }

    with torch.no_grad():
        pred_seq = model(input_dict, train_mode=False)
    pred_v = data_seq_to_patch(model.patch_size, pred_seq, Ff, Hh, Ww, batch_size=1)

    # frame 维度上的 delta
    delta_target = target_v[:, :, 1:] - target_v[:, :, :-1]  # [1,C,F-1,H,W]
    delta_pred = pred_v[:, :, 1:] - pred_v[:, :, :-1]

    vae = _load_vae_from_base(Path(args.base_ckpt_dir), device=device, dtype=torch.bfloat16)
    vae.eval()
    decoded = _decode_latents_to_frames(vae, latents)[0].cpu()  # [F,Himg,Wimg,3]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_pairs = min(args.max_frame_pairs, delta_target.shape[2])
    for i in range(max_pairs):
        # [C,H,W] -> [H,W]
        hm_t = torch.sqrt(torch.clamp((delta_target[0, :, i].float() ** 2).sum(dim=0), min=1e-12))
        hm_p = torch.sqrt(torch.clamp((delta_pred[0, :, i].float() ** 2).sum(dim=0), min=1e-12))

        def _norm(x):
            x = x - x.min()
            return x / (x.max() + 1e-6)

        hm_t = _norm(hm_t)
        hm_p = _norm(hm_p)

        h_img, w_img = decoded.shape[1], decoded.shape[2]
        hm_t = F.interpolate(hm_t[None, None], size=(h_img, w_img), mode="bilinear", align_corners=False)[0, 0].cpu()
        hm_p = F.interpolate(hm_p[None, None], size=(h_img, w_img), mode="bilinear", align_corners=False)[0, 0].cpu()

        z_prev = decoded[i]
        z_cur = decoded[i + 1]

        _save_panel(z_prev, z_cur, hm_t, args.alpha, out_dir / f"target_delta_v_frame_{i+1:03d}.png")
        _save_panel(z_prev, z_cur, hm_p, args.alpha, out_dir / f"pred_delta_v_frame_{i+1:03d}.png")

    print(f"Done. Saved {max_pairs * 2} images to {out_dir}")


if __name__ == "__main__":
    main()
