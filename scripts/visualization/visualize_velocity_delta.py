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
import json
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan

from wan_va.configs import VA_CONFIGS
from wan_va.dataset import MultiLatentLeRobotDataset
from wan_va.modules.utils import load_transformer
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


def _decode_one_group(vae, latents_bcfhw: torch.Tensor) -> torch.Tensor:
    """Decode one latent group with channel count == VAE z_dim.

    Input: normalized latent [B,C,F,H,W]
    Output: uint8 [B,F,H,W,3]
    """
    latents = latents_bcfhw.to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype)
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    latents = latents * std + mean

    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0]  # [B,3,F,H,W], [-1,1]
    video = ((video.clamp(-1, 1) + 1.0) * 0.5 * 255.0).round().to(torch.uint8)
    return video.permute(0, 2, 3, 4, 1).contiguous()


def _decode_latents_to_frames(vae, latents_bcfhw: torch.Tensor) -> torch.Tensor:
    """Decode normalized latents to RGB frames.

    - If C == z_dim: decode directly.
    - If C is multiple of z_dim (e.g. 48=3x16): split channels by group and decode each,
      then concatenate decoded views along image width.
    """
    z_dim = int(getattr(vae.config, "z_dim", len(vae.config.latents_mean)))
    c = int(latents_bcfhw.shape[1])

    if c == z_dim:
        return _decode_one_group(vae, latents_bcfhw)

    if c % z_dim != 0:
        raise ValueError(
            f"Latent channels C={c} is not compatible with VAE z_dim={z_dim}. "
            "Expected C==z_dim or C being a multiple of z_dim."
        )

    num_groups = c // z_dim
    decoded_groups = []
    for g in range(num_groups):
        sub = latents_bcfhw[:, g * z_dim : (g + 1) * z_dim]
        decoded_groups.append(_decode_one_group(vae, sub))
    # [B,F,H,W,3] x G -> [B,F,H,W*G,3]
    return torch.cat(decoded_groups, dim=3)


def _merge_group_heatmaps(hm_chw: torch.Tensor, z_dim: int) -> torch.Tensor:
    """hm_chw: [C,H,W] -> [H,W*G] if C=G*z_dim, else [H,W]."""
    c = hm_chw.shape[0]
    if c == z_dim:
        return torch.sqrt(torch.clamp((hm_chw.float() ** 2).sum(dim=0), min=1e-12))
    if c % z_dim != 0:
        raise ValueError(f"Cannot split heatmap channels C={c} by z_dim={z_dim}")
    g = c // z_dim
    parts = []
    for i in range(g):
        part = hm_chw[i * z_dim : (i + 1) * z_dim]
        parts.append(torch.sqrt(torch.clamp((part.float() ** 2).sum(dim=0), min=1e-12)))
    return torch.cat(parts, dim=1)


def _load_video_frames_range(video_path: Path, start_frame: int, end_frame: int) -> list[torch.Tensor]:
    """Load RGB frames [start_frame, end_frame) from one mp4 via pyav.

    Returns a list of uint8 tensors [3,H,W].
    """
    import av

    out = []
    with av.open(str(video_path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i >= end_frame:
                break
            if i < start_frame:
                continue
            arr = frame.to_ndarray(format="rgb24")
            t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(torch.uint8)
            out.append(t)
    return out


def _resize_chw_uint8(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    x = img.float()[None]
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x[0].round().clamp(0, 255).to(torch.uint8)


def _compose_frame_robotwin_tshape(high: torch.Tensor, left: torch.Tensor, right: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # high -> (h,w), wrist -> (h//2,w//2), then [wrist_row; high]
    high_r = _resize_chw_uint8(high, h, w)
    left_r = _resize_chw_uint8(left, h // 2, w // 2)
    right_r = _resize_chw_uint8(right, h // 2, w // 2)
    wrist_row = torch.cat([left_r, right_r], dim=2)
    comp = torch.cat([wrist_row, high_r], dim=1)
    return comp.permute(1, 2, 0).contiguous()  # [H,W,3]


def _build_original_frame_sequence(cur_dset, cur_meta: dict, num_latent_frames: int, cfg) -> list[torch.Tensor]:
    """Build composed original frames aligned with latent layout.

    Returns list length == num_latent_frames * 4 of uint8 [H,W,3].
    """
    root = Path(cur_dset.repo_id)
    info = json.loads((root / "meta" / "info.json").read_text())
    ep = int(cur_meta["episode_index"])
    start = int(cur_meta["start_frame"])
    end = int(cur_meta["end_frame"])
    chunk_size = int(info.get("chunks_size", 100000))
    chunk = ep // chunk_size
    tpl = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")

    max_needed = num_latent_frames * 4
    end_eff = min(end, start + max_needed)

    cams = cfg.obs_cam_keys
    cam_frames = {}
    for cam in cams:
        rel = tpl.format(episode_chunk=chunk, video_key=cam, episode_index=ep)
        video_file = root / rel
        cam_frames[cam] = _load_video_frames_range(video_file, start, end_eff)

    T = min(len(v) for v in cam_frames.values())
    if T == 0:
        return []

    out = []
    for i in range(T):
        if cfg.env_type == "robotwin_tshape" and len(cams) >= 3:
            frame = _compose_frame_robotwin_tshape(
                cam_frames[cams[0]][i],
                cam_frames[cams[1]][i],
                cam_frames[cams[2]][i],
                h=cfg.height,
                w=cfg.width,
            )
        else:
            # generic: resize all to (h,w), concat in width
            parts = [_resize_chw_uint8(cam_frames[c][i], cfg.height, cfg.width) for c in cams]
            frame = torch.cat(parts, dim=2).permute(1, 2, 0).contiguous()
        out.append(frame)
    return out


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
    p.add_argument(
        "--base_ckpt_dir",
        type=str,
        required=True,
        help="Wan base ckpt root (preferred, contains vae/) or direct VAE dir",
    )

    p.add_argument("--sample_index", type=int, nargs="+", default=[0])
    p.add_argument("--noise_seed", type=int, default=0)
    p.add_argument("--timestep_id", type=int, default=500, help="diffusion timestep index [0,999]")

    p.add_argument("--dataset_path", type=str, default=None, help="optional override config.dataset_path")
    p.add_argument("--latent_subdir", type=str, default=None, help="optional override config.latent_subdir")

    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--max_frame_pairs", type=int, default=10)
    p.add_argument("--delta_source", type=str, default="pred", choices=["pred", "target"], help="Use pred_v or target_v for Δv")
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

    vae = _load_vae_from_base(Path(args.base_ckpt_dir), device=device, dtype=torch.bfloat16)
    vae.eval()
    for si in args.sample_index:
        # sample = dataset[si]
        sample = dataset[si]
        dset_id = dataset.item_id_to_dataset_id[si]
        local_idx = si - dataset.acc_dset_num[dset_id]
        cur_dset = dataset._datasets[dset_id]
        cur_meta = cur_dset.new_metas[local_idx]

        latents = sample["latents"].unsqueeze(0).to(device)  # [1,C,F,H,W]
        text_emb = sample["text_emb"].unsqueeze(0).to(device)
        _, _, Ff, Hh, Ww = latents.shape

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

        # frame 维度上的 delta（按用户定义：delta[0]=v0-0，delta[t]=v_t-v_{t-1}）
        delta_target = torch.zeros_like(target_v)
        delta_pred = torch.zeros_like(pred_v)
        delta_target[:, :, 0] = target_v[:, :, 0]
        delta_pred[:, :, 0] = pred_v[:, :, 0]
        delta_target[:, :, 1:] = target_v[:, :, 1:] - target_v[:, :, :-1]
        delta_pred[:, :, 1:] = pred_v[:, :, 1:] - pred_v[:, :, :-1]

        z_dim = int(getattr(vae.config, "z_dim", len(vae.config.latents_mean)))

        if latents.shape[1] != z_dim and latents.shape[1] % z_dim != 0:
            raise ValueError(
                f"Latent channels {latents.shape[1]} incompatible with VAE z_dim={z_dim}. "
                "Please check dataset latent format or base VAE path."
            )

        # original composed frames (o1...): used as final visualization background
        original_frames = _build_original_frame_sequence(cur_dset, cur_meta, num_latent_frames=Ff, cfg=cfg)
        if len(original_frames) < Ff * 4:
            print(f"[warn] original frames only {len(original_frames)} < required {Ff*4}, will visualize available frames only")

        # out_dir = Path(args.out_dir)
        # out_dir.mkdir(parents=True, exist_ok=True)

        base_dir = Path(args.out_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        out_dir = base_dir / f"{timestamp}_sample_{si}"
        out_dir.mkdir(parents=True, exist_ok=True)

        delta_used = delta_pred if args.delta_source == "pred" else delta_target
        max_pairs = min(args.max_frame_pairs, delta_used.shape[2])
        
        saved_delta_sources = set() 
        for i in range(max_pairs):
            # [C,H,W] -> [H,W]
            hm = _merge_group_heatmaps(delta_used[0, :, i], z_dim=z_dim)

            def _norm(x):
                x = x - x.min()
                return x / (x.max() + 1e-6)

            hm = _norm(hm)

            # map delta_i to original frames [4i, 4i+1, 4i+2, 4i+3]
            st = i * 4
            ed = st + 4
            if st >= len(original_frames):
                break
            # group_frames = original_frames[st:min(ed, len(original_frames))]
            # for j, bg in enumerate(group_frames):
            #     h_img, w_img = bg.shape[0], bg.shape[1]
            #     hm_up = F.interpolate(hm[None, None], size=(h_img, w_img), mode="bilinear", align_corners=False)[0, 0].cpu()
            #     hm_rgb = _to_color_heatmap(hm_up)
            #     ov = _overlay(bg, hm_rgb, args.alpha)

            #     frame_global = st + j
            #     from PIL import Image
            #     Image.fromarray(ov.numpy()).save(out_dir / f"delta_{args.delta_source}_latent_{i+1:03d}_frame_{frame_global+1:04d}.png")
            #     Image.fromarray(hm_rgb.numpy()).save(out_dir / f"weight_{args.delta_source}_latent_{i+1:03d}_frame_{frame_global+1:04d}.png")
            group_frames = original_frames[st:min(ed, len(original_frames))]
            for j, bg in enumerate(group_frames):
                h_img, w_img = bg.shape[0], bg.shape[1]
                hm_up = F.interpolate(hm[None, None], size=(h_img, w_img), mode="bilinear", align_corners=False)[0, 0].cpu()
                hm_rgb = _to_color_heatmap(hm_up)
                ov = _overlay(bg, hm_rgb, args.alpha)

                frame_global = st + j

                # 只保存每个 delta_source 的第一张
                if i not in saved_delta_sources:
                    from PIL import Image
                    Image.fromarray(ov.numpy()).save(out_dir / f"delta_{args.delta_source}_latent_{i+1:03d}_frame_{frame_global+1:04d}.png")
                    # Image.fromarray(hm_rgb.numpy()).save(out_dir / f"weight_{args.delta_source}_latent_{i+1:03d}_frame_{frame_global+1:04d}.png")

                    saved_delta_sources.add(i)

        print(f"Done. Saved overlays/weights to {out_dir}")


if __name__ == "__main__":
    main()
