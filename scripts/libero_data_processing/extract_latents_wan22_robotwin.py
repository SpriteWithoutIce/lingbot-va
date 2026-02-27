#!/usr/bin/env python3
"""
使用 Wan2.2 VAE + T5 为 LeRobot 格式数据集抽取 latent，生成 LingBot-VA README 要求的 .pth 文件。

README 要求每个 .pth 包含:
  latent, latent_num_frames, latent_height, latent_width,
  video_num_frames, video_height, video_width,
  text_emb, text, frame_ids, start_frame, end_frame, fps, ori_fps

依赖: 安装 Wan2.2 仓库依赖，以及 decord 或 torchvision 用于读视频。
  pip install decord  # 或使用 torchvision 读整段视频

用法:
  # 单数据集
  python scripts/libero/extract_latents_wan22.py \
    --wan22_root /home/jwhe/linyihan/Wan2.2 \
    --ckpt_dir /home/jwhe/linyihan/Wan2.2-TI2V-5B \
    --dataset_path /home/jwhe/linyihan/datasets/libero_lingbot/libero_spatial_dataset

  # 可选: 只处理前 N 个 episode（调试用）
  python ... --max_episodes 2
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from transformers import T5TokenizerFast, UMT5EncoderModel

def get_text_emb(prompt, tokenizer, text_encoder, device, dtype,
                 max_sequence_length=512):
    """与 wan_va_server._get_t5_prompt_embeds 逻辑相同：
    tokenize → encode → 截到实际 token 数 → pad 回 max_sequence_length。
    返回 (max_sequence_length, dim) 的 bfloat16 张量。"""
    if isinstance(prompt, str):
        prompt = [prompt]
    prompt = [prompt_clean(u) for u in prompt]

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()

    with torch.no_grad():
        embeds = text_encoder(input_ids, attention_mask=mask).last_hidden_state
    embeds = embeds.to(dtype=dtype, device=device)

    # 截掉 padding，再 pad 回 max_sequence_length（与 server 一致）
    embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
    embeds = torch.stack([
        torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
        for u in embeds
    ], dim=0)
    return embeds[0]  # (max_sequence_length, dim)


# 添加 Wan2.2 到 path
def _add_wan22(wan22_root):
    r = Path(wan22_root).resolve()
    if not r.is_dir():
        raise FileNotFoundError(f"Wan2.2 root not found: {r}")
    wan_dir = r / "wan"
    if wan_dir not in sys.path:
        sys.path.insert(0, str(r))


def load_video_frames(video_path, start_frame, end_frame, ori_fps, device="cuda"):
    """加载 [start_frame, end_frame) 的视频帧，返回 (C, T, H, W)，范围 [0,1]，再在脚本里转为 [-1,1]。
    LIBERO 等数据集常用 AV1 编码，decord 对 AV1 支持差会报 cannot find video stream；
    优先用 PyAV（支持 AV1），失败再试 decord / torchvision。"""
    path = str(video_path)
    # 1) PyAV：对 AV1 等编码支持好（LeRobot 常用）
    try:
        import av
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            frames = []
            for i, frame in enumerate(container.decode(video=0)):
                if i >= end_frame:
                    break
                if i < start_frame:
                    continue
                img = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
                frames.append(img)
            if not frames:
                return None
            import numpy as np
            arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, C)
            t = torch.from_numpy(arr).permute(3, 0, 1, 2).to(device)  # (C, T, H, W)
            return t
    except Exception:
        pass
    # 2) decord（部分编码可能报 cannot find video stream）
    try:
        import decord
        decord.bridge.set_bridge("torch")
        reader = decord.VideoReader(path)
        indices = list(range(start_frame, min(end_frame, len(reader))))
        if not indices:
            return None
        frames = reader.get_batch(indices)  # (T, H, W, C)
        frames = frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W), [0,1]
        return frames.permute(1, 0, 2, 3).to(device)  # (C, T, H, W)
    except Exception:
        pass
    # 3) torchvision 读整段
    try:
        import torchvision.io as tv_io
        v, _, _ = tv_io.read_video(path, start_pts=0, end_pts=None, pts_unit="sec")
        if v is None or v.shape[0] == 0:
            return None
        v = v[start_frame:end_frame].float() / 255.0  # (T, H, W, C)
        return v.permute(3, 0, 1, 2).to(device)  # (C, T, H, W)
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Extract Wan2.2 latents for LingBot-VA dataset")
    parser.add_argument("--wan22_root", type=str, default="/home/jwhe/linyihan/Wan2.2",
                        help="Path to Wan2.2 repo root")
    parser.add_argument("--ckpt_dir", type=str, default="/home/jwhe/linyihan/Wan2.2-TI2V-5B",
                        help="Fallback checkpoint dir (used if vae_dir not set)")
    parser.add_argument("--vae_dir", type=str, default=None,
                        help="Path to diffusers-format VAE directory (AutoencoderKLWan). "
                             "Defaults to <ckpt_dir>/vae if not set.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Root of one LeRobot dataset (e.g. .../libero_spatial_dataset)")
    parser.add_argument("--video_keys", type=str, nargs="+",
                        default=["observation.images.image", "observation.images.wrist_image"],
                        help="Camera keys under videos/chunk-xxx/")
    parser.add_argument("--target_fps", type=int, default=None,
                        help="If set, sample frames at this fps; else use all frames in [start_frame, end_frame]")
    parser.add_argument("--frame_stride", type=int, default=None,
                        help="If set, sample every N-th frame (e.g. 4 means frames 0,4,8,...). "
                             "Takes priority over --target_fps.")
    parser.add_argument("--height", type=int, default=None,
                        help="Target height for cam_high (first video_key). "
                             "If not set, no resizing is performed.")
    parser.add_argument("--width", type=int, default=None,
                        help="Target width for cam_high (first video_key). "
                             "If not set, no resizing is performed.")
    parser.add_argument("--wrist_height", type=int, default=None,
                        help="Target height for wrist cameras (remaining video_keys). "
                             "Defaults to height//2 if not set (robotwin_tshape behavior). "
                             "Set to the same as --height for single-wrist datasets (e.g. libero).")
    parser.add_argument("--wrist_width", type=int, default=None,
                        help="Target width for wrist cameras (remaining video_keys). "
                             "Defaults to width//2 if not set (robotwin_tshape behavior). "
                             "Set to the same as --width for single-wrist datasets (e.g. libero).")
    parser.add_argument("--text_encoder_dir", type=str,
                        default="/home/jwhe/linyihan/lingbot-va-base/text_encoder",
                        help="HuggingFace-format UMT5 text encoder directory.")
    parser.add_argument("--tokenizer_dir", type=str,
                        default="/home/jwhe/linyihan/lingbot-va-base/tokenizer",
                        help="HuggingFace-format T5 tokenizer directory.")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Process only first N episodes (for debugging)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ckpt = Path(args.ckpt_dir)
    dataset_root = Path(args.dataset_path)
    meta_dir = dataset_root / "meta"
    episodes_file = meta_dir / "episodes.jsonl"
    info_file = meta_dir / "info.json"
    if not episodes_file.exists():
        raise FileNotFoundError(f"Need {episodes_file} with action_config")
    if not info_file.exists():
        raise FileNotFoundError(f"Need {info_file}")

    with open(info_file) as f:
        info = json.load(f)
    ori_fps = int(info.get("fps", 20))
    video_path_tpl = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")

    episodes = []
    with open(episodes_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vae_dir = args.vae_dir if args.vae_dir is not None else str(ckpt / "vae")

    print("Loading Wan2.2 VAE (AutoencoderKLWan) from", vae_dir)
    vae_raw = AutoencoderKLWan.from_pretrained(vae_dir, torch_dtype=torch.bfloat16).to(device)
    vae_device = next(vae_raw.parameters()).device
    # 预先把 latents_mean / latents_std 转成 tensor，用于归一化（与 server normalize_latents 一致）
    _lat_mean = torch.tensor(vae_raw.config.latents_mean,
                             dtype=torch.float32, device=vae_device).view(-1, 1, 1, 1)
    _lat_std_inv = (1.0 / torch.tensor(vae_raw.config.latents_std,
                                       dtype=torch.float32, device=vae_device)).view(-1, 1, 1, 1)

    print("Loading tokenizer from", args.tokenizer_dir)
    tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer_dir)
    print("Loading text encoder (UMT5) from", args.text_encoder_dir)
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.text_encoder_dir, torch_dtype=torch.bfloat16
    ).to(device)
    text_encoder.eval()

    latents_root = dataset_root / "latents"
    chunk = "chunk-000"
    for vk in args.video_keys:
        (latents_root / chunk / vk).mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm
    for ep in tqdm(episodes, desc="Episodes"):
        episode_index = ep["episode_index"]
        episode_chunk = (episode_index // info.get("chunks_size", 100000)) if "chunks_size" in info else 0
        action_configs = ep.get("action_config")
        if not action_configs:
            continue
        length = ep.get("length", 0)
        for ac in action_configs:
            start_frame = ac["start_frame"]
            end_frame = ac["end_frame"]
            action_text = ac.get("action_text", "")
            if end_frame <= start_frame:
                continue

            # Sample frame indices
            if args.frame_stride is not None:
                # 固定步长：每隔 frame_stride 帧取一帧，不依赖 fps
                frame_ids = list(range(start_frame, end_frame, args.frame_stride))
            elif args.target_fps is not None and args.target_fps != ori_fps:
                # 按 fps 比例采样，步长取整数以与标准数据集一致
                step = max(1, round(ori_fps / args.target_fps))
                num_out = max(1, int((end_frame - start_frame) * args.target_fps / ori_fps))
                frame_ids = [start_frame + i * step for i in range(num_out)]
                frame_ids = [min(f, end_frame - 1) for f in frame_ids]
            else:
                frame_ids = list(range(start_frame, end_frame))

            video_num_frames = len(frame_ids)
            if video_num_frames == 0:
                continue

            # Load one camera to get H,W (assume same for all)
            first_key = args.video_keys[0]
            video_dir = dataset_root / "videos" / f"chunk-{episode_chunk:03d}" / first_key
            video_file = video_dir / f"episode_{episode_index:06d}.mp4"
            if not video_file.exists():
                tqdm.write(f"Skip {video_file} (not found)")
                continue

            first_frames = load_video_frames(
                video_file, start_frame, end_frame, ori_fps, device
            )
            if first_frames is None or first_frames.shape[1] == 0:
                tqdm.write(f"Skip episode {episode_index} (no frames)")
                continue
            _, T_in, video_height, video_width = first_frames.shape
            # 按 frame_ids 抽取对应帧（相对于 start_frame 的局部索引）
            local_ids = [min(f - start_frame, T_in - 1) for f in frame_ids]
            first_frames = first_frames[:, local_ids, :, :]
            # resize cam_high to target resolution (C, T, H, W) → interpolate on H, W
            if args.height is not None and args.width is not None:
                C, T, H, W = first_frames.shape
                first_frames = F.interpolate(
                    first_frames.permute(1, 0, 2, 3),  # (T, C, H, W)
                    size=(args.height, args.width),
                    mode='bilinear',
                    align_corners=False,
                ).permute(1, 0, 2, 3)  # (C, T, H, W)
                video_height, video_width = args.height, args.width
            video_tensor = first_frames
            video_tensor = (video_tensor * 2.0 - 1.0).to(torch.bfloat16)

            x_in = video_tensor.unsqueeze(0).to(vae_device).to(torch.bfloat16)
            with torch.no_grad():
                enc_out = vae_raw.encode(x_in)
            z = enc_out.latent_dist.mean.squeeze(0)  # (C, T', H', W')
            z = ((z.float() - _lat_mean) * _lat_std_inv).to(torch.bfloat16)
            latent_num_frames, latent_height, latent_width = int(z.shape[1]), int(z.shape[2]), int(z.shape[3])
            z_flat = z.reshape(z.shape[0], -1).permute(1, 0).contiguous()  # (T*H*W, C)
            latent = z_flat.to(torch.bfloat16)

            text_emb = get_text_emb(
                action_text, tokenizer, text_encoder, device, torch.bfloat16
            )

            out = {
                "latent": latent,
                "latent_num_frames": int(latent_num_frames),
                "latent_height": int(latent_height),
                "latent_width": int(latent_width),
                "video_num_frames": int(video_num_frames),
                "video_height": int(video_height),
                "video_width": int(video_width),
                "text_emb": text_emb,
                "text": action_text,
                "frame_ids": frame_ids,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "fps": args.target_fps or ori_fps,
                "ori_fps": ori_fps,
            }
            out_path = latents_root / chunk / first_key / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            torch.save(out, out_path)

            for vk in args.video_keys[1:]:
                video_dir_k = dataset_root / "videos" / f"chunk-{episode_chunk:03d}" / vk
                video_file_k = video_dir_k / f"episode_{episode_index:06d}.mp4"
                if not video_file_k.exists():
                    tqdm.write(f"Missing {video_file_k}")
                    continue
                frames_k = load_video_frames(video_file_k, start_frame, end_frame, ori_fps, device)
                if frames_k is None or frames_k.shape[1] == 0:
                    continue
                T_in_k = frames_k.shape[1]
                local_ids_k = [min(f - start_frame, T_in_k - 1) for f in frame_ids]
                frames_k = frames_k[:, local_ids_k, :, :]
                # resize wrist cameras to target resolution
                if args.height is not None and args.width is not None:
                    wrist_h = args.wrist_height if args.wrist_height is not None else args.height // 2
                    wrist_w = args.wrist_width if args.wrist_width is not None else args.width // 2
                    frames_k = F.interpolate(
                        frames_k.permute(1, 0, 2, 3),  # (T, C, H, W)
                        size=(wrist_h, wrist_w),
                        mode='bilinear',
                        align_corners=False,
                    ).permute(1, 0, 2, 3)  # (C, T, H, W)
                frames_k = (frames_k * 2.0 - 1.0).to(torch.bfloat16)
                xk_in = frames_k.unsqueeze(0).to(vae_device).to(torch.bfloat16)
                with torch.no_grad():
                    enc_out_k = vae_raw.encode(xk_in)
                zk = enc_out_k.latent_dist.mean.squeeze(0)  # (C, T', H', W')
                zk = ((zk.float() - _lat_mean) * _lat_std_inv).to(torch.bfloat16)
                zk_flat = zk.reshape(zk.shape[0], -1).permute(1, 0).contiguous().to(torch.bfloat16)
                if int(zk.shape[1]) != int(out["latent_num_frames"]):
                    print(f"Skip episode {episode_index} (latent_num_frames mismatch): {int(zk.shape[1])} != {int(out['latent_num_frames'])}")
                out_k = {
                    "latent": zk_flat,
                    "latent_num_frames": int(zk.shape[1]),
                    "latent_height": int(zk.shape[2]),
                    "latent_width": int(zk.shape[3]),
                    "video_num_frames": out["video_num_frames"],
                    "video_height": int(frames_k.shape[2]),
                    "video_width": int(frames_k.shape[3]),
                    "text_emb": text_emb,
                    "text": action_text,
                    "frame_ids": frame_ids,
                    "start_frame": out["start_frame"],
                    "end_frame": out["end_frame"],
                    "fps": out["fps"],
                    "ori_fps": ori_fps,
                }
                out_path_k = latents_root / chunk / vk / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
                torch.save(out_k, out_path_k)

    print("Done. Latents saved under", latents_root)


if __name__ == "__main__":
    main()
