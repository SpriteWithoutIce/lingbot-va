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
                        help="Path to Wan2.2-TI2V-5B checkpoint (contains Wan2.2_VAE.pth, models_t5_umt5-xxl-enc-bf16.pth, google/umt5-xxl tokenizer)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Root of one LeRobot dataset (e.g. .../libero_spatial_dataset)")
    parser.add_argument("--video_keys", type=str, nargs="+",
                        default=["observation.images.image", "observation.images.wrist_image"],
                        help="Camera keys under videos/chunk-xxx/")
    parser.add_argument("--target_fps", type=int, default=None,
                        help="If set, sample frames at this fps; else use all frames in [start_frame, end_frame]")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Process only first N episodes (for debugging)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    _add_wan22(args.wan22_root)
    from wan.modules.vae2_2 import Wan2_2_VAE
    from wan.modules.t5 import T5EncoderModel
    from wan.configs import WAN_CONFIGS
    config = WAN_CONFIGS["ti2v-5B"]

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
    vae_path = ckpt / config.vae_checkpoint
    t5_path = ckpt / config.t5_checkpoint
    t5_tok = config.t5_tokenizer
    if (ckpt / t5_tok).exists():
        tokenizer_path = str(ckpt / t5_tok)
    else:
        tokenizer_path = t5_tok

    print("Loading Wan2.2 VAE from", vae_path)
    vae = Wan2_2_VAE(
        vae_pth=str(vae_path),
        device=device,
    )
    print("Loading T5 encoder from", t5_path)
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=device,
        checkpoint_path=str(t5_path),
        tokenizer_path=tokenizer_path,
    )

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
            if args.target_fps is not None and args.target_fps != ori_fps:
                import math
                num_out = max(1, int((end_frame - start_frame) * args.target_fps / ori_fps))
                step = (end_frame - start_frame) / num_out
                frame_ids = [start_frame + int(i * step) for i in range(num_out)]
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
            if T_in != video_num_frames:
                first_frames = first_frames[:, :video_num_frames]
            video_tensor = first_frames
            video_tensor = (video_tensor * 2.0 - 1.0).to(torch.bfloat16)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
                lat_list = vae.encode([video_tensor])
            if lat_list is None or len(lat_list) == 0:
                tqdm.write(f"Skip episode {episode_index} (VAE encode failed)")
                continue
            z = lat_list[0]
            latent_num_frames, latent_height, latent_width = int(z.shape[1]), int(z.shape[2]), int(z.shape[3])
            z_flat = z.reshape(z.shape[0], -1).permute(1, 0).contiguous()  # (T*H*W, C)
            latent = z_flat.to(torch.bfloat16)

            text_emb_list = text_encoder([action_text], device)
            text_emb = text_emb_list[0].to(torch.bfloat16)

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
                if frames_k.shape[1] != video_num_frames:
                    frames_k = frames_k[:, :video_num_frames]
                frames_k = (frames_k * 2.0 - 1.0).to(torch.bfloat16)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
                    lat_k = vae.encode([frames_k])
                if lat_k is None or len(lat_k) == 0:
                    continue
                zk = lat_k[0]
                zk_flat = zk.reshape(zk.shape[0], -1).permute(1, 0).contiguous().to(torch.bfloat16)
                if int(zk.shape[1]) != int(out["latent_num_frames"]):
                    print(f"Skip episode {episode_index} (latent_num_frames mismatch): {int(zk.shape[1])} != {int(out['latent_num_frames'])}")
                out_k = {
                    "latent": zk_flat,
                    "latent_num_frames": out["latent_num_frames"],
                    "latent_height": out["latent_height"],
                    "latent_width": out["latent_width"],
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
