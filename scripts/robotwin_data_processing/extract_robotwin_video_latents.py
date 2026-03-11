#!/usr/bin/env python3
"""Extract RobotWin video-only Wan latents into separate folders for finetuning.

Supports both:
1) a single LeRobot dataset root (.../meta/info.json), or
2) a parent directory containing many dataset roots.

Important: RobotWin latent packing follows wan_va_server._encode_obs exactly:
- cam_high resized to (H, W)
- wrist cams resized to (H/2, W/2)
- high and wrist views encoded separately
- latent concat layout matches server path
- temporal behavior is still 4 video frames -> 1 latent frame
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from transformers import T5TokenizerFast, UMT5EncoderModel

from scripts.libero_data_processing.extract_latents_wan22 import get_text_emb, load_video_frames
from wan_va.modules.utils import WanVAEStreamingWrapper


def normalize_latents(latents, vae):
    latents_mean = torch.tensor(vae.config.latents_mean).to(latents.device)
    latents_std = torch.tensor(vae.config.latents_std).to(latents.device)
    latents_mean = latents_mean.view(1, -1, 1, 1, 1)
    latents_std = latents_std.view(1, -1, 1, 1, 1)
    return ((latents.float() - latents_mean) * (1.0 / latents_std)).to(latents)


def discover_dataset_roots(path: Path):
    direct_meta = path / "meta"
    if (direct_meta / "info.json").exists() and (direct_meta / "episodes.jsonl").exists():
        return [path]

    roots = []
    for info in path.glob("**/meta/info.json"):
        root = info.parent.parent
        if (root / "meta/episodes.jsonl").exists():
            roots.append(root)
    return sorted(set(roots))


def resize_view(clip, h, w):
    # clip: (C, T, H, W) in [0,1]
    clip = F.interpolate(
        clip.permute(1, 0, 2, 3),
        size=(h, w),
        mode='bilinear',
        align_corners=False,
    ).permute(1, 0, 2, 3)
    return clip


def encode_robotwin_views(view_clips, streaming_vae, streaming_vae_half, dtype):
    """Match wan_va_server._encode_obs robotwin_tshape branch.

    Args:
        view_clips: list of [cam_high, cam_left_wrist, cam_right_wrist], each (C,T,H,W) in [0,1]
    Returns:
        video_latent: (1, C, F, H_lat, W_lat)
    """
    videos = [v.unsqueeze(0) for v in view_clips]  # -> (1,C,T,H,W)
    vae_device = next(streaming_vae.vae.parameters()).device

    videos_high = videos[0] * 2.0 - 1.0
    videos_left_and_right = torch.cat(videos[1:], dim=0) * 2.0 - 1.0

    with torch.no_grad():
        streaming_vae.clear_cache()
        streaming_vae_half.clear_cache()
        enc_out_high = streaming_vae.encode_chunk(videos_high.to(vae_device).to(dtype))
        enc_out_left_and_right = streaming_vae_half.encode_chunk(videos_left_and_right.to(vae_device).to(dtype))

    enc_out = torch.cat(
        [torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1), enc_out_high],
        dim=-2,
    )
    mu, _ = torch.chunk(enc_out, 2, dim=1)
    mu_norm = normalize_latents(mu, streaming_vae.vae)
    video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-2)
    return video_latent


def process_one_dataset(root, args, streaming_vae, streaming_vae_half, tokenizer, text_encoder, device, dtype):
    info = json.loads((root / 'meta' / 'info.json').read_text())
    episodes = [json.loads(x) for x in (root / 'meta' / 'episodes.jsonl').read_text().splitlines() if x.strip()]
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]

    out_root = root / args.output_subdir
    ori_fps = int(info.get('fps', 20))
    chunks_size = int(info.get('chunks_size', 100000))

    print(f"[extract] dataset={root} episodes={len(episodes)}")

    for ep in episodes:
        ep_idx = ep['episode_index']
        chunk = ep_idx // chunks_size
        for ac in ep.get('action_config') or []:
            st, ed = int(ac['start_frame']), int(ac['end_frame'])

            latent_frame_ids = list(range(st, ed, 4))
            raw_frame_ids = []
            for f in latent_frame_ids:
                raw_frame_ids.extend([f, min(f + 1, ed - 1), min(f + 2, ed - 1), min(f + 3, ed - 1)])
            if len(raw_frame_ids) < 4:
                continue

            text = prompt_clean(ac.get('action_text', ''))
            text_emb = get_text_emb(text, tokenizer, text_encoder, device, dtype)

            view_clips = []
            for k_i, vk in enumerate(args.video_keys):
                vfile = root / 'videos' / f'chunk-{chunk:03d}' / vk / f'episode_{ep_idx:06d}.mp4'
                if not vfile.exists():
                    view_clips = []
                    break

                clip = load_video_frames(vfile, st, ed, ori_fps, device=device)
                if clip is None or clip.shape[1] == 0:
                    view_clips = []
                    break

                local = [min(fid - st, clip.shape[1] - 1) for fid in raw_frame_ids]
                clip = clip[:, local]

                tgt_h = args.height if k_i == 0 else args.height // 2
                tgt_w = args.width if k_i == 0 else args.width // 2
                view_clips.append(resize_view(clip, tgt_h, tgt_w))

            if len(view_clips) != len(args.video_keys):
                continue

            video_latent = encode_robotwin_views(
                view_clips,
                streaming_vae=streaming_vae,
                streaming_vae_half=streaming_vae_half,
                dtype=dtype,
            )
            lf, c, lft, lh, lw = 1, video_latent.shape[1], video_latent.shape[2], video_latent.shape[3], video_latent.shape[4]
            assert lf == 1
            lat = video_latent[0].permute(1, 2, 3, 0).contiguous()  # (F,H,W,C)

            # save one pth for each camera key to match current dataset loader behavior
            for vk in args.video_keys:
                payload = {
                    'latent': lat.reshape(lft * lh * lw, c).cpu(),
                    'latent_num_frames': lft,
                    'latent_height': lh,
                    'latent_width': lw,
                    'video_num_frames': len(raw_frame_ids),
                    'video_height': args.height,
                    'video_width': args.width,
                    'text_emb': text_emb.cpu(),
                    'text': text,
                    'frame_ids': torch.tensor(latent_frame_ids[:lft], dtype=torch.long),
                    'start_frame': st,
                    'end_frame': ed,
                    'fps': max(1, ori_fps // 4),
                    'ori_fps': ori_fps,
                }
                out_dir = out_root / f'chunk-{chunk:03d}' / vk
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f'episode_{ep_idx:06d}_{st}_{ed}.pth'
                torch.save(payload, out_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', type=str, required=True,
                    help='Single dataset root OR a parent directory containing many dataset roots.')
    ap.add_argument('--vae_dir', type=str, required=True)
    ap.add_argument('--text_encoder_dir', type=str, required=True)
    ap.add_argument('--tokenizer_dir', type=str, required=True)
    ap.add_argument('--video_keys', nargs='+', default=[
        'observation.images.cam_high',
        'observation.images.cam_left_wrist',
        'observation.images.cam_right_wrist',
    ])
    ap.add_argument('--output_subdir', type=str, default='latents_video_ft')
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--width', type=int, default=320)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--max_episodes', type=int, default=None)
    args = ap.parse_args()

    scan_root = Path(args.dataset_path)
    dataset_roots = discover_dataset_roots(scan_root)
    if not dataset_roots:
        raise FileNotFoundError(f'No LeRobot dataset root found under: {scan_root}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(args.vae_dir, torch_dtype=dtype).to(device)
    vae_half = AutoencoderKLWan.from_pretrained(args.vae_dir, torch_dtype=dtype).to(device)
    streaming_vae = WanVAEStreamingWrapper(vae)
    streaming_vae_half = WanVAEStreamingWrapper(vae_half)

    tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer_dir)
    text_encoder = UMT5EncoderModel.from_pretrained(args.text_encoder_dir, torch_dtype=dtype).to(device)
    text_encoder.eval()

    print(f"[extract] found {len(dataset_roots)} dataset roots")
    for root in dataset_roots:
        process_one_dataset(
            root,
            args,
            streaming_vae,
            streaming_vae_half,
            tokenizer,
            text_encoder,
            device,
            dtype,
        )


if __name__ == '__main__':
    main()
