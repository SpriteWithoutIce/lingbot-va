#!/usr/bin/env python3
"""Extract RobotWin video-only Wan latents into separate folders for finetuning.

Supports both:
1) a single LeRobot dataset root (.../meta/info.json), or
2) a parent directory containing many dataset roots.

Temporal processing follows WAN convention: every 4 video frames -> 1 latent frame.
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


def normalize_latents(latents, vae):
    lat_mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32, device=latents.device).view(1, -1, 1, 1, 1)
    lat_std_inv = (1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32, device=latents.device)).view(1, -1, 1, 1, 1)
    return (latents.float() - lat_mean) * lat_std_inv


def encode_video_to_latent(video, vae):
    t = (video.shape[1] // 4) * 4
    video = video[:, :t]
    if t == 0:
        return None
    x = (video * 2.0 - 1.0).unsqueeze(0).to(torch.bfloat16)
    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        latents = posterior.sample()
        latents = normalize_latents(latents, vae)
    return latents[0].permute(1, 2, 3, 0).contiguous()  # (F,H,W,C)


def discover_dataset_roots(path: Path):
    """Return all directories that look like LeRobot dataset roots."""
    direct_meta = path / "meta"
    if (direct_meta / "info.json").exists() and (direct_meta / "episodes.jsonl").exists():
        return [path]

    roots = []
    for info in path.glob("**/meta/info.json"):
        root = info.parent.parent
        if (root / "meta/episodes.jsonl").exists():
            roots.append(root)
    return sorted(set(roots))


def process_one_dataset(root, args, vae, tokenizer, text_encoder, device):
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
            text_emb = get_text_emb(text, tokenizer, text_encoder, device, torch.bfloat16)

            for k_i, vk in enumerate(args.video_keys):
                vfile = root / 'videos' / f'chunk-{chunk:03d}' / vk / f'episode_{ep_idx:06d}.mp4'
                if not vfile.exists():
                    continue

                clip = load_video_frames(vfile, st, ed, ori_fps, device=device)
                if clip is None or clip.shape[1] == 0:
                    continue

                local = [min(fid - st, clip.shape[1] - 1) for fid in raw_frame_ids]
                clip = clip[:, local]

                tgt_h = args.height if k_i == 0 else args.wrist_height
                tgt_w = args.width if k_i == 0 else args.wrist_width
                clip = F.interpolate(
                    clip.permute(1, 0, 2, 3),
                    size=(tgt_h, tgt_w),
                    mode='bilinear',
                    align_corners=False,
                ).permute(1, 0, 2, 3)

                lat = encode_video_to_latent(clip, vae)
                if lat is None:
                    continue

                lf, lh, lw, c = lat.shape
                payload = {
                    'latent': lat.reshape(lf * lh * lw, c).cpu(),
                    'latent_num_frames': lf,
                    'latent_height': lh,
                    'latent_width': lw,
                    'video_num_frames': clip.shape[1],
                    'video_height': tgt_h,
                    'video_width': tgt_w,
                    'text_emb': text_emb.cpu(),
                    'text': text,
                    'frame_ids': torch.tensor(latent_frame_ids[:lf], dtype=torch.long),
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
    ap.add_argument('--wrist_height', type=int, default=128)
    ap.add_argument('--wrist_width', type=int, default=160)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--max_episodes', type=int, default=None)
    args = ap.parse_args()

    scan_root = Path(args.dataset_path)
    dataset_roots = discover_dataset_roots(scan_root)
    if not dataset_roots:
        raise FileNotFoundError(f'No LeRobot dataset root found under: {scan_root}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    vae = AutoencoderKLWan.from_pretrained(args.vae_dir, torch_dtype=torch.bfloat16).to(device)
    tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer_dir)
    text_encoder = UMT5EncoderModel.from_pretrained(args.text_encoder_dir, torch_dtype=torch.bfloat16).to(device)
    text_encoder.eval()

    print(f"[extract] found {len(dataset_roots)} dataset roots")
    for root in dataset_roots:
        process_one_dataset(root, args, vae, tokenizer, text_encoder, device)


if __name__ == '__main__':
    main()
