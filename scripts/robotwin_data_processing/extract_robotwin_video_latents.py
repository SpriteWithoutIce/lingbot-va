#!/usr/bin/env python3
"""Extract RobotWin video-only Wan latents, one folder per camera view.

- Each of the three views is encoded separately (no concatenation). Each view
  gets its own latent from its own video file.
- Temporal: 4 video frames -> 1 latent frame. Excess frames are dropped:
  e.g. 139 frames -> 139//4 = 34 latent frames (use frames 0..135 only).
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


def process_one_dataset(root, args, vae, tokenizer, text_encoder, device, dtype):
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
            # 4 video frames -> 1 latent frame; drop excess (e.g. 139 -> 34 latent, use 136 frames)
            n_latent = (ed - st) // 4
            if n_latent < 1:
                continue
            effective_end = st + n_latent * 4
            latent_frame_ids = [st + 4 * i for i in range(n_latent)]

            text = prompt_clean(ac.get('action_text', ''))
            text_emb = get_text_emb(text, tokenizer, text_encoder, device, dtype)

            vae_device = next(vae.parameters()).device

            for k_i, vk in enumerate(args.video_keys):
                vfile = root / 'videos' / f'chunk-{chunk:03d}' / vk / f'episode_{ep_idx:06d}.mp4'
                if not vfile.exists():
                    continue

                clip = load_video_frames(vfile, st, effective_end, ori_fps, device=device)
                if clip is None or clip.shape[1] == 0:
                    continue

                tgt_h = args.height if k_i == 0 else args.height // 2
                tgt_w = args.width if k_i == 0 else args.width // 2
                clip = resize_view(clip, tgt_h, tgt_w)
                # (C, T, H, W) in [0,1] -> (1, C, T, H, W) in [-1,1]
                clip = clip.unsqueeze(0).to(vae_device).to(dtype) * 2.0 - 1.0

                with torch.no_grad():
                    enc = vae.encode(clip).latent_dist.mode()
                mu_norm = normalize_latents(enc, vae)
                # (1, C, F, h, w) -> (F, h, w, C)
                video_latent = mu_norm[0].permute(1, 2, 3, 0).contiguous()
                lft, lh, lw, c = video_latent.shape

                payload = {
                    'latent': video_latent.reshape(lft * lh * lw, c).cpu(),
                    'latent_num_frames': lft,
                    'latent_height': lh,
                    'latent_width': lw,
                    'video_num_frames': n_latent * 4,
                    'video_height': tgt_h,
                    'video_width': tgt_w,
                    'text_emb': text_emb.cpu(),
                    'text': text,
                    'frame_ids': torch.tensor(latent_frame_ids, dtype=torch.long),
                    'start_frame': st,
                    'end_frame': effective_end,
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

    tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer_dir)
    text_encoder = UMT5EncoderModel.from_pretrained(args.text_encoder_dir, torch_dtype=dtype).to(device)
    text_encoder.eval()

    print(f"[extract] found {len(dataset_roots)} dataset roots")
    for root in dataset_roots:
        process_one_dataset(
            root,
            args,
            vae,
            tokenizer,
            text_encoder,
            device,
            dtype,
        )


if __name__ == '__main__':
    main()
