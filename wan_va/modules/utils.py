# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from safetensors.torch import load_file
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel
from .video_finetune_model import WanVideoFinetuneTransformer3DModel


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    print("loading text encoder from", text_encoder_path)
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, ):
    print("loading tokenizer from", tokenizer_path)
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, )
    return tokenizer


def _load_wan_official_state_dict(ckpt_root: Path):
    index_file = ckpt_root / "diffusion_pytorch_model.safetensors.index.json"
    single_file = ckpt_root / "diffusion_pytorch_model.safetensors"

    state_dict = {}
    if index_file.exists():
        index = json.loads(index_file.read_text())
        shard_files = sorted(set(index["weight_map"].values()))
        for sf in shard_files:
            state_dict.update(load_file(str(ckpt_root / sf), device="cpu"))
    elif single_file.exists():
        state_dict.update(load_file(str(single_file), device="cpu"))
    else:
        raise FileNotFoundError(
            f"No Wan official safetensors found under {ckpt_root}. "
            "Expected diffusion_pytorch_model.safetensors(.index.json)."
        )
    return state_dict


def _build_video_model_from_wan_config(ckpt_root: Path, attn_mode: str):
    cfg_file = ckpt_root / "config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing Wan config.json under {ckpt_root}")
    cfg = json.loads(cfg_file.read_text())
    if cfg.get("_class_name") != "WanModel":
        raise ValueError(
            f"Expected WanModel config, got {cfg.get('_class_name')} in {cfg_file}"
        )

    dim = int(cfg["dim"])
    num_heads = int(cfg["num_heads"])
    model = WanVideoFinetuneTransformer3DModel(
        patch_size=[1, 2, 2],
        num_attention_heads=num_heads,
        attention_head_dim=dim // num_heads,
        in_channels=int(cfg.get("in_dim", 48)),
        out_channels=int(cfg.get("out_dim", 48)),
        text_dim=4096,
        freq_dim=int(cfg.get("freq_dim", 256)),
        ffn_dim=int(cfg.get("ffn_dim", 14336)),
        num_layers=int(cfg.get("num_layers", 30)),
        cross_attn_norm=True,
        eps=float(cfg.get("eps", 1e-6)),
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode=attn_mode,
    )
    return model


def _remap_wan_key_to_video_model(k: str):
    candidates = [k]
    replacements = [
        ("patch_embedding.", "patch_embedding_mlp."),
        ("time_embedding.", "condition_embedder.time_embedder."),
        ("time_projection.", "condition_embedder.time_proj."),
        ("text_embedding.", "condition_embedder.text_embedder."),
        ("norm.", "norm_out."),
        ("head.", "proj_out."),
        ("modulation.", "scale_shift_table."),
    ]
    for src, dst in replacements:
        if src in k:
            candidates.append(k.replace(src, dst))
    if k.startswith("model."):
        candidates.append(k[len("model."):])
    return candidates


def _load_video_model_from_wan_official(
    ckpt_root: Path,
    torch_dtype,
    torch_device,
    attn_mode="torch",
):
    print("loading video finetune model from official Wan checkpoint", ckpt_root)
    model = _build_video_model_from_wan_config(ckpt_root, attn_mode)
    src_sd = _load_wan_official_state_dict(ckpt_root)
    tgt_sd = model.state_dict()

    merged = {}
    used_src = set()
    for sk, sv in src_sd.items():
        for candidate in _remap_wan_key_to_video_model(sk):
            if candidate in tgt_sd and tgt_sd[candidate].shape == sv.shape:
                merged[candidate] = sv
                used_src.add(sk)
                break

    missing, unexpected = model.load_state_dict(merged, strict=False)
    print(
        f"[wan init] matched={len(merged)} used_src={len(used_src)}/{len(src_sd)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )

    model = model.to(torch_device)
    if torch_dtype is not None:
        model = model.to(torch_dtype)
    return model


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    attn_mode="torch",
    model_name="wan_va",
    transformer_source="lingbot_va",
):
    print("loading transformer from", transformer_path)
    if model_name == "wan_video_finetune" and transformer_source == "wan_official":
        return _load_video_model_from_wan_official(
            ckpt_root=Path(transformer_path),
            torch_dtype=torch_dtype,
            torch_device=torch_device,
            attn_mode=attn_mode,
        )

    model_map = {
        "wan_va": WanTransformer3DModel,
        "wan_video_finetune": WanVideoFinetuneTransformer3DModel,
    }
    if model_name not in model_map:
        raise ValueError(f"Unsupported transformer model_name: {model_name}. Supported: {list(model_map.keys())}")
    model_cls = model_map[model_name]
    model = model_cls.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
        attn_mode=attn_mode,
    )
    return model.to(torch_device)


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size,
               width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames,
               height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk,
                           feat_cache=self.feat_cache,
                           feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc
