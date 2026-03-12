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


def remap_video_model_state_dict_to_wan_official(state_dict, in_channels=48, patch_size=(1, 2, 2)):
    """将 WanVideoFinetuneTransformer3DModel 的 state_dict 转成 Wan 官方 ckpt 的 key，便于用 Wan 方式加载。

    仅包含我们模型里存在的 key；norm3.weight/bias 等 Wan 有、我们无的不会出现在输出中。
    """
    out = {}
    # 用于 patch_embedding 的 Conv3d 形状
    kt, kh, kw = patch_size[0], patch_size[1], patch_size[2]
    for k, v in state_dict.items():
        v = v.detach().cpu()
        new_k = None
        if k == "patch_embedding_mlp.weight":
            # Linear (dim, in*kt*kh*kw) -> Conv3d (dim, in, kt, kh, kw)
            dim, flat = v.shape
            out["patch_embedding.weight"] = v.reshape(dim, in_channels, kt, kh, kw).clone()
            continue
        if k == "patch_embedding_mlp.bias":
            out["patch_embedding.bias"] = v.clone()
            continue
        if k.startswith("condition_embedder.time_embedder.linear_1."):
            new_k = k.replace("condition_embedder.time_embedder.linear_1.", "time_embedding.0.")
        elif k.startswith("condition_embedder.time_embedder.linear_2."):
            new_k = k.replace("condition_embedder.time_embedder.linear_2.", "time_embedding.2.")
        elif k.startswith("condition_embedder.time_proj."):
            new_k = k.replace("condition_embedder.time_proj.", "time_projection.1.")
        elif k.startswith("condition_embedder.text_embedder.linear_1."):
            new_k = k.replace("condition_embedder.text_embedder.linear_1.", "text_embedding.0.")
        elif k.startswith("condition_embedder.text_embedder.linear_2."):
            new_k = k.replace("condition_embedder.text_embedder.linear_2.", "text_embedding.2.")
        elif k == "scale_shift_table":
            new_k = "head.modulation"
        elif k.startswith("proj_out."):
            new_k = k.replace("proj_out.", "head.head.")
        elif ".attn1." in k:
            new_k = k.replace(".attn1.", ".self_attn.")
            new_k = new_k.replace(".to_q.", ".q.").replace(".to_k.", ".k.").replace(".to_v.", ".v.")
            new_k = new_k.replace(".to_out.0.", ".o.")
        elif ".attn2." in k:
            new_k = k.replace(".attn2.", ".cross_attn.")
            new_k = new_k.replace(".to_q.", ".q.").replace(".to_k.", ".k.").replace(".to_v.", ".v.")
            new_k = new_k.replace(".to_out.0.", ".o.")
        elif ".scale_shift_table" in k and "blocks." in k:
            new_k = k.replace(".scale_shift_table", ".modulation")
        elif ".ffn.net.0.proj." in k:
            new_k = k.replace(".ffn.net.0.proj.", ".ffn.0.")
        elif ".ffn.net.2." in k:
            new_k = k.replace(".ffn.net.2.", ".ffn.2.")
        elif "blocks." in k and ".norm2." in k:
            new_k = k  # 与 Wan 一致，无需改名
        if new_k is not None:
            out[new_k] = v.clone()
    return out


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
    """Map official Wan checkpoint keys to WanVideoFinetuneTransformer3DModel state_dict keys."""
    candidates = [k]

    # 1) 去掉可选的 "model." 前缀（部分 checkpoint 带此前缀）
    if k.startswith("model."):
        candidates.append(k[len("model."):])

    # 2) 顶层/embedding 的替换（与官方 Wan 命名一致）
    replacements = [
        ("patch_embedding.", "patch_embedding_mlp."),
        ("time_embedding.0.", "condition_embedder.time_embedder.linear_1."),
        ("time_embedding.2.", "condition_embedder.time_embedder.linear_2."),
        ("time_projection.1.", "condition_embedder.time_proj."),
        ("text_embedding.0.", "condition_embedder.text_embedder.linear_1."),
        ("text_embedding.2.", "condition_embedder.text_embedder.linear_2."),
        ("head.head.", "proj_out."),
        ("head.modulation", "scale_shift_table"),
    ]
    for src, dst in replacements:
        if k.startswith(src) or (src in k and k.replace(src, dst) != k):
            candidates.append(k.replace(src, dst))

    # 3) 兼容旧逻辑里的通用替换（用于非官方格式）
    for src, dst in [
        ("time_embedding.", "condition_embedder.time_embedder."),
        ("time_projection.", "condition_embedder.time_proj."),
        ("text_embedding.", "condition_embedder.text_embedder."),
        ("norm.", "norm_out."),
        ("head.", "proj_out."),
        ("modulation.", "scale_shift_table."),
    ]:
        if src in k:
            candidates.append(k.replace(src, dst))

    # 4) blocks 内：官方 Wan 使用 self_attn / cross_attn / modulation / ffn.0|2，我们使用 attn1 / attn2 / scale_shift_table / ffn.net.0|2
    if ".self_attn." in k:
        c = k.replace(".self_attn.", ".attn1.")
        c = c.replace(".q.", ".to_q.").replace(".k.", ".to_k.").replace(".v.", ".to_v.").replace(".o.", ".to_out.0.")
        candidates.append(c)
    if ".cross_attn." in k:
        c = k.replace(".cross_attn.", ".attn2.")
        c = c.replace(".q.", ".to_q.").replace(".k.", ".to_k.").replace(".v.", ".to_v.").replace(".o.", ".to_out.0.")
        candidates.append(c)
    if ".modulation" in k and "blocks." in k:
        candidates.append(k.replace(".modulation", ".scale_shift_table"))
    # diffusers FeedForward: net.0 是 GELU(proj=Linear), net.2 是 Linear；故 ffn.0 -> net.0.proj, ffn.2 -> net.2
    if ".ffn.0." in k:
        candidates.append(k.replace(".ffn.0.", ".ffn.net.0.proj."))
    if ".ffn.2." in k:
        candidates.append(k.replace(".ffn.2.", ".ffn.net.2."))

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

    # patch_embedding: 官方为 Conv3d (out, in, 1, 2, 2)，我们为 Linear (out, in*1*2*2)，需 reshape 后加载
    if "patch_embedding.weight" in src_sd and "patch_embedding.weight" not in used_src:
        w = src_sd["patch_embedding.weight"]
        if w.ndim == 5 and "patch_embedding_mlp.weight" in tgt_sd:
            # Conv3d weight (dim, in_dim, kT, kH, kW) -> Linear (dim, in_dim * kT * kH * kW)
            out_dim, in_dim = w.shape[0], w.shape[1]
            merged["patch_embedding_mlp.weight"] = w.reshape(out_dim, -1).clone()
            used_src.add("patch_embedding.weight")
    if "patch_embedding.bias" in src_sd and "patch_embedding.bias" not in used_src:
        b = src_sd["patch_embedding.bias"]
        if "patch_embedding_mlp.bias" in tgt_sd and b.shape == tgt_sd["patch_embedding_mlp.bias"].shape:
            merged["patch_embedding_mlp.bias"] = b.clone()
            used_src.add("patch_embedding.bias")

    missing, unexpected = model.load_state_dict(merged, strict=False)
    unused_src = [k for k in src_sd if k not in used_src]
    n_norm3 = sum(1 for k in unused_src if ".norm3." in k)
    other_unused = [k for k in unused_src if ".norm3." not in k]
    print(
        f"[wan init] matched={len(merged)} used_src={len(used_src)}/{len(src_sd)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    if unused_src:
        print(
            f"[wan init] unused: {len(unused_src)} ({n_norm3}× norm3.weight/bias [no affine in model]"
            + (f", {len(other_unused)} other)" if other_unused else ")")
        )
        if other_unused:
            for k in other_unused[:15]:
                print("  ", k)
            if len(other_unused) > 15:
                print("  ...")

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
