# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange

from .model import (
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    WanTransformerBlock,
)


class WanVideoFinetuneTransformer3DModel(ModelMixin, ConfigMixin):
    """Video-only WAN transformer for latent velocity-field training (flow matching).

    This model only consumes noisy video latents (+ text condition + timestep) and predicts
    latent velocity v, i.e. no action branch is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "patch_embedding_mlp",
        "condition_embedder",
        "norm",
    ]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "text_norm1",
        "norm2",
        "text_norm2",
        "norm3",
        "text_norm3",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size=[1, 2, 2],
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-06,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="torch",
    ):
        super().__init__()
        self.patch_size = patch_size

        inner_dim = num_attention_heads * attention_head_dim
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2],
            inner_dim,
        )
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    cross_attn_norm,
                    eps,
                    attn_mode=attn_mode,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    def _input_embed(self, latents):
        hidden_states = rearrange(
            latents,
            "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            p3=self.patch_size[2],
        )
        return self.patch_embedding_mlp(hidden_states)

    def _time_embed(self, timesteps, noisy_latents, dtype):
        latent_time_steps = torch.repeat_interleave(
            timesteps,
            (noisy_latents.shape[-2] // self.patch_size[1])
            * (noisy_latents.shape[-1] // self.patch_size[2]),
            dim=1,
        )
        temb, timestep_proj = self.condition_embedder(latent_time_steps, dtype=dtype)
        return temb, timestep_proj.unflatten(2, (6, -1))

    def forward_train(self, input_dict):
        latent_dict = input_dict["latent_dict"]
        latent_dict["noisy_latents"] = latent_dict["noisy_latents"].to(torch.bfloat16)
        latent_dict["latent"] = latent_dict["latent"].to(torch.bfloat16)
        return self.forward(latent_dict, train_mode=False)

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        train_mode=False,
        return_layer_hidden_states=False,
        return_layer_cross_attn=False,
    ):
        if train_mode:
            return self.forward_train(input_dict)

        latent_hidden_states = self._input_embed(input_dict["noisy_latents"])
        text_hidden_states = self.condition_embedder.text_embedder(input_dict["text_emb"])

        rotary_emb = self.rope(input_dict["grid_id"])[:, :, None]
        temb, timestep_proj = self._time_embed(
            input_dict["timesteps"],
            input_dict["noisy_latents"],
            latent_hidden_states.dtype,
        )

        encoder_attention_mask = input_dict.get("encoder_attention_mask", None)
        layer_hidden_states_list = [] if return_layer_hidden_states else None
        layer_cross_attn_list = [] if return_layer_cross_attn else None

        for block in self.blocks:
            block_out = block(
                latent_hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=update_cache,
                cache_name=cache_name,
                return_cross_attn=return_layer_cross_attn,
                encoder_attention_mask=encoder_attention_mask,
            )
            if return_layer_cross_attn:
                latent_hidden_states, cross_attn_probs = block_out
                layer_cross_attn_list.append(cross_attn_probs)
            else:
                latent_hidden_states = block_out
            if return_layer_hidden_states:
                layer_hidden_states_list.append(latent_hidden_states.clone())

        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(2, dim=1)
        latent_hidden_states = (
            self.norm_out(latent_hidden_states.float()) * (1.0 + scale.squeeze(1)) + shift.squeeze(1)
        ).type_as(latent_hidden_states)

        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(
            latent_hidden_states,
            "b l (n c) -> b (l n) c",
            n=math.prod(self.patch_size),
        )

        if return_layer_hidden_states and return_layer_cross_attn:
            return latent_hidden_states, layer_hidden_states_list, layer_cross_attn_list
        if return_layer_hidden_states:
            return latent_hidden_states, layer_hidden_states_list
        if return_layer_cross_attn:
            return latent_hidden_states, layer_cross_attn_list
        return latent_hidden_states
