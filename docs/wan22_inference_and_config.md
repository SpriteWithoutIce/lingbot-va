# Wan2.2 官方代码：推理流程与模型配置说明

> 说明：**Wan2.2 官方仓库 (Wan2.2) 只提供推理代码，没有训练脚本**。以下从 `Wan2.2` 源码整理「模型怎么调、怎么用」，供 lingbot-va 对齐 TI2V-5B 时参考。训练细节需参考论文或第三方实现（如 DiffSynth-Studio、LightX2V 等）。

---

## 1. 仓库结构概览

- **入口**：`generate.py`，按 `--task` 选择任务（如 `ti2v-5B`、`t2v-A14B`、`i2v-A14B` 等）。
- **配置**：`wan/configs/` 下每个任务一个 EasyDict 配置（如 `wan_ti2v_5B.py`）。
- **模型**：`wan/modules/model.py` 里 `WanModel`（DiT 主干），`from_pretrained(checkpoint_dir)` 加载 safetensors。
- **推理管道**：`wan/textimage2video.py`（TI2V）、`wan/text2video.py`（T2V）、`wan/image2video.py`（I2V）等，内部组装修理 T5、VAE、WanModel 和采样器。

---

## 2. TI2V-5B 配置（与 Wan2.2-TI2V-5B 一致）

来自 `wan/configs/wan_ti2v_5B.py` 与 `shared_config.py`：

| 项 | 值 | 说明 |
|----|-----|------|
| **Transformer** | | |
| `dim` | 3072 | 隐藏维 |
| `num_heads` | 24 | 注意力头数 |
| `num_layers` | 30 | Block 数 |
| `ffn_dim` | 14336 | FFN 中间维 |
| `freq_dim` | 256 | 时间 embedding 频域维 |
| `patch_size` | (1, 2, 2) | 时间/高/宽 patch |
| `qk_norm` | True | Q/K RMSNorm |
| `cross_attn_norm` | True | Cross-attn 后 LayerNorm |
| `eps` | 1e-6 | Norm 的 eps |
| **VAE** | | |
| `vae_stride` | (4, 16, 16) | VAE 时间/空间 stride |
| **推理** | | |
| `num_train_timesteps` | 1000 | Flow matching 训练步数 |
| `sample_steps` | 50 | 采样步数 |
| `sample_shift` | 5.0 | 采样 shift |
| `sample_guide_scale` | 5.0 | CFG scale |
| `sample_solver` | unipc | 采样器（UniPC） |
| `sample_fps` | 24 | 输出帧率 |
| `frame_num` | 121 | 默认生成帧数（4n+1） |
| **T5** | | |
| `text_len` | 512 | 文本 token 长度 |

---

## 3. WanModel 结构（`wan/modules/model.py`）

- **输入**：`x`（list of `[C, F, H, W]`），`t`（timestep），`context`（文本 embedding list），`seq_len`。
- **输出**：list of 去噪后的 latent tensor（与输入同空间形状）。

### 3.1 模块组成

1. **patch_embedding**：`nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)`，对应 ckpt 里 `patch_embedding.weight/bias`。
2. **time_embedding**：`nn.Sequential(Linear(freq_dim, dim), SiLU(), Linear(dim, dim))`，对应 `time_embedding.0/2`。
3. **time_projection**：`nn.Sequential(SiLU(), Linear(dim, dim*6))`，对应 `time_projection.1`。
4. **text_embedding**：`nn.Sequential(Linear(text_dim, dim), GELU(approximate='tanh'), Linear(dim, dim))`，对应 `text_embedding.0/2`。
5. **blocks**：`WanAttentionBlock` × num_layers，每个 block 包含：
   - `norm1`：WanLayerNorm(dim, elementwise_affine=False)
   - `self_attn`：q/k/v/o + norm_q/norm_k（WanRMSNorm）
   - `norm3`：WanLayerNorm(dim, elementwise_affine=True)（cross_attn 前）
   - `cross_attn`：同上 Q/K/V/O + norm
   - `norm2`：WanLayerNorm(dim, elementwise_affine=False)
   - `ffn`：`nn.Sequential(Linear(dim, ffn_dim), GELU(approximate='tanh'), Linear(ffn_dim, dim))`，对应 ckpt 的 `ffn.0`、`ffn.2`
   - `modulation`：`nn.Parameter(1, 6, dim)`，对应 ckpt 的 `blocks.*.modulation`
6. **head**：`Head` = `norm` + `head`(Linear) + `modulation`(1, 2, dim)，对应 `head.head`、`head.modulation`。

### 3.2 官方 block 与 lingbot-va 的对应关系

| 官方 (Wan2.2) | lingbot-va (WanVideoFinetune) |
|---------------|-------------------------------|
| `blocks.*.self_attn` (q,k,v,o,norm_q,norm_k) | `blocks.*.attn1` (to_q, to_k, to_v, to_out.0, norm_q, norm_k) |
| `blocks.*.cross_attn` 同上 | `blocks.*.attn2` 同上 |
| `blocks.*.modulation` (1,6,dim) | `blocks.*.scale_shift_table` (1,6,dim) |
| `blocks.*.ffn.0 / ffn.2` | `blocks.*.ffn.net.0 / ffn.net.2`（diffusers FeedForward） |
| `blocks.*.norm3` (elementwise_affine=True，有 weight/bias) | 你们 `norm3` 为 elementwise_affine=False，**无** weight/bias，故 ckpt 中 norm3 不加载 |
| `head.head` + `head.modulation` (1,2,dim) | `proj_out` + 顶层 `scale_shift_table` (1,2,dim) |

---

## 4. 推理流程（TI2V，`wan/textimage2video.py`）

1. **准备**：按 `vae_stride`、`patch_size` 算 latent 形状与 `seq_len`；T5 编码 prompt / negative prompt。
2. **采样器**：`FlowUniPCMultistepScheduler`，`num_train_timesteps=1000`，`set_timesteps(sampling_steps, device, shift=shift)`（如 50 步、shift=5.0）。
3. **每步**：
   - `noise_pred_cond = model(latent_model_input, t=timestep, context=context, seq_len=seq_len)[0]`
   - `noise_pred_uncond = model(..., context=context_null, ...)[0]`
   - `noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)`
   - `latents = scheduler.step(noise_pred, t, latents, ...)`
4. **解码**：`vae.decode(latents)` 得到视频帧。

T2V 与 I2V 仅输入构造不同（I2V 首帧用 VAE 编码得到，mask 后与噪声混合），模型调用方式相同。

---

## 5. 如何“对齐”训练（仅从推理侧可推断的）

- **目标**：Flow Matching，`num_train_timesteps=1000`，预测的应是 flow / velocity。
- **采样**：UniPC，50 步，shift=5.0，CFG=5.0。
- **训练代码**：官方未开源；README 提到 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 支持 Wan2.2 的 full training、LoRA 等，可作训练参考。
- **ckpt 与 config**：`Wan2.2-TI2V-5B` 的 `config.json` 与上面 TI2V-5B 配置一致（dim/num_heads/num_layers/ffn_dim 等），lingbot-va 从该 ckpt 加载时需做 key 映射（已在 `wan_va/modules/utils.py` 中实现）。

---

## 6. 小结

- **Wan2.2 仓库**：只有推理（`generate.py` + `wan/` 下 pipeline 与 `WanModel`），**没有训练脚本**。
- **怎么调模型**：通过 `wan/configs/wan_ti2v_5B.py` 等配置 + `WanModel.from_pretrained(ckpt_dir)` + 上述 forward 接口（x, t, context, seq_len）与采样循环即可复现官方推理行为。
- **怎么训**：需结合论文/博客或 DiffSynth-Studio 等第三方实现；从推理侧可知训练应为 Flow Matching、1000 timesteps、UniPC 与 CFG 等超参与官方一致即可对齐推理。
