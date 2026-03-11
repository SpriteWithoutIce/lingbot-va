# 从训练代码可视化 `Δv_t = v_t - v_{t-1}`（叠加到 `z_t/z_{t-1}`）

你现在只有训练代码与训练保存的 transformer，这个文档对应的脚本就是为这个场景写的：

- 脚本：`scripts/visualization/visualize_velocity_delta.py`
- 直接复用训练侧组件：`VA_CONFIGS`、`MultiLatentLeRobotDataset`、`FlowMatchScheduler`、`load_transformer`
- **模型加载**：从你训练保存的 `checkpoint_step_xxx/transformer` 加载 DiT
- **decoder 加载**：优先按 `root + subfolder=vae` 方式从 base Wan ckpt 加载；也兼容直接传 `vae` 目录

---

## 可视化逻辑（与训练目标对齐）

1. 从数据集取一个样本 latent `z`（`[C,F,H,W]`）。
2. 用训练同款 flow-matching 目标构造：
   - `noisy_latents = add_noise(z, noise, timestep)`
   - `target_v = noise - z`
3. 用训练 ckpt 的 transformer 前向得到 `pred_v`。
4. 在 **frame 维度** 计算：
   - `Δv_t(target) = target_v[:, t] - target_v[:, t-1]`
   - `Δv_t(pred) = pred_v[:, t] - pred_v[:, t-1]`
5. 将 `|Δv_t|` 聚合成 2D 权重矩阵，并映射回原视频帧：
   - `Δv_1` 覆盖到 `o1,o2,o3,o4`
   - `Δv_2` 覆盖到 `o5,o6,o7,o8`
   - 一般地 `Δv_k` 覆盖到第 `[(k-1)*4+1, k*4]` 这 4 帧

### 关于你遇到的 `48 vs 16` 报错

- Wan VAE 的 `z_dim` 通常是 16。
- 你的训练 latent 可能是 `C=48`（例如 3 路视角各 16 通道在 channel 维拼接）。
- 当前脚本已支持：当 `C = N * z_dim` 时，自动按通道分组（每组 16）分别 decode，最后把 N 路解码图按宽度拼接展示。
- `Δv` 热力图也会按同样分组生成并横向拼接，和解码图保持一一对应。

### 现在可视化背景是原图（不是 decode 图）

- 脚本会根据 `episode_index/start_frame/end_frame` 直接从 `videos/.../episode_xxx.mp4` 读取原帧。
- 然后按训练时的多视角拼接方式（如 robotwin_tshape）合成背景图，再叠加 `Δv` 权重矩阵。
- 这样你可以直接在原始 `o_t` 上看到最相关区域。

> 这里的 `t` 是视频 latent 的帧索引（不是 diffusion step）。

---

## 用法

```bash
python scripts/visualization/visualize_velocity_delta.py \
  --config_name robotwin_video_train \
  --train_ckpt /path/to/checkpoint_step_1600 \
  --base_ckpt_dir /home/jwhe/linyihan/Wan2.2-TI2V-5B \
  --sample_index 0 \
  --timestep_id 500 \
  --delta_source pred \
  --out_dir outputs/velocity_delta_viz_train
```

### 常用参数

- `--train_ckpt`: 既可传 `checkpoint_step_xxx`，也可直接传其下 `transformer/`
- `--base_ckpt_dir`: **推荐**传 Wan base ckpt 根目录（含 `vae/`）。也可直接传 `vae/` 目录。

> 如果你把 `--base_ckpt_dir` 指到 TI2V 根目录但未正确走 `vae` 子目录，会出现
> “The config attributes {'dim', ...} were passed to AutoencoderKLWan” 这类告警。
> 当前脚本已优先使用 `subfolder="vae"` 避免此问题。
- `--dataset_path`: 可覆盖 config 里的数据集路径
- `--latent_subdir`: 可覆盖 config 里的 latent 子目录（如 `latents_video_ft`）
- `--timestep_id`: 构造 noisy latent 时用的 diffusion timestep index（0~999）
- `--max_frame_pairs`: 最多输出多少组相邻帧 `(t-1,t)` 的图
- `--delta_source`: 用 `pred`（模型输出）或 `target`（训练目标）来计算并可视化 `Δv`

---

## 输出文件

输出为每个原始帧一张叠加图 + 一张权重图：

- `delta_<source>_latent_XXX_frame_YYYY.png`：权重矩阵覆盖在原图上的可视化
- `weight_<source>_latent_XXX_frame_YYYY.png`：对应权重矩阵本身
