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
5. 将 `|Δv_t|` 聚合成 2D 热力图，映射并叠加到 Wan VAE 解码后的 `z_t/z_{t-1}` 图像上。

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

---

## 输出文件

每个相邻帧对会输出两张图：

- `target_delta_v_frame_XXX.png`：目标速度场的 `Δv_t`
- `pred_delta_v_frame_XXX.png`：模型预测速度场的 `Δv_t`

每张图是三联图：

- 左：`z_{t-1}` + `|Δv_t|` 叠加
- 中：`z_t` + `|Δv_t|` 叠加
- 右：`|Δv_t|` 热力图
