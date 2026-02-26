# 用 Wan2.2 VAE 生成 LingBot-VA 所需的 latent

根据 [LingBot-VA README](README.md) 的 **Step 3**，后训练需要先用 **Wan2.2 VAE** 对视频抽 latent，并把 `.pth` 放到数据集的 `latents/` 下。这里说明如何用 Wan2.2 仓库和 5B 权重生成 README 要求的那份东西。

## README 要求生成的「东西」

每个 **`.pth`** 是一个字典，需包含：

| 键 | 类型 | 说明 |
|----|------|------|
| `latent` | `Tensor [N, C]` (bfloat16) | 展平的 VAE latent，N = latent_num_frames × latent_height × latent_width |
| `latent_num_frames` | int | latent 时间帧数 |
| `latent_height` / `latent_width` | int | latent 空间高/宽 |
| `video_num_frames` | int | 源视频（或采样后）的帧数 |
| `video_height` / `video_width` | int | 源视频高/宽（像素） |
| `text_emb` | `Tensor [L, D]` (bfloat16) | 用 **Wan2.2 文本编码器**（T5）编码的 action 描述 |
| `text` | str | 原始 action 描述文本 |
| `frame_ids` | list[int] | 采样出的帧在 episode 中的下标 |
| `start_frame` / `end_frame` | int | 与 `episodes.jsonl` 里 `action_config` 一致 |
| `fps` | int | 采样用的目标 fps |
| `ori_fps` | int | 原始 episode 的 fps |

文件命名：`episode_{index:06d}_{start_frame}_{end_frame}.pth`，与 `action_config` 的起止帧对应。

目录结构要与 `videos/` 一致，例如：

```
<dataset_root>/
├── videos/
│   └── chunk-000/
│       ├── observation.images.image/
│       └── observation.images.wrist_image/
├── latents/
│   └── chunk-000/
│       ├── observation.images.image/
│       │   └── episode_000000_0_110.pth
│       └── observation.images.wrist_image/
│           └── episode_000000_0_110.pth
└── meta/
    └── episodes.jsonl   # 必须已含 action_config
```

---

## 环境与权重

- **Wan2.2 仓库**：例如 `/home/jwhe/linyihan/Wan2.2`
- **Wan2.2-TI2V-5B 权重目录**：例如 `/home/jwhe/linyihan/Wan2.2-TI2V-5B`  
  该目录下需包含：
  - `Wan2.2_VAE.pth`（VAE 权重）
  - `models_t5_umt5-xxl-enc-bf16.pth`（T5 编码器权重，用于 `text_emb`）
  - 文本编码器用的 tokenizer：若本地没有，脚本会按 `google/umt5-xxl` 从 HuggingFace 拉取

- **Python**：安装 Wan2.2 的依赖；另外需要能读视频的库之一：
  - `decord`（推荐）：`pip install decord`
  - 或使用 `torchvision` 读整段视频（无需额外安装）

---

## 使用本仓库提供的脚本

在 **lingbot-va** 仓库根目录下执行（或自行调整 `--wan22_root` / `--ckpt_dir`）：

```bash
cd /home/jwhe/linyihan/lingbot-va

# 单个数据集（例如 libero_spatial_dataset）
python scripts/libero/extract_latents_wan22.py \
  --wan22_root /home/jwhe/linyihan/Wan2.2 \
  --ckpt_dir /home/jwhe/linyihan/Wan2.2-TI2V-5B \
  --dataset_path /home/jwhe/linyihan/datasets/libero_lingbot/libero_spatial_dataset
```

- 脚本会：
  - 从 `meta/episodes.jsonl` 读取每个 episode 的 `action_config`（必须已有）
  - 从 `meta/info.json` 读取 `fps` 等
  - 按 `videos/chunk-xxx/<video_key>/episode_XXX.mp4` 读视频
  - 用 Wan2.2 VAE 编码视频 → 得到 `latent` 及 latent 的 T/H/W
  - 用 Wan2.2 的 T5 编码 `action_text` → 得到 `text_emb`
  - 按 README 要求写出每个相机、每个 segment 的 `.pth` 到 `latents/chunk-xxx/<video_key>/`

- 可选参数：
  - `--video_keys`：默认 `observation.images.image`、`observation.images.wrist_image`，与当前 LIBERO 数据集一致时可省略
  - `--target_fps`：若设，会按该 fps 对帧做采样再编码；不设则用 `[start_frame, end_frame)` 内全部帧
  - `--max_episodes N`：只处理前 N 个 episode，便于调试
  - `--device`：默认 `cuda`

调试时可先跑 2 个 episode 看是否报错、输出形状是否正确：

```bash
python scripts/libero/extract_latents_wan22.py \
  --wan22_root /home/jwhe/linyihan/Wan2.2 \
  --ckpt_dir /home/jwhe/linyihan/Wan2.2-TI2V-5B \
  --dataset_path /home/jwhe/linyihan/datasets/libero_lingbot/libero_spatial_dataset \
  --max_episodes 2
```

---

## 四个 LIBERO 数据集各跑一遍

对 `libero_lingbot` 下四个子数据集分别执行一次即可：

```bash
for name in libero_spatial_dataset libero_object_dataset libero_goal_dataset libero_10_dataset; do
  python scripts/libero/extract_latents_wan22.py \
    --wan22_root /home/jwhe/linyihan/Wan2.2 \
    --ckpt_dir /home/jwhe/linyihan/Wan2.2-TI2V-5B \
    --dataset_path /home/jwhe/linyihan/datasets/libero_lingbot/$name
done
```

每个数据集都会在各自根目录下生成 `latents/`，满足 README 的目录与字段要求。

---

## 小结

- **要生成的东西**：README 里 Step 3 规定的那张表 + 命名 `episode_{index}_{start_frame}_{end_frame}.pth` + 与 `videos/` 一致的 `latents/` 结构。
- **用什么生成**：Wan2.2 仓库里的 **Wan2.2 VAE** 编码视频得到 `latent`，用同仓库的 **T5** 编码 `action_text` 得到 `text_emb`；5B 权重目录假设为 `/home/jwhe/linyihan/Wan2.2-TI2V-5B`（仍在上传时可先跑 `--max_episodes 2` 做路径与格式检查）。
