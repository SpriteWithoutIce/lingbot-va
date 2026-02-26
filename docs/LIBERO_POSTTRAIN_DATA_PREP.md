# LIBERO 数据集用于 LingBot-VA 后训练的预处理说明

你的数据集路径：`/home/jwhe/linyihan/datasets/libero`。

LingBot-VA 后训练要求数据为 **LeRobot 格式 + action_config + Wan2.2 抽好的 latent**。下面按「你当前有什么」「还缺什么」「怎么做」说明。

---

## 1. 当前数据集状态（与 LingBot-VA / reasoningVLA 对比）

| 项目 | 你的 libero | LingBot-VA 后训练要求 | reasoningVLA 用法 |
|------|-------------|------------------------|------------------|
| 格式 | LeRobot v2.1（meta + parquet） | 同左 | 同左，`dataset_path` + `calculate_global_stats` |
| `meta/episodes.jsonl` | 有 `episode_index`, `tasks`, `length` | 还需 **`action_config`** | 不要求 action_config |
| 视频 | `total_videos: 0`，只有 parquet | 需要 **videos/** 或能抽 latent 的输入 | 可有 latent（`latent_idx`/`latent_z`） |
| 相机 | `image`, `wrist_image` (info.json) | 需与 **obs_cam_keys** 一致 | 用 meta 里的 camera_keys |
| Latent | 无（或 parquet 里已有 latent_z） | 需要 **latents/** 下 .pth（Wan2.2 VAE） | 可选，用 latent 时用 `libero_latent` |

结论：  
- **必须做**：给 `episodes.jsonl` 加上 **action_config**。  
- **必须做**：得到 **Wan2.2 VAE 的 latent**（要么先有视频再抽 latent，要么有从 parquet 抽 latent 的流程）。  
- **配置上**：LingBot-VA 的 `obs_cam_keys`、action 维数等要和你数据集一致（LIBERO 是 7 维 action、2 个相机）。

---

## 2. 必须做的处理

### Step 1：为 `episodes.jsonl` 添加 `action_config`

每条 episode 需要带 `action_config`，例如：

```json
{
  "episode_index": 0,
  "tasks": ["put the white mug on the left plate ..."],
  "length": 214,
  "action_config": [
    {
      "start_frame": 0,
      "end_frame": 214,
      "action_text": "put the white mug on the left plate ..."
    }
  ]
}
```

- 单段任务：`start_frame=0`，`end_frame=length`，`action_text` 用 `tasks[0]` 即可。

已提供脚本（建议先不覆盖原文件，检查后再替换）：

```bash
cd /home/jwhe/linyihan/lingbot-va
python scripts/libero/add_action_config_to_episodes.py --dataset_path /home/jwhe/linyihan/datasets/libero
# 会生成 meta/episodes_with_action_config.jsonl
# 检查无误后，可替换原文件或下次加 --in_place
python scripts/libero/add_action_config_to_episodes.py --dataset_path /home/jwhe/linyihan/datasets/libero --in_place
```

---

### Step 2：准备 Wan2.2 VAE 所需的「视频」或等价输入

LingBot-VA 的 latent 来自 **Wan2.2 VAE**，官方说明是「对视频做编码」。你当前是 **无 videos、只有 parquet**，因此要么：

- **方案 A**：从 parquet 里把 `image`、`wrist_image` 按帧导出为 **mp4**，放到 LeRobot 约定的目录下，再用 Wan2.2 的脚本抽 latent；  
- **方案 B**：若 [Wan-Video](https://github.com/Wan-Video) 或 LingBot-VA 提供了「从图像序列/parquet 直接抽 latent」的脚本，则按该脚本要求组织输入并输出到下面的 `latents/` 结构。

LeRobot 常见视频路径（与 info.json 中 `video_path` 一致）：

- `videos/chunk-000/observation.images.image/episode_000000.mp4`
- `videos/chunk-000/observation.images.wrist_image/episode_000000.mp4`

（若你的数据集用的是别的 key，例如 `observation.images.cam_high`，需要和 latent 抽取脚本、以及后面 Step 3 的 `obs_cam_keys` 一致。）

---

### Step 3：用 Wan2.2 VAE 抽取 latent，并放到 `latents/` 下

按 README 要求，latent 目录需与 `videos/` 结构对应，例如：

```
your_dataset/
├── videos/
│   └── chunk-000/
│       ├── observation.images.image/
│       │   └── episode_000000.mp4, ...
│       └── observation.images.wrist_image/
│           └── episode_000000.mp4, ...
├── latents/
│   └── chunk-000/
│       ├── observation.images.image/
│       │   └── episode_000000_0_214.pth, ...
│       └── observation.images.wrist_image/
│           └── episode_000000_0_214.pth, ...
└── meta/
    └── episodes.jsonl   # 已含 action_config
```

每个 `.pth` 需包含 README 中列出的字段（如 `latent`, `latent_num_frames`, `text_emb`, `text`, `frame_ids`, `start_frame`, `end_frame`, `fps`, `ori_fps` 等）。  
命名规则：`episode_{index:06d}_{start_frame}_{end_frame}.pth`，与 `action_config` 里该段的 `start_frame`/`end_frame` 一致。

Wan2.2 的具体用法请参考 [Wan-Video 文档](https://github.com/Wan-Video)。

---

## 3. LingBot-VA 侧配置（LIBERO 与 Robotwin 的差异）

- **obs_cam_keys**：必须和数据集里「用于生成 latent 的 key」一致。你当前是 **2 个相机**：`image`、`wrist_image`。若 latent 目录为 `observation.images.image` 等，则配置里应写为相同的 key，例如：
  - `obs_cam_keys = ['observation.images.image', 'observation.images.wrist_image']`
  或你实际使用的 key。
- **action 维度**：LIBERO 是 **7 维**；当前 `lerobot_latent_dataset.py` 里 `_action_post_process` 是为 Robotwin **30 维双臂**写的。要跑 LIBERO 后训练，需要：
  - 要么为 LIBERO 单独写/改一份 config（`action_dim=7`、对应的 `used_action_channel_ids`、`norm_stat` 等），并在 dataset 中为 7 维 action 做对齐；  
  - 要么在现有代码里加 LIBERO 分支，只取 7 维并做相应归一化。

`dataset_path`、`empty_emb_path` 等指向你的 libero 目录即可；`norm_stat` 可用你已有的 `meta/stats.json` 里 actions 的 q01/q99（或与 reasoningVLA 一样先跑 `calculate_global_stats.py`）。

---

## 4. 和 reasoningVLA 的对应关系

- reasoningVLA 用同一份 LeRobot 数据时：`dataset_path` → `dataset2feature.yaml`，并跑 `calculate_global_stats.py`，不要求 `action_config`，可选 latent（如 `libero_latent`）。
- LingBot-VA 后训练多做了两件事：  
  (1) **action_config**（本脚本已做）；  
  (2) **Wan2.2 的 latent** 放在 `latents/` 下，并配好 **obs_cam_keys** 和 **action 维数/归一化**。

你当前数据集在「给 LingBot-VA 用」时还需要：**完成 Step 1（已提供脚本）、Step 2（视频或等价输入）、Step 3（抽 latent）、以及 LIBERO 专用 config/代码适配**。
