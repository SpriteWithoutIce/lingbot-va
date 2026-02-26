# LIBERO-Spatial 后训练

## 数据集是否满足 README 要求

对 `datasets/libero_lingbot/libero_spatial_dataset`：

| README 要求 | 状态 |
|-------------|------|
| **Step 1** LeRobot 格式（data/, videos/, meta/） | ✅ 已有 |
| **Step 2** `meta/episodes.jsonl` 含 `action_config` | ✅ 已有 |
| **Step 3** `latents/` 下按 `episode_{index}_{start}_{end}.pth` 放置 Wan2.2 抽好的 latent | ✅ 需已跑完 `extract_latents_wan22.py` |

另外需要在数据集目录下提供 **`empty_emb.pt`**（T5 对空/负向 prompt 的 embedding，用于 cfg）。若尚未生成，见下节。

## empty_emb.pt

训练会读 `{dataset_path}/empty_emb.pt`。若不存在，可二选一：

1. **从已有 Robotwin 数据集拷一份**（同一 T5 即可）  
   ```bash
   cp /path/to/robotwin-clean-and-aug-lerobot/empty_emb.pt /home/jwhe/linyihan/datasets/libero_lingbot/libero_spatial_dataset/
   ```
2. **用 Wan2.2 T5 生成**  
   用 Wan2.2 仓库加载 T5，对空字符串或负向 prompt 编码一次，把得到的 tensor 存成 `empty_emb.pt`（与 lingbot-va 用的 T5 一致即可）。

## 训练前：base 的 attn_mode

README 要求：**训练时** base 的 `transformer/config.json` 里 `attn_mode` 必须为 `"flex"`。若当前是 `"torch"` 或 `"flashattn"`，请先改为 `"flex"` 再启动训练。

## 启动命令

在 **lingbot-va** 仓库根目录执行：

```bash
# 默认 8 卡，base=/home/jwhe/linyihan/lingbot-va-base，dataset=libero_spatial_dataset
NGPU=8 bash script/run_libero_spatial_posttrain.sh
```

覆盖路径或步数示例：

```bash
BASE_CKPT=/home/jwhe/linyihan/lingbot-va-base \
DATASET=/home/jwhe/linyihan/datasets/libero_lingbot/libero_spatial_dataset \
SAVE_ROOT=./train_out/libero_spatial_posttrain \
NGPU=4 bash script/run_libero_spatial_posttrain.sh
```

更多超参可在脚本末尾加 `wan_va.train` 的参数，例如：

```bash
... run_libero_spatial_posttrain.sh num_steps=10000 learning_rate=5e-6
```
