# Semantic Probe README (LIBERO + WAN-VA)

这份文档从 0 开始说明当前仓库里的 **semantic probe** 到底在算什么、热力图怎么来的，以及每一步的**维度变化**。

> 代码对应入口：
> - 客户端：`evaluation/libero_eval/run_libero_eval.py`
> - 服务端：`wan_va/wan_va_server.py`
> - 模型注意力：`wan_va/modules/model.py`

---

## 1. 目标：我们在可视化什么？

我们要看的不是像素级显著性（如 Grad-CAM），而是：

- 在扩散推理的每个 denoise step，
- Transformer 的 **cross-attention（视觉 query -> 文本 key）**
- 对当前观测帧空间 token 的关注强度分布。

最后把这个空间分布映射回 2D 网格并叠加到当前 RGB 图像，形成热力图 overlay。

---

## 2. 端到端流程概览

1. `run_libero_eval.py` 在推理时发送：
   - `obs`（当前最新观测）
   - `prompt`
   - `semantic_probe=True`（按配置触发）
2. `wan_va_server.py::_infer` 在视频去噪循环中，额外请求 transformer 返回每层 cross-attn 概率。
3. 服务端将 cross-attn 聚合成空间 map（每层一张），并按 denoise step 收集。
4. 客户端收到后只取：
   - 最后一个 denoise step
   - 最后一层 map
   - agent-view 半区
   画成 overlay PNG。

---

## 3. 输入与 latent 的关键维度

下面以常见 LIBERO 配置（`height=256, width=256, patch=[1,2,2], frame_chunk_size=2`）为例：

### 3.1 视觉 latent

服务端 `_encode_obs` 会把多视角图像编码成视频 latent，然后在高度方向拼接视角（libero: wrist + agent）。

记：
- `F = frame_chunk_size`
- `H_lat = latent_height`
- `W_lat = latent_width`
- `C_lat = 48`

推理中的视频 noisy latent 形状：

- `latents`: `[1, 48, F, H_lat, W_lat]`

其中（libero）：
- `H_lat = (height / 16) * num_views`
- `W_lat = width / 16`

如果 `height=width=256, num_views=2`，则：
- `H_lat=32, W_lat=16`。

### 3.2 patch token 化

`patch_size = [1,2,2]`，所以每帧 token 网格：

- `H_tok = H_lat / 2`
- `W_tok = W_lat / 2`

query token 总数：

- `Q = F * H_tok * W_tok`

示例中：
- `H_tok=16, W_tok=8, F=2` → `Q=256`。

### 3.3 文本 token

文本 embedding（T5 输出后再过 text embedder）长度为 `T`（有效 token 数 <= 512）。

- 文本 hidden：`[B, T, D]`
- 对应 padding mask：`[B, T]`（True 表示有效 token）

---

## 4. cross-attention 是怎么提取的（精确到张量维度）

在 `WanAttention.forward` 里，cross-attn 概率用：

- `query`: `[B, Q, H, Dh]`
- `key`: `[B, T, H, Dh]`

转置后计算：

- `scores = (q @ k^T) * scale`
- `scores` 形状：`[B, H, Q, T]`

再应用 `key_padding_mask` 把无效文本 token 置极小值后 softmax：

- `attn_probs = softmax(scores, dim=-1)`
- `attn_probs` 形状仍是：`[B, H, Q, T]`

这就是每层返回的 cross-attn 图（每个 head、每个 query token，对所有文本 token 的分布）。

---

## 5. 从 cross-attention 到“空间热力图”

服务端 `_build_attention_probe_data` 做了以下聚合：

### Step A: 过滤 padding token

- `attn = attn[:, :, valid_text]`
- 形状从 `[H, Q, T]` 变为 `[H, Q, T_valid]`

### Step B: 沿文本 token 聚合

当前实现选的是：

- `max over text tokens`

即每个 head、每个 query 只保留对任一文本 token 的最大注意力：

- `attn.max(dim=-1).values` → `[H, Q]`

### Step C: 沿 head 聚合

- `mean over heads`

- `query_score = ...mean(dim=0)` → `[Q]`

### Step D: 还原时空网格并取指定帧

- reshape 为 `[F, H_tok, W_tok]`
- 取 `frame_id`（`semantic_probe_frame_id`）

得到单层 2D map：

- `query_map`: `[H_tok, W_tok]`

### Step E: 每层重复

对所有 transformer block 重复，得到：

- `maps_per_step`: `[L, H_tok, W_tok]`（`L`=层数）

再按 denoise step 追加，最终返回给客户端：

- `maps`: `[S, L, H_tok, W_tok]`（`S`=denoise steps）

---

## 6. 为什么客户端要切一半高度（agent view）

libero 下 latent 高度方向拼了两路视角（wrist + agent）。

因此客户端把 `maps[..., H_tok//2:, :]` 取出来作为 `maps_agent`，只可视化 agent-view 对应半区，避免把 wrist 区域混在一起。

切分后（示例）：

- 原 `H_tok=16` → agent 半区 `8`
- map 变成 `[S, L, 8, W_tok]`

---

## 7. 热力图 overlay 如何生成

客户端 `_save_semantic_overlay`：

1. 对 2D map 做 min-max 归一化到 `[0,1]`
2. 用 `matplotlib.cm.jet` 上色成 RGB
3. resize 到原图分辨率
4. 与原图按 `0.55 * image + 0.45 * heatmap` 混合
5. 保存 PNG

当前策略只保存：

- 最后 denoise step（`sid = S-1`）
- 最后一层（`layer = -1`）

文件命名：

- `taskXX_epYY_envstepZZZZ_denoisestepSSS_layer_last_overlay.png`

例如扩散步数 20 时，`SSS` 通常是 `019`。

---

## 8. 同时返回的统计量含义

服务端还保留了两类层级统计（对每层 map）：

1. `std`：map 标准差。越大通常表示空间响应越集中/不均匀。
2. `Moran's I`：8 邻域空间自相关。越高通常表示空间结构越连贯。

> 注意：当前客户端按需求不再保存 npz，但服务端 payload 中仍会携带这些字段，后续需要可重新落盘。

---

## 9. 常见误解与解释

1. **这不是原始像素 attention**
   - attention 是在 latent token 网格上，最后 resize 到图像。
2. **颜色是相对值，不是绝对概率**
   - 每张图单独 min-max，跨图对比需谨慎。
3. **最后一层不一定最好**
   - 只是当前导出规则；可改为导出多层或层平均。

---

## 10. 你如果要改策略，最常改这三处

1. 文本聚合方式：`max over text` 改成 `mean over text` 或指定关键词 token。
2. head 聚合方式：`mean over heads` 改成 `max over heads`。
3. 导出层/步：目前是“最后步 + 最后层”，可改为多步多层。

---

## 11. 最小复现参数建议

- `semantic_probe=True`
- `semantic_probe_chunk_stride=1`
- `semantic_probe_frame_id=0`（或你关心的帧）
- latent denoise steps = 20

这样每次重规划都会输出 1 张最终 attention overlay，便于时序对比。
