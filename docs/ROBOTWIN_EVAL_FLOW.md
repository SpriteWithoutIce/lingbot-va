# Robotwin 测试逻辑（Client + Server）

## Client 流程（eval_polict_client_openpi.py）

1. **reset**：`model.infer(dict(reset=True, prompt=prompt))`
2. 每轮循环：
   - **拿 action**：`model.infer(dict(obs=first_obs, prompt=..., ...))`
     - `first_obs` = **单条观测**（一个 dict：3 个 camera + state）
     - 即 **请求 action 时只送 1 帧**
   - 用返回的 `action` 在环境里逐步执行
   - 执行过程中每 `action_per_frame` 步把当前 obs 放进 `key_frame_list`
   - **更新 cache**：`model.infer(dict(obs=key_frame_list, compute_kv_cache=True, ...))`
     - `key_frame_list` = **多帧**（本轮执行中采样的多个 key frame）
     - 只用来更新 transformer 的 KV cache，**不返回 action**

结论：**要 action 的那次一定是单帧推理**；多帧只用于 compute_kv_cache。

## Server 流程（wan_va_server.py）

- **普通 infer**（要 action）：
  - `infer(obs)` → `_infer(obs)` → `_encode_obs(obs)`，此时 `obs['obs']` = `[first_obs]`，即 **1 帧**
  - 编码得到 `init_latent`，再以它为条件生成 `frame_chunk_size` 帧的 action
  - 语义上就是 **单帧推理**：用当前 1 帧观测做条件，生成 action

- **compute_kv_cache**：
  - `obs['obs']` = `key_frame_list`（多帧），编码后只用于更新 transformer KV cache，不生成 action

## 单帧与「3 帧」的关系

- **逻辑上**：推理始终是单帧（client 只送 1 帧来要 action）。
- **实现上**：VAE encoder 里 `time_conv` 的 kernel 在时间维为 3，要求输入时间维 ≥ 3；单帧时 T=1 会报错。
- 因此可在 **server 内部** 在送入 VAE 前把 1 帧在时间维 repeat 成 3（仅满足 VAE 约束），**不改变「单帧推理」语义**：条件仍是当前 1 帧，repeat 只是编码器兼容处理。
