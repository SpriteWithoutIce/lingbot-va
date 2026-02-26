# LIBERO 评估说明（lingbot-va）

本目录提供在 **LIBERO** 基准上评估 lingbot-va 策略的脚本，采用与 [robotwin](../robotwin) 相同的 **一机一客户端** 方式：先启动策略服务器，再在另一终端运行 LIBERO 环境客户端，客户端通过 WebSocket 向服务器请求动作。

评估流程与 reasoningvla 的 [libero_eval](https://github.com/.../reasoningVLA/evaluation/libero_eval) 一致。

---

## 环境准备

### 1. lingbot-va 环境（跑 server）

使用项目已有的 conda 环境（能跑 `wan_va` 训练/推理即可），例如：

```bash
conda activate lingbotva   # 或你的环境名
cd /home/jwhe/linyihan/lingbot-va
```

### 2. LIBERO 仿真环境（跑 client）

按 [LIBERO 官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO) 安装仿真与 benchmark：

```bash
# 建议单独一个 conda 环境
conda create -n libero python=3.8
conda activate libero
apt-get install -y libgl1 libosmesa6 libosmesa6-dev   # 如未安装

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch torchvision torchaudio   # 按需指定 cuda 版本
pip install -e .
cd ..
```

客户端脚本需在**能 import `libero`** 的环境下运行（即安装好 LIBERO 的环境）。

---

## 默认 checkpoint

- 默认策略权重目录：**`/home/jwhe/linyihan/lingbot-va-base`**
- 在 `wan_va/configs/va_libero_spatial_cfg.py` 中通过 `wan22_pretrained_model_name_or_path` 配置；启动 server 时可通过覆盖该配置使用其他 ckpt。

---

## 使用方法

### 1. 启动策略服务器（Terminal 1）

在 **lingbot-va 项目根目录**、使用 **lingbot-va 的 conda 环境**：

```bash
cd /home/jwhe/linyihan/lingbot-va
conda activate lingbotva

# 默认单卡 0，端口 29536
bash evaluation/libero_eval/launch_server.sh
```

可选环境变量：

- `CUDA_VISIBLE_DEVICES`：使用的 GPU（默认 `0`）
- `START_PORT`：WebSocket 端口（默认 `29536`）
- `SAVE_ROOT`：服务器输出目录（默认 `./evaluation/libero_eval/server_out`）

例如指定 GPU 和端口：

```bash
CUDA_VISIBLE_DEVICES=1 START_PORT=29537 bash evaluation/libero_eval/launch_server.sh
```

### 2. 运行评估客户端（Terminal 2）

在 **能 import libero** 的环境中（可同机、另一 conda 环境），从 **lingbot-va 项目根目录** 运行：

```bash
cd /home/jwhe/linyihan/lingbot-va
conda activate libero   # 或已安装 libero 的环境

# 默认：libero_spatial，端口 29536
bash evaluation/libero_eval/launch_client.sh
```

指定 task suite 和端口（与 server 一致）：

```bash
bash evaluation/libero_eval/launch_client.sh libero_object 29536
```

或直接调用 Python：

```bash
python -m evaluation.libero_eval.run_libero_eval \
    --host 127.0.0.1 \
    --port 29536 \
    --task_suite_name libero_spatial \
    --replan_steps 10
```

常用参数：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--host` | 策略服务器地址 | `127.0.0.1` |
| `--port` | 策略服务器端口（需与 server 一致） | `29536` |
| `--task_suite_name` | LIBERO suite | `libero_spatial` |
| `--replan_steps` | 每轮请求动作执行的步数 | `10` |
| `--num_trials_per_task` | 每个任务的 rollout 数 | `1` |
| `--video_out_path` | 录像保存目录 | `data/libero/videos` |

支持的 `task_suite_name`：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`、`libero_90`。

---

## 目录与脚本说明

- **`launch_server.sh`**：启动 lingbot-va 策略服务器（`wan_va_server`，config `libero_spatial`）。
- **`launch_client.sh`**：调用 `run_libero_eval`，默认连本机 29536，可传 `task_suite_name` 和 `port`。
- **`run_libero_eval.py`**：评估主逻辑：加载 LIBERO suite、重置/步进环境、图像预处理、通过 WebSocket 请求动作并执行。
- **`websocket_client_policy.py`**：WebSocket 客户端，与 `wan_va/utils/.../websocket_policy_server` 对应。
- **`image_tools.py`**：图像 resize/pad、uint8 转换。
- **`msgpack_numpy.py`**：msgpack 序列化 numpy，与 server 端一致。

---

## 使用训练得到的 checkpoint 做评估

训练时 `train.py` 会按 `save_interval` 在 `save_root/checkpoints/` 下保存：

- `checkpoint_step_{step}/transformer/diffusion_pytorch_model.safetensors`
- `checkpoint_step_{step}/transformer/config.json`

**只保存了 transformer**，VAE、tokenizer、text_encoder 仍用 base（`wan22_pretrained_model_name_or_path`）。

评估时用该 checkpoint 的方式：

1. **启动 server 时加 `--ckpt`**（推荐）  
   将「checkpoint 根目录」传给 server（即包含 `transformer/` 的那一层，例如 `.../checkpoints/checkpoint_step_800`），server 会从 `ckpt/transformer` 加载 transformer，其余组件仍从 config 里的 base 路径加载：

   ```bash
   # 示例：使用 train_out/libero_all 下 step 800 的 ckpt
   CKPT=./train_out/libero_all/checkpoints/checkpoint_step_800 \
     bash evaluation/libero_eval/launch_server.sh
   ```

   或直接调用：

   ```bash
   python -m torch.distributed.run --nproc_per_node 1 --master_port 29561 \
     wan_va/wan_va_server.py --config-name libero_spatial --port 29536 \
     --ckpt ./train_out/libero_all/checkpoints/checkpoint_step_800
   ```

2. **不改 launch 脚本时**  
   编辑 `wan_va/configs/va_libero_spatial_cfg.py` 中的 `wan22_pretrained_model_name_or_path`，指向一个「完整」模型目录（包含 vae、tokenizer、text_encoder、transformer）。若要用训练 ckpt，需先把 base 拷一份，再用该 checkpoint 的 `transformer/` 覆盖这份里的 `transformer/`，然后把 `wan22_pretrained_model_name_or_path` 指到这份合并后的目录。

---

## 简要流程回顾

1. **Terminal 1**：在 lingbot-va 根目录、lingbotva 环境下执行 `bash evaluation/libero_eval/launch_server.sh`，等待出现监听端口。
2. **Terminal 2**：在 libero 环境下执行 `bash evaluation/libero_eval/launch_client.sh [suite] [port]`（或等价的 `python -m evaluation.libero_eval.run_libero_eval ...`）。
3. 客户端会按 task 和 trial 跑完所选 suite，并在控制台打印成功率，录像写入 `video_out_path`。

若 server 未就绪，客户端会持续重连直到 server 启动。
