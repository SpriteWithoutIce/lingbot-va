#!/bin/bash
# 用法:
#   单任务:   ./launch_client.sh [save_root] <task_name>
#   按组测:   ./launch_client.sh [save_root] <group_id>   # group_id=0..6 跑该组内所有 task
#   全组测:   ./launch_client.sh [save_root] all         # 按组顺序跑全部 task
# 例: ./launch_client.sh ./results 0  或  ./launch_client.sh ./results all
#
# 说明：评测必须在本地跑 Sapien 并渲染相机图供策略使用，无法“完全不渲染”。
# 无头/无显示器时若报 "failed to find a rendering device" 可依次尝试：
#   1) 已优先加载 conda 的 Vulkan、VK_ICD_FILENAMES 回退到 /etc/vulkan
#   2) 使用虚拟显示：USE_XVFB=1 ./launch_client.sh ...（需安装 xorg-x11-server-Xvfb）
#   3) 确保存在 /usr/share/glvnd/egl_vendor.d/10_nvidia.json（无则从本机拷贝）

export NVIDIA_DRIVER_CAPABILITIES=all
# 优先加载 conda 环境里的 Vulkan/EGL，不用服务器上有问题的那套（需先 conda install -c conda-forge vulkan-tools libvulkan-loader）
CONDA_LIB=""
if [[ -n "$CONDA_PREFIX" ]] && [[ -d "$CONDA_PREFIX/lib" ]]; then
  CONDA_LIB="$CONDA_PREFIX/lib"
fi
export LD_LIBRARY_PATH="${CONDA_LIB:+$CONDA_LIB:}/usr/lib64:/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
elif [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi
export ROBOTWIN_SKIP_RENDER_TEST=1
export ROBOTWIN_HEADLESS=1
export SAPIEN_RENDER_SYSTEM=egl
# 无头/服务器下：USE_XVFB=1 时用虚拟显示（xvfb-run），否则 EGL 离屏
if [[ -n "$USE_XVFB" ]] && [[ "$USE_XVFB" != "0" ]]; then
  # 不置空 DISPLAY，由 xvfb-run 提供 :99 等；下面 run_one_task 会用 xvfb-run 包一层
  export PYOPENGL_PLATFORM=egl
else
  export DISPLAY=""
  export PYOPENGL_PLATFORM=egl
fi
export CUDA_VISIBLE_DEVICES=2

ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-/home/jwhe/linyihan/RoboTwin}
export ROBOTWIN_ROOT

# 连本机 server 时绕过代理，避免 HTTP 503（proxy rejected）
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

task_groups=(
  "stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe"
  "adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch"
  "shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket"
  "pick_diverse_bottles open_microwave beat_block_hammer press_stapler click_bell move_playingcard_away open_laptop move_can_pot"
  "stack_bowls_two place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox"
  "click_alarmclock blocks_ranking_size place_phone_stand place_can_basket place_object_scale place_a2b_left grab_roller place_dual_shoes"
  "place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb"
)

save_root=${1:-'./results'}
group_or_task=${2:-"adjust_bottle"}

# policy 在 RoboTwin 里：ROBOTWIN_ROOT/policy/ACT/deploy_policy.yml；policy_name=ACT 即用该目录
policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
seed=0
PORT=${PORT:-29056}
HOST=${HOST:-127.0.0.1}
test_num=${TEST_NUM:-100}

run_one_task() {
  local task_name=$1
  local base_cmd=(
    env PYTHONWARNINGS=ignore::UserWarning XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
    python -m evaluation.robotwin.eval_polict_client_openpi --config policy/$policy_name/deploy_policy.yml
  )
  local extra_args=(
    --overrides
    --task_name "${task_name}"
    --task_config "${task_config}"
    --train_config_name "${train_config_name}"
    --model_name "${model_name}"
    --ckpt_setting "${model_name}"
    --seed "${seed}"
    --policy_name "${policy_name}"
    --save_root "${save_root}"
    --host "${HOST}"
    --port "${PORT}"
    --video_guidance_scale 5
    --action_guidance_scale 1
    --test_num "${test_num}"
  )
  if [[ -n "$USE_XVFB" ]] && [[ "$USE_XVFB" != "0" ]] && command -v xvfb-run &>/dev/null; then
    xvfb-run -a "${base_cmd[@]}" "${extra_args[@]}"
  else
    "${base_cmd[@]}" "${extra_args[@]}"
  fi
}

# 按 group 跑：收集要跑的 task 列表
if [[ "$group_or_task" == "all" ]]; then
  tasks=()
  for g in "${task_groups[@]}"; do
    read -r -a arr <<< "$g"
    tasks+=("${arr[@]}")
  done
  echo "[launch_client] Running ALL tasks (${#tasks[@]} total), save_root=$save_root"
elif [[ "$group_or_task" =~ ^[0-9]+$ ]] && (( group_or_task >= 0 && group_or_task < ${#task_groups[@]} )); then
  read -r -a tasks <<< "${task_groups[$group_or_task]}"
  echo "[launch_client] Group $group_or_task (${#tasks[@]} tasks), save_root=$save_root"
else
  # 单任务
  tasks=("$group_or_task")
  echo "[launch_client] Single task: $group_or_task, save_root=$save_root"
fi

for i in "${!tasks[@]}"; do
  task_name="${tasks[$i]}"
  echo -e "\033[33m========== [$((i+1))/${#tasks[@]}] task_name=$task_name ==========\033[0m"
  run_one_task "$task_name"
done
echo -e "\033[32mDone. Total tasks: ${#tasks[@]}\033[0m"


