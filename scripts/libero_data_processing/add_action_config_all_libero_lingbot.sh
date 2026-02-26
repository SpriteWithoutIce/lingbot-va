#!/usr/bin/bash
# 为 libero_lingbot 下四个数据集批量添加 action_config 并写回 meta/episodes.jsonl
# 用法: 在 lingbot-va 根目录执行:  bash scripts/libero/add_action_config_all_libero_lingbot.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARENT="${LIBERO_LINGBOT_ROOT:-/home/jwhe/linyihan/datasets/libero_lingbot}"

for name in libero_spatial_dataset libero_object_dataset libero_goal_dataset libero_10_dataset; do
  dir="$PARENT/$name"
  if [ ! -d "$dir" ]; then
    echo "Skip (not found): $dir"
    continue
  fi
  if [ ! -f "$dir/meta/episodes.jsonl" ]; then
    echo "Skip (no episodes.jsonl): $dir"
    continue
  fi
  echo "Processing: $dir"
  python "$REPO_ROOT/scripts/libero/add_action_config_to_episodes.py" --dataset_path "$dir" --in_place
done
echo "Done."
