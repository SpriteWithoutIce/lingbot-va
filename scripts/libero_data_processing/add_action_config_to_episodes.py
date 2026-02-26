#!/usr/bin/env python3
"""
为 LeRobot 格式的 LIBERO 数据集在 meta/episodes.jsonl 中增加 action_config 字段，
以满足 LingBot-VA 后训练数据格式要求。

用法:
  python add_action_config_to_episodes.py --dataset_path /home/jwhe/linyihan/datasets/libero
  python add_action_config_to_episodes.py --dataset_path /path/to/libero --in_place
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Add action_config to episodes.jsonl for LingBot-VA")
    parser.add_argument("--dataset_path", type=str, required=True, help="Root path of LeRobot dataset (e.g. .../libero)")
    parser.add_argument("--in_place", action="store_true", help="Overwrite meta/episodes.jsonl in place; default: write to meta/episodes_with_action_config.jsonl")
    args = parser.parse_args()

    root = Path(args.dataset_path)
    meta_dir = root / "meta"
    episodes_file = meta_dir / "episodes.jsonl"

    if not episodes_file.exists():
        raise FileNotFoundError(f"Not found: {episodes_file}")

    out_file = meta_dir / "episodes.jsonl" if args.in_place else meta_dir / "episodes_with_action_config.jsonl"
    lines_out = []

    with open(episodes_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "action_config" in rec:
                lines_out.append(rec)
                continue
            length = rec.get("length", 0)
            tasks = rec.get("tasks", [])
            action_text = tasks[0] if tasks else "Manipulation task."
            action_config = [
                {
                    "start_frame": 0,
                    "end_frame": length,
                    "action_text": action_text,
                }
            ]
            rec["action_config"] = action_config
            lines_out.append(rec)

    with open(out_file, "w", encoding="utf-8") as f:
        for rec in lines_out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(lines_out)} episodes to {out_file}")
    if not args.in_place:
        print("Replace meta/episodes.jsonl with this file if you are satisfied, or use --in_place next time.")


if __name__ == "__main__":
    main()
