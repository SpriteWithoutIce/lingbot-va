#!/usr/bin/env python3
"""
用 Qwen3-VL 对 LIBERO episode 第一帧做 grounding：从 episodes.jsonl 取任务指令，
框出指令中相关物品。

未指定 --objects 时默认用 VLM 根据「图像+任务」自动列举物体（--extract-method vlm），
适配任意 prompt 句式，无需写正则；可选 --extract-method regex 使用简单正则备用。

用法一（指定数据集 + episode + 输出目录，自动截首帧并做 grounding）:
  python run_qwen_grounding_episode.py --data-root /path/to/libero_object_dataset --episode 2 --output-dir /path/to/out_dir
  首帧保存为 {output_dir}/episode_000002_frame0.png，grounding 保存为 {output_dir}/episode_000002_frame0_qwen_grounding.png

用法二（已有首帧图片）:
  python run_qwen_grounding_episode.py --image /path/to/frame.png --episodes-jsonl /path/to/meta/episodes.jsonl --episode 2 --output /path/to/result.png

Qwen3-VL 需 transformers>=4.57，建议用 qwenvl 环境；截帧需 PyAV，若用 --data-root 请先 pip install av。
"""
import argparse
import json
import re
import sys
import torch
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_PATH = "/home/jwhe/linyihan/VLM/Qwen3-VL-2B-Instruct"

# 视频首帧固定子路径（相对 data-root）
VIDEO_SUBPATH = "videos/chunk-000/observation.images.image"
META_EPISODES = "meta/episodes.jsonl"

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (128, 0, 255), (0, 128, 255),
]


def extract_first_frame(video_path: Path, output_path: Path) -> bool:
    """从视频提取第一帧保存为 PNG，成功返回 True。"""
    try:
        import av
    except ImportError:
        print("Error: 使用 --data-root 时需安装 PyAV: pip install av", file=sys.stderr)
        return False
    if not video_path.exists():
        print(f"Error: 视频不存在 {video_path}", file=sys.stderr)
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            Image.fromarray(img).save(str(output_path))
            print(f"已保存首帧: {output_path} shape={img.shape}")
            return True
    return False


def get_task_for_episode(episodes_jsonl: Path, episode_index: int) -> str:
    """从 episodes.jsonl 读取指定 episode 的 task 文本。"""
    with open(episodes_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("episode_index") == episode_index:
                tasks = rec.get("tasks", [])
                return tasks[0] if tasks else ""
    return ""


def task_to_objects_vlm(model, processor, image_path: Path, task: str, max_tokens: int = 128) -> list:
    """
    用 VLM 根据「图像 + 任务」自动列出要框选的物体名，不依赖 prompt 句式，适用于任意指令。
    """
    prompt = (
        f"Given this image and the following instruction: \"{task}\". "
        f"List the names of all physical objects that are mentioned in the instruction "
        f"and that you could locate in the image. Output only a comma-separated list of object names, nothing else."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    # 解析逗号分隔列表，去掉可能的前后缀说明
    output_text = re.sub(r"^(?:here are|objects?:\s*|list:\s*)\s*", "", output_text, flags=re.I)
    objects = [s.strip() for s in re.split(r"[,，]", output_text) if s.strip()]
    return objects if objects else []


def task_to_objects_regex(task: str) -> list:
    """
    从任务文本中用正则提取物体名（仅支持部分句式，作备用）。
    """
    objects = []
    m = re.search(r"pick up the ([^n]+?)(?:\s+next to|\s+and)", task, re.I)
    if m:
        objects.append(m.group(1).strip())
    m = re.search(r"next to the ([^\s]+(?:\s+[^\s]+)*?)(?:\s+and|\s*$)", task, re.I)
    if m:
        objects.append(m.group(1).strip())
    m = re.search(r"place it (?:on|in) the (.+?)(?:\.|$)", task, re.I)
    if m:
        objects.append(m.group(1).strip())
    seen = set()
    out = []
    for o in objects:
        if o and o not in seen:
            seen.add(o)
            out.append(o)
    return out if out else ["object"]


def parse_bbox_from_text(text: str, img_width: int, img_height: int):
    """从模型输出中解析 bounding box。"""
    results = []
    json_pattern = r'\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]'
    for m in re.finditer(json_pattern, text):
        x1, y1, x2, y2 = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
        if max(x1, y1, x2, y2) > 1:
            x1, y1 = x1 / 1000 * img_width, y1 / 1000 * img_height
            x2, y2 = x2 / 1000 * img_width, y2 / 1000 * img_height
        else:
            x1, y1 = x1 * img_width, y1 * img_height
            x2, y2 = x2 * img_width, y2 * img_height
        results.append((x1, y1, x2, y2))
    if results:
        return results
    pair_pattern = r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)\s*,?\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)'
    for m in re.finditer(pair_pattern, text):
        x1, y1, x2, y2 = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
        if max(x1, y1, x2, y2) > 1:
            x1, y1 = x1 / 1000 * img_width, y1 / 1000 * img_height
            x2, y2 = x2 / 1000 * img_width, y2 / 1000 * img_height
        else:
            x1, y1 = x1 * img_width, y1 * img_height
            x2, y2 = x2 * img_width, y2 * img_height
        results.append((x1, y1, x2, y2))
    return results


def draw_boxes(image: Image.Image, all_boxes: list, all_labels: list, line_width: int = 3):
    """在图片上绘制多组 bounding boxes，每组一个标签。"""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()
    idx = 0
    for label, boxes in zip(all_labels, all_boxes):
        color = COLORS[idx % len(COLORS)]
        idx += 1
        for box in boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            text_bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - th - 2), label, fill="white", font=font)
    return image


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL grounding: 按 episode 任务框出相关物品")
    parser.add_argument("--data-root", type=str, default=None,
                        help="数据集根目录，如 .../libero_object_dataset；与 --episode、--output-dir 一起用时自动截首帧并做 grounding")
    parser.add_argument("--episode", type=int, default=0,
                        help="episode 编号，与 --data-root 配合时也用于视频名 episode_XXXXXX.mp4")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="首帧与 grounding 结果保存目录（与 --data-root 一起用时必填）")
    parser.add_argument("--image", type=str, default=None,
                        help="第一帧图片路径（不填且用了 --data-root 则从视频截取）")
    parser.add_argument("--episodes-jsonl", type=str, default=None,
                        help="episodes.jsonl 路径；不填且用了 --data-root 则用 {data_root}/meta/episodes.jsonl")
    parser.add_argument("--objects", type=str, default=None,
                        help="逗号分隔的物品名；不填则用 --extract-method 从 task 得到")
    parser.add_argument("--extract-method", type=str, choices=["vlm", "regex"], default="vlm",
                        help="未指定 --objects 时如何从 task 得到物体列表: vlm=用模型看图+指令自动列举(适配任意 prompt)，regex=正则(仅部分句式)")
    parser.add_argument("--output", type=str, default=None,
                        help="grounding 结果图片路径（仅在不使用 --data-root 时生效）")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    if data_root is not None:
        # 模式：指定 data-root + output-dir，自动截首帧并做 grounding
        if not output_dir:
            raise ValueError("使用 --data-root 时请同时指定 --output-dir")
        data_root = data_root.resolve()
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = data_root / VIDEO_SUBPATH / f"episode_{args.episode:06d}.mp4"
        episodes_jsonl = data_root / META_EPISODES
        frame0_name = f"episode_{args.episode:06d}_frame0.png"
        image_path = output_dir / frame0_name
        if not image_path.exists():
            if not extract_first_frame(video_path, image_path):
                sys.exit(1)
        else:
            print(f"使用已有首帧: {image_path}")
        grounding_output_path = output_dir / f"episode_{args.episode:06d}_frame0_qwen_grounding.png"
        if not episodes_jsonl.exists():
            raise FileNotFoundError(f"episodes.jsonl 不存在: {episodes_jsonl}")
    else:
        # 模式：仅指定图片 + episodes-jsonl + episode
        if not args.image:
            raise ValueError("请指定 --image，或使用 --data-root + --episode + --output-dir")
        image_path = Path(args.image).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")
        episodes_jsonl = Path(args.episodes_jsonl or "").resolve()
        if not episodes_jsonl or not episodes_jsonl.exists():
            raise FileNotFoundError(f"请指定存在的 --episodes-jsonl: {episodes_jsonl}")
        grounding_output_path = Path(args.output or str(image_path.parent / f"{image_path.stem}_qwen_grounding.png"))
        grounding_output_path.parent.mkdir(parents=True, exist_ok=True)

    task = get_task_for_episode(Path(episodes_jsonl), args.episode)
    if not task:
        raise ValueError(f"episodes.jsonl 中未找到 episode_index={args.episode}")

    print(f"加载模型: {MODEL_PATH}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    if args.objects:
        objects = [s.strip() for s in args.objects.split(",") if s.strip()]
    else:
        if args.extract_method == "vlm":
            print("使用 VLM 从「图像+任务」自动列举要框选的物体…")
            objects = task_to_objects_vlm(model, processor, image_path, task, max_tokens=128)
            if not objects:
                print("VLM 未返回物体列表，回退到 regex 解析。")
                objects = task_to_objects_regex(task)
        else:
            objects = task_to_objects_regex(task)
    print(f"Task: {task}")
    print(f"Objects to detect: {objects}")

    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size
    all_boxes, all_labels = [], []

    for obj in objects:
        prompt = (
            f"Detect all '{obj}' in this image. "
            f"Output each bounding box as [x_min, y_min, x_max, y_max] "
            f"with coordinates in range [0, 1000]. Return all results as a JSON list."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path.resolve())},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(f"  [{obj}] 模型输出: {output_text[:200]}...")
        boxes = parse_bbox_from_text(output_text, img_width, img_height)
        all_boxes.append(boxes)
        all_labels.append(obj)
        if boxes:
            print(f"  -> 检测到 {len(boxes)} 个框")

    result_image = draw_boxes(image.copy(), all_boxes, all_labels)
    result_image.save(str(grounding_output_path))
    print(f"\n结果已保存: {grounding_output_path}")


if __name__ == "__main__":
    main()
