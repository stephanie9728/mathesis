#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_gdino.py  —  单帧 GroundingDINO 调试脚本

- 和 Vid-B 使用同一套 TASK_PROMPTS / TASK_KEYWORDS / phrase_to_label 逻辑
- 不写 json，不统计整段视频，只看一帧的检测结果 + 一张带框图

用法示例：

  # 显式指定 task
  python run_gdino.py \
    --task grasp_apple_uncertainty \
    --image camera_demo_apple/frames/00720.png

  # 不指定 task，则从图片上层目录名推：.../cut_fruit_tool_error/frames/xxxx.png
  python run_gdino.py \
    --image some_root/frames/00010.png
"""

import argparse
from pathlib import Path

import cv2
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate

# ============================================================
# 1. Task -> Prompt / 关键词 / 标签优先级（和 Vid-B 保持一致）
# ============================================================

TASK_PROMPTS = {
    # 错误工具切水果
    "cut_fruit_tool_error": (
        "person . hand . apple . red apple . banana . "
        "knife . fork . spoon . cutting board . chopping board . table . bowl . cup"
    ),
    # 倒麦片能力不足
    "pour_cereal_inability": (
        "person . hand . cereal box . cereal . milk carton . bottle . "
        "bowl . cup . mug . spoon . table"
    ),
    # 抓苹果：不确定位置
    "grasp_apple_uncertainty": (
        "hand . apple . red apple . bowl . white bowl . table"
    ),
}

TASK_KEYWORDS = {
    "cut_fruit_tool_error": [
        "person", "hand", "apple", "banana",
        "knife", "fork", "spoon",
        "cutting board", "chopping board",
        "bowl", "cup", "table",
    ],
    "pour_cereal_inability": [
        "person", "hand",
        "cereal box", "cereal",
        "milk carton", "bottle",
        "bowl", "cup", "mug", "spoon", "table",
    ],
    "grasp_apple_uncertainty": [
        "hand", "apple", "red apple", "bowl", "white bowl", "table",
    ],
}

LABEL_PRIORITY = {
    "cut_fruit_tool_error": [
        "knife", "fork", "spoon",
        "apple", "banana",
        "bowl", "cup",
        "cutting board", "chopping board",
    ],
    "pour_cereal_inability": [
        "cereal box", "cereal",
        "bowl", "cup", "mug",
        "spoon",
        "milk carton", "bottle",
    ],
    "grasp_apple_uncertainty": [
        "apple", "red apple",
        "banana", "fruit",
        "bowl", "white bowl",
    ],
}

TOPK_PER_KEYWORD = 3


# ============================================================
# 2. phrase -> 简洁 label（和 Vid-B 一致）
# ============================================================

def phrase_to_label(phrase: str, task: str) -> str:
    lower = phrase.lower()
    for kw in LABEL_PRIORITY[task]:
        if kw in lower:
            if "apple" in kw:
                return "apple"
            if "bowl" in kw:
                return "bowl"
            if "cutting board" in kw or "chopping board" in kw:
                return "cutting board"
            return kw
    return "object"


# ============================================================
# 3. 主逻辑：单帧调试
# ============================================================

def main():
    parser = argparse.ArgumentParser("Single-frame GroundingDINO debug (Vid-B style)")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="单帧图片路径 (png/jpg)。例如 camera_demo_apple/frames/00720.png",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="可选: cut_fruit_tool_error | pour_cereal_inability | grasp_apple_uncertainty; "
             "若不指定，则用 image 上上级目录名推断。",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="GroundingDINO/weights/groundingdino_swint_ogc.pth",
    )
    parser.add_argument("--box_thresh", type=float, default=0.35)
    parser.add_argument("--text_thresh", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出可视化路径（默认：和 image 同目录，后缀 _gdino_annotated.jpg）",
    )
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到图片: {image_path}")

    # ---------- 推断 task ----------
    task = args.task
    if task is None:
        # 假设结构类似：.../<video_root>/frames/00010.png
        # video_root 名字就是 task 名，比如 grasp_apple_uncertainty
        if image_path.parent.name == "frames":
            candidate = image_path.parent.parent.name
        else:
            candidate = image_path.parent.name

        if candidate in TASK_PROMPTS:
            task = candidate
            print(f"ℹ️ 未显式指定 --task，从目录名推断为: {task}")
        else:
            print(f"⚠️ 无法从目录名推断 task，使用 grasp_apple_uncertainty 兜底")
            task = "grasp_apple_uncertainty"

    if task not in TASK_PROMPTS:
        raise ValueError(f"未知 task: {task}，可选: {list(TASK_PROMPTS.keys())}")

    prompt = TASK_PROMPTS[task]
    keywords = TASK_KEYWORDS[task]

    print(f"\n✅ Task     : {task}")
    print(f"✅ Image    : {image_path}")
    print(f"✅ Prompt   : {prompt}\n")

    # ---------- 加载模型 ----------
    print("✅ Loading GroundingDINO...")
    model = load_model(args.config, args.weights)
    print("✅ GroundingDINO loaded\n")

    # ---------- 推理 ----------
    image_source, image = load_image(str(image_path))
    h, w = image_source.shape[:2]

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=prompt,
        box_threshold=args.box_thresh,
        text_threshold=args.text_thresh,
    )

    print(f"[GDINO] Total boxes: {boxes.shape[0]}")
    if boxes.shape[0] == 0:
        print("⚠️ 没有任何框通过阈值，退出。")
        return

    # ---------- 关键词过滤 + 每个关键词 Top-K ----------
    filtered = []
    for kw in keywords:
        bucket = []
        for b, l, p in zip(boxes, logits, phrases):
            if kw in p.lower():
                bucket.append((b, l, p))
        if not bucket:
            continue
        bucket = sorted(bucket, key=lambda x: float(x[1]), reverse=True)
        filtered.extend(bucket[:TOPK_PER_KEYWORD])

    if not filtered:
        print("⚠️ 关键词过滤后没有剩余框。")
        return

    boxes_keep = torch.stack([x[0] for x in filtered], dim=0)
    logits_keep = torch.stack([x[1] for x in filtered], dim=0)
    phrases_keep = [x[2] for x in filtered]

    # ---------- 打印结果（和后面 Vid-B 节点统计逻辑一致） ----------
    print("\n[GDINO] Kept detections:")
    label_counts = {}
    for b, s, ph in zip(boxes_keep, logits_keep, phrases_keep):
        label = phrase_to_label(ph, task)
        label_counts[label] = label_counts.get(label, 0) + 1
        print(f"  label={label:12s} score={float(s):.3f} phrase={ph}")

    print("\n[GDINO] Label counts:")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {lbl}: {cnt}")

    # ---------- 保存带框预览 ----------
    annotated = annotate(
        image_source=image_source,
        boxes=boxes_keep,
        logits=logits_keep,
        phrases=phrases_keep,
    )

    if args.output is None:
        out_path = image_path.with_name(f"{image_path.stem}_gdino_annotated.jpg")
    else:
        out_path = Path(args.output)

    cv2.imwrite(str(out_path), annotated[..., ::-1])
    print(f"\n✅ Annotated saved to: {out_path}")


if __name__ == "__main__":
    main()
