#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug GroundingDINO on a single frame with task-specific prompts.

Usage example:
  python run_gdino.py \
    --task grasp_apple_uncertainty \
    --image camera_demo_apple/frames/00720.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate

# -----------------------------
# 1. 每个 task 对应的 prompt
# -----------------------------
TASK_PROMPTS = {
    "cut_fruit_tool_error": (
        "person . hand . apple . red apple . banana . knife . fork . spoon "
        ". cutting board . chopping board . cup . bowl"
    ),
    "pour_cereal_inability": (
        "person . hand . cereal box . cereal . milk carton . bottle "
        ". bowl . cup . mug . spoon "
    ),
    "grasp_apple_uncertainty": (
        "apple . red apple . bowl . white bowl"
    ),
}

# 用来做关键词过滤的子串（只要 phrase 里含任意一个就保留）
TASK_KEYWORDS = {
    "cut_fruit_tool_error": [
        "person", "hand", "apple", "banana",
        "knife", "fork", "spoon", "board", "bowl", "cup", "table"
    ],
    "pour_cereal_inability": [
        "person", "hand", "cereal", "milk",
        "bowl", "cup", "mug", "spoon", "table"
    ],
    "grasp_apple_uncertainty": [
        "person", "hand", "apple", "bowl", "cup", "table"
    ],
}


# -----------------------------
# 2. 简单版 crop_images（自包含）
# -----------------------------
def crop_images(image_source, boxes, phrases, output_dir, max_crops=50):
    """
    image_source: numpy array (H,W,3), RGB
    boxes: torch.Tensor[N,4] or np.ndarray, normalized 0~1, xyxy
    phrases: list[str]，长度 N
    """
    h, w = image_source.shape[:2]

    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.asarray(boxes)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crops = []
    num = min(len(boxes_np), max_crops)

    for i in range(num):
        x1, y1, x2, y2 = boxes_np[i]

        # 归一化 -> 像素坐标，并裁到图像内
        x1 = int(max(0, min(w - 1, x1 * w)))
        x2 = int(max(0, min(w - 1, x2 * w)))
        y1 = int(max(0, min(h - 1, y1 * h)))
        y2 = int(max(0, min(h - 1, y2 * h)))

        if x2 < x1 or y2 < y1:
            continue

        crop = image_source[y1:y2, x1:x2, :]
        crops.append(crop)

        out_path = output_dir / f"crop_{i:03d}.jpg"
        # image_source 是 RGB，cv2 需要 BGR
        cv2.imwrite(str(out_path), crop[..., ::-1])
    
    print("[DEBUG] first 5 boxes (normalized):", boxes_np[:5])

    return crops


# -----------------------------
# 3. 主逻辑
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task id: cut_fruit_tool_error | pour_cereal_inability | grasp_apple_uncertainty",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Relative or absolute path to a single RGB frame (png/jpg).",
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
    parser.add_argument("--box_thresh", type=float, default=0.5)
    parser.add_argument("--text_thresh", type=float, default=0.5)
    args = parser.parse_args()

    task = args.task
    image_path = Path(args.image).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片: {image_path}")

    if task not in TASK_PROMPTS:
        raise ValueError(f"未知 task: {task}. 可选: {list(TASK_PROMPTS.keys())}")

    text_prompt = TASK_PROMPTS[task]
    keywords = TASK_KEYWORDS[task]

    print(f"\n[GDINO] Task  : {task}")
    print(f"[GDINO] Image : {image_path}")
    print(f"[GDINO] Prompt: {text_prompt}\n")

    # -----------------------------
    # 3.1 加载模型
    # -----------------------------
    print("✅ Loading GroundingDINO...")
    model = load_model(args.config, args.weights)
    print("✅ GroundingDINO loaded")

    # -----------------------------
    # 3.2 加载图像 & 推理
    # -----------------------------
    image_source, image = load_image(str(image_path))  # image_source: RGB np, image: transformed

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=args.box_thresh,
        text_threshold=args.text_thresh,
    )

    # boxes: [N, 4] torch.Tensor, 0–1, xyxy
    print(f"[GDINO] Boxes shape : {boxes.shape}")
    print(f"[GDINO] #phrases    : {len(phrases)}")

    if boxes.shape[0] == 0:
        print("⚠️ 没有任何框通过 score 阈值，直接退出。")
        return

    # -----------------------------
    # 3.3 关键词过滤
    # -----------------------------
    keep_indices = []
    keyword_counts = {k: 0 for k in keywords}

    for i, ph in enumerate(phrases):
        lower = ph.lower()
        hit_any = False
        for kw in keywords:
            if kw in lower:
                keyword_counts[kw] += 1
                hit_any = True
        if hit_any:
            keep_indices.append(i)

    if not keep_indices:
        print("[GDINO] 关键词过滤后没有剩余框。")
        return

    keep_indices_t = torch.as_tensor(keep_indices, dtype=torch.long, device=boxes.device)
    boxes_keep = boxes[keep_indices_t]      # Tensor [M,4]
    logits_keep = logits[keep_indices_t]    # Tensor [M]
    phrases_keep = [phrases[i] for i in keep_indices]

    # -----------------------------
    # 3.4 每个关键词保留 Top-K 框（可调）
    # -----------------------------
    TOPK_PER_CLASS = 3

    new_boxes = []
    new_logits = []
    new_phrases = []

    for kw in keywords:
        matched = []
        for b, l, p in zip(boxes_keep, logits_keep, phrases_keep):
            if kw in p.lower():
                matched.append((b, l, p))

        # 按置信度从高到低排序
        matched = sorted(matched, key=lambda x: float(x[1]), reverse=True)
        matched = matched[:TOPK_PER_CLASS]

        for b, l, p in matched:
            new_boxes.append(b)
            new_logits.append(l)
            new_phrases.append(p)

    if len(new_boxes) > 0:
        boxes_keep = torch.stack(new_boxes, dim=0)
        logits_keep = torch.stack(new_logits, dim=0)
        phrases_keep = new_phrases

    print(f"[GDINO] Kept {boxes_keep.shape[0]} boxes after keyword + Top-K filter.")
    print(f"[GDINO] Keyword counts (before Top-K): {keyword_counts}")

    # -----------------------------
    # 3.5 保存 crop（debug）
    # -----------------------------
    out_dir = Path("gdino_debug") / f"{image_path.stem}_{task}"
    out_dir.mkdir(parents=True, exist_ok=True)

    crops = crop_images(
        image_source=image_source,
        boxes=boxes_keep,
        phrases=phrases_keep,
        output_dir=str(out_dir),
    )
    print(f"✅ Crops saved to: {out_dir} (共 {len(crops)} 张，最多 50 张)")

    # -----------------------------
    # 3.6 annotate 画框
    # -----------------------------
    if boxes_keep.shape[0] > 0:
        annotated = annotate(
            image_source=image_source,
            boxes=boxes_keep,       # 仍然是 0–1 Tensor
            logits=logits_keep,
            phrases=phrases_keep,
        )
        out_annot = out_dir.parent / f"{image_path.stem}_{task}_annotated.jpg"
        # annotate 输出是 RGB，cv2 需要 BGR
        cv2.imwrite(str(out_annot), annotated[..., ::-1])
        print(f"✅ Annotated image: {out_annot}")
    else:
        print("⚠️ boxes_keep 为空，跳过 annotate。")


if __name__ == "__main__":
    main()
