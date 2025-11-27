#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vid-B (Clean Version)
- Task-specific GroundingDINO detection
- Input: video_root/frames/*.png
- Output:
    - yolo_nodes.json
    - yolo_scene_graph.json
    - relations_timeline.json
    - pred_annotated_cliptrk.jpg
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate

# ============================================================
# ✅ 1. Task -> Prompt & Keywords（与你 run_gdino.py 对齐）
# ============================================================

TASK_PROMPTS = {
    "cut_fruit_tool_error": (
        "person . hand . apple . red apple . banana . "
        "knife . fork . spoon . cutting board . chopping board . table . bowl . cup"
    ),
    "pour_cereal_inability": (
        "person . hand . cereal box . cereal . milk carton . bottle . "
        "bowl . cup . mug . spoon . table"
    ),
    "grasp_apple_uncertainty": (
        "hand . apple . red apple . bowl . white bowl . table"
    ),
}

TASK_KEYWORDS = {
    "cut_fruit_tool_error": ["person", "hand", "apple", "banana", "knife", "fork", "spoon", "board", "bowl", "cup", "table"],
    "pour_cereal_inability": ["person", "hand", "cereal", "milk", "bowl", "cup", "mug", "spoon", "table"],
    "grasp_apple_uncertainty": ["hand", "apple", "bowl", "table"],
}

TOPK_PER_CLASS = 3


# ============================================================
# ✅ 2. 工具函数：0-1 box -> 像素
# ============================================================

def boxes_to_pixel(boxes, h, w):
    boxes = boxes.clone()
    boxes[:, 0] *= w
    boxes[:, 2] *= w
    boxes[:, 1] *= h
    boxes[:, 3] *= h
    return boxes.cpu().numpy().astype(int)


# ============================================================
# ✅ 3. 主逻辑
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_root", type=str)
    args = parser.parse_args()

    video_root = Path(args.video_root).resolve()
    frame_dir = video_root / "frames"

    if not frame_dir.exists():
        raise FileNotFoundError(f"找不到 frames 目录: {frame_dir}")

    task_name = video_root.name
    if task_name not in TASK_PROMPTS:
        print(f"⚠️ 未注册 task: {task_name}，使用 grasp_apple_uncertainty 兜底")
        task_name = "grasp_apple_uncertainty"

    prompt = TASK_PROMPTS[task_name]
    keywords = TASK_KEYWORDS[task_name]

    print(f"\n✅ Vid-B Task : {task_name}")
    print(f"✅ Prompt     : {prompt}")

    # -----------------------------
    # 加载模型
    # -----------------------------
    print("✅ Loading GroundingDINO...")
    model = load_model(
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "GroundingDINO/weights/groundingdino_swint_ogc.pth",
    )
    print("✅ GroundingDINO loaded\n")

    # -----------------------------
    # 逐帧处理
    # -----------------------------
    nodes_all = {}
    relations_timeline = []

    frame_paths = sorted(frame_dir.glob("*.png")) + sorted(frame_dir.glob("*.jpg"))
    if len(frame_paths) == 0:
        raise RuntimeError("❌ frames 目录里没有图片")

    for frame_id, frame_path in enumerate(frame_paths):
        image_source, image = load_image(str(frame_path))
        h, w = image_source.shape[:2]

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=0.35,
            text_threshold=0.25,
        )

        if boxes.shape[0] == 0:
            continue

        # -----------------------------
        # Keyword + TopK 过滤
        # -----------------------------
        filtered = []
        for kw in keywords:
            candidates = []
            for b, l, p in zip(boxes, logits, phrases):
                if kw in p.lower():
                    candidates.append((b, l, p))
            candidates = sorted(candidates, key=lambda x: float(x[1]), reverse=True)
            filtered.extend(candidates[:TOPK_PER_CLASS])

        if len(filtered) == 0:
            continue

        boxes_keep = torch.stack([x[0] for x in filtered])
        logits_keep = torch.stack([x[1] for x in filtered])
        phrases_keep = [x[2] for x in filtered]

        boxes_px = boxes_to_pixel(boxes_keep, h, w)

        frame_nodes = []
        for i, (box, label) in enumerate(zip(boxes_px, phrases_keep)):
            node_id = f"{frame_id:06d}_{i:02d}"
            node = {
                "id": node_id,
                "frame": frame_id,
                "label": label,
                "bbox": box.tolist(),
            }
            frame_nodes.append(node)

        nodes_all[str(frame_id)] = frame_nodes

        # 简单关系（占位用）
        relations_timeline.append({
            "frame": frame_id,
            "relations": []
        })

        # -----------------------------
        # 首帧保存 annotated 预览
        # -----------------------------
        if frame_id == 0:
            annotated = annotate(
                image_source=image_source,
                boxes=boxes_keep,
                logits=logits_keep,
                phrases=phrases_keep,
            )
            out_img = video_root / "pred_annotated_cliptrk.jpg"
            cv2.imwrite(str(out_img), annotated[..., ::-1])
            print(f"✅ Annotated preview saved: {out_img}")

    # -----------------------------
    # 输出标准文件（对齐原 Vid-B）
    # -----------------------------
    nodes_path = video_root / "yolo_nodes.json"
    scene_path = video_root / "yolo_scene_graph.json"
    rel_path = video_root / "relations_timeline.json"

    with open(nodes_path, "w") as f:
        json.dump(nodes_all, f, indent=2)

    with open(scene_path, "w") as f:
        json.dump({"nodes": nodes_all, "edges": []}, f, indent=2)

    with open(rel_path, "w") as f:
        json.dump(relations_timeline, f, indent=2)

    print("\n✅ Vid-B Finished")
    print("nodes saved       :", nodes_path)
    print("scene graph saved :", scene_path)
    print("relations timeline:", rel_path)


if __name__ == "__main__":
    main()
