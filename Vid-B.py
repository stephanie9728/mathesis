#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vid-B (GroundingDINO clean version)

- 输入：VIDEO_ROOT/frames/*.png 或 *.jpg
- 输出：
    VIDEO_ROOT/yolo_nodes.json
    VIDEO_ROOT/yolo_scene_graph.json
    VIDEO_ROOT/relations_timeline.json
    VIDEO_ROOT/pred_annotated_cliptrk.jpg

- 只负责：用 GroundingDINO 做检测 + 统计每类物体的出现次数
- 不做跟踪、不做关系推断（edges 留空），
  所有高层解释交给 LLM。
"""

import argparse
import json
from pathlib import Path

import cv2
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate

# ============================================================
# 1. Task -> Prompt / 关键词 / 标签优先级
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

# 用来在 phrase 里做匹配的关键词（过滤 + 统计）
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

VIDEO_ROOT_TO_TASK = {
    "camera_demo_fruit":  "cut_fruit_tool_error",
    "camera_demo_cereal": "pour_cereal_inability",
    "camera_demo_apple":  "grasp_apple_uncertainty",
}

# 把 phrase 映射成“简洁标签”的优先级
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

TOPK_PER_KEYWORD = 3       # 每个关键词最多保留多少框
MAX_FRAMES = 300           # 最多处理多少帧（防止太慢）


# ============================================================
# 2. phrase -> 简洁 label
# ============================================================

def phrase_to_label(phrase: str, task: str) -> str:
    """
    把 GroundingDINO 的 phrase（长串）简化成一个 label（apple / bowl / knife ...）
    按 LABEL_PRIORITY 顺序匹配，匹配到第一个就返回。
    """
    lower = phrase.lower()
    for kw in LABEL_PRIORITY[task]:
        if kw in lower:
            # 简化一下：比如 "red apple" -> "apple"，"white bowl" -> "bowl"
            if "apple" in kw:
                return "apple"
            if "bowl" in kw:
                return "bowl"
            if "cutting board" in kw or "chopping board" in kw:
                return "cutting board"
            return kw
    return "object"  # 兜底，不太会用到


# ============================================================
# 3. 主逻辑
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_root", type=str)
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
    args = parser.parse_args()

    video_root = Path(args.video_root).resolve()
    frame_dir = video_root / "frames"

    if not frame_dir.exists():
        raise FileNotFoundError(f"找不到 frames 目录: {frame_dir}")

    task_name = VIDEO_ROOT_TO_TASK.get(video_root.name)

    if task_name is None:
        print(f"⚠️ 未识别 video_root 名: {video_root.name}，使用 grasp_apple_uncertainty 兜底")
        task_name = "grasp_apple_uncertainty"

    prompt = TASK_PROMPTS[task_name]
    keywords = TASK_KEYWORDS[task_name]


    print(f"\n✅ Vid-B Task : {task_name}")
    print(f"✅ Video Root : {video_root}")
    print(f"✅ Prompt     : {prompt}\n")

    # -----------------------------
    # 3.1 加载 GroundingDINO
    # -----------------------------
    print("✅ Loading GroundingDINO...")
    model = load_model(args.config, args.weights)
    print("✅ GroundingDINO loaded\n")

    # -----------------------------
    # 3.2 逐帧推理
    # -----------------------------
    frame_paths = sorted(frame_dir.glob("*.png")) + sorted(frame_dir.glob("*.jpg"))
    frame_paths = frame_paths[:MAX_FRAMES]

    if not frame_paths:
        raise RuntimeError(f"❌ {frame_dir} 下没有帧图片")

    nodes_count = {}            # label -> count
    relations_timeline = []     # 占位，暂时不写真实关系
    annotated_saved = False

    for frame_id, frame_path in enumerate(frame_paths):
        image_source, image = load_image(str(frame_path))
        h, w = image_source.shape[:2]

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=args.box_thresh,
            text_threshold=args.text_thresh,
        )

        if boxes.shape[0] == 0:
            relations_timeline.append({"frame": frame_id, "relations": []})
            continue

        # -----------------------------
        # 关键词过滤 + 按 keyword 做 Top-K
        # -----------------------------
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
            relations_timeline.append({"frame": frame_id, "relations": []})
            continue

        boxes_keep = torch.stack([x[0] for x in filtered], dim=0)
        logits_keep = torch.stack([x[1] for x in filtered], dim=0)
        phrases_keep = [x[2] for x in filtered]

        # -----------------------------
        # 统计 label 计数
        # -----------------------------
        for ph in phrases_keep:
            label = phrase_to_label(ph, task_name)
            nodes_count[label] = nodes_count.get(label, 0) + 1

        # relations_timeline 先占位，后面如果你想添加 “near / in / on” 再改
        relations_timeline.append({"frame": frame_id, "relations": []})

        # -----------------------------
        # 保存第一帧的可视化预览
        # -----------------------------
        if not annotated_saved and boxes_keep.shape[0] > 0:
            annotated = annotate(
                image_source=image_source,
                boxes=boxes_keep,
                logits=logits_keep,
                phrases=phrases_keep,
            )
            out_img = video_root / "pred_annotated_cliptrk.jpg"
            cv2.imwrite(str(out_img), annotated[..., ::-1])
            annotated_saved = True
            print(f"✅ Annotated preview saved: {out_img}")

    # ========================================================
    # 3.3 输出与旧版 Vid-B 完全兼容的 JSON 结构
    # ========================================================

    # nodes: list[{"label": str, "count": int}]
    nodes_list = [
        {"label": lbl, "count": int(cnt)}
        for lbl, cnt in sorted(nodes_count.items(), key=lambda x: (-x[1], x[0]))
    ]

    # edges 先留空，Vid-E 目前走纯 LLM，就不靠自动关系触发
    edges_list = []

    nodes_path = video_root / "yolo_nodes.json"
    scene_path = video_root / "yolo_scene_graph.json"
    rel_path = video_root / "relations_timeline.json"

    with open(nodes_path, "w") as f:
        json.dump(
            {
                "video_root": str(video_root),
                "generator": "GroundingDINO",
                "sampled_frames": len(frame_paths),
                "nodes": nodes_list,
            },
            f,
            indent=2,
        )

    with open(scene_path, "w") as f:
        json.dump(
            {
                "video_root": str(video_root),
                "generator": "GroundingDINO",
                "sampled_frames": len(frame_paths),
                "nodes": nodes_list,
                "edges": edges_list,
            },
            f,
            indent=2,
        )

    with open(rel_path, "w") as f:
        json.dump(relations_timeline, f, indent=2)

    print("\n✅ Vid-B Finished")
    print("nodes saved       :", nodes_path)
    print("scene graph saved :", scene_path)
    print("relations timeline:", rel_path)


if __name__ == "__main__":
    main()
