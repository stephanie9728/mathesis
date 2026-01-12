#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vid-B (GroundingDINO clean + task-id driven + window-safe)

Input:
  VIDEO_ROOT/frames/*.png or *.jpg

Output:
  VIDEO_ROOT/yolo_nodes.json
  VIDEO_ROOT/yolo_scene_graph.json
  VIDEO_ROOT/relations_timeline.json
  VIDEO_ROOT/pred_annotated_cliptrk.jpg (optional)

Design (paper-friendly):
- Vid-B only extracts visual facts (object mentions + counts).
- No error_type / belief / intention encoding here.
- High-level reasoning belongs to Vid-D / Vid-E.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    annotate,
)

# ============================================================
# 1) task_id -> prompt (single source of truth)
# ============================================================

TASK_PROMPTS = {
    "missing_knife_two_drawers": (
        "knife . drawer . top drawer . bottom drawer . table . hand"
    ),
    "fork_grasp_feasibility_uncertainty": (
        "fork . table . hand"
    ),
    "bottle_grasp_slippage": (
        "bottle . plastic bottle . table . shelf . hand"
    ),
    "path_obstruction_cup_transport": (
        "cup . tray . table . obstacle . hand"
    ),
}

# ============================================================
# 2) Defaults
# ============================================================

DEFAULT_MAX_FRAMES = 150
DEFAULT_FRAME_STRIDE = 10
DEFAULT_SAVE_PREVIEW = False


# ============================================================
# Helpers
# ============================================================

def extract_keywords_from_prompt(prompt: str) -> List[str]:
    kws = []
    for chunk in prompt.split("."):
        kw = chunk.strip().lower()
        if kw:
            kws.append(kw)
    return kws


def infer_task_id(video_root: Path) -> str:
    """
    Support both:
      - <task_id>
      - <task_id>/windows/w000030
    """
    # window path: <task_id>/windows/w000030
    if video_root.parent.name == "windows" and video_root.name.startswith("w"):
        return video_root.parent.parent.name
    return video_root.name


def phrase_to_label(phrase: str, keywords: List[str]) -> str:
    lower = (phrase or "").lower()

    for kw in keywords:
        if kw in lower:
            if "knife" in kw:
                return "knife"
            if "fork" in kw:
                return "fork"
            if "bottle" in kw:
                return "bottle"
            if "cup" in kw:
                return "cup"
            if "tray" in kw:
                return "tray"
            if "shelf" in kw:
                return "shelf"
            if "drawer" in kw:
                return "drawer"
            if "obstacle" in kw or "object" in kw:
                return "obstacle"
            if "table" in kw:
                return "table"
            if "hand" in kw:
                return "hand"
            return kw

    return "object"


def choose_topk_for_keyword(kw: str, keywords: List[str], default_topk: int = 1) -> int:
    """
    Keep the policy explicit and stable:
    - For 'drawer' variants, allow more candidates (helps find both drawers/handles).
    - Otherwise default top-1.
    """
    kw = kw.lower().strip()
    if "drawer" in kw:
        return 2
    return default_topk


def gather_frames(frame_dir: Path, max_frames: int, frame_stride: int) -> List[Path]:
    frame_paths = sorted(frame_dir.glob("*.png")) + sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.jpeg"))
    frame_paths = frame_paths[:max_frames]
    frame_paths = frame_paths[::max(1, frame_stride)]
    return frame_paths


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_root", type=str)

    parser.add_argument(
        "--config",
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--weights",
        default="GroundingDINO/weights/groundingdino_swint_ogc.pth",
    )

    parser.add_argument("--box_thresh", type=float, default=0.35)
    parser.add_argument("--text_thresh", type=float, default=0.25)

    parser.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--frame_stride", type=int, default=DEFAULT_FRAME_STRIDE)
    parser.add_argument("--save_preview", action="store_true", default=DEFAULT_SAVE_PREVIEW)

    # NEW: manual override if needed
    parser.add_argument("--task_id", default="", help="Optional: override inferred task_id")

    args = parser.parse_args()

    video_root = Path(args.video_root).resolve()
    frame_dir = video_root / "frames"

    if not frame_dir.exists():
        raise FileNotFoundError(f"❌ frames 目录不存在: {frame_dir}")

    # window-safe task_id inference
    task_id = args.task_id.strip() or infer_task_id(video_root)
    if task_id not in TASK_PROMPTS:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Known tasks: {list(TASK_PROMPTS.keys())}"
        )

    prompt = TASK_PROMPTS[task_id]
    keywords = extract_keywords_from_prompt(prompt)

    print("\n✅ Vid-B")
    print(f"  video_root : {video_root}")
    print(f"  task_id    : {task_id}")
    print(f"  prompt     : {prompt}")
    print(f"  keywords   : {keywords}")
    print(f"  max_frames : {args.max_frames}")
    print(f"  stride     : {args.frame_stride}")
    print(f"  thresh     : box={args.box_thresh}, text={args.text_thresh}")
    print(f"  preview    : {args.save_preview}\n")

    # -----------------------------
    # Load GroundingDINO
    # -----------------------------
    print("🔄 Loading GroundingDINO...")
    model = load_model(args.config, args.weights)
    print("✅ GroundingDINO loaded\n")

    # -----------------------------
    # Frames
    # -----------------------------
    frame_paths = gather_frames(frame_dir, args.max_frames, args.frame_stride)
    if not frame_paths:
        raise RuntimeError(f"❌ {frame_dir} 下没有帧图片")

    nodes_count = {}
    relations_timeline = []
    empty_frames = 0
    annotated_saved = False

    # -----------------------------
    # Per-frame inference
    # -----------------------------
    for frame_id, frame_path in enumerate(frame_paths):
        image_source, image = load_image(str(frame_path))

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=args.box_thresh,
            text_threshold=args.text_thresh,
        )

        if boxes is None or boxes.shape[0] == 0:
            empty_frames += 1
            relations_timeline.append({"frame": frame_id, "relations": []})
            continue

        # keyword-bucket filtering
        filtered: List[Tuple[torch.Tensor, torch.Tensor, str]] = []
        for kw in keywords:
            bucket = []
            for b, l, p in zip(boxes, logits, phrases):
                if kw in (p or "").lower():
                    bucket.append((b, l, p))
            if bucket:
                bucket = sorted(bucket, key=lambda x: float(x[1]), reverse=True)
                topk = choose_topk_for_keyword(kw, keywords, default_topk=1)
                filtered.extend(bucket[:topk])

        if not filtered:
            empty_frames += 1
            relations_timeline.append({"frame": frame_id, "relations": []})
            continue

        boxes_keep = torch.stack([x[0] for x in filtered], dim=0)
        logits_keep = torch.stack([x[1] for x in filtered], dim=0)
        phrases_keep = [x[2] for x in filtered]

        for ph in phrases_keep:
            label = phrase_to_label(ph, keywords)
            nodes_count[label] = nodes_count.get(label, 0) + 1

        relations_timeline.append({"frame": frame_id, "relations": []})

        if args.save_preview and not annotated_saved:
            annotated = annotate(
                image_source=image_source,
                boxes=boxes_keep,
                logits=logits_keep,
                phrases=phrases_keep,
            )
            out_img = video_root / "pred_annotated_cliptrk.jpg"
            cv2.imwrite(str(out_img), annotated[..., ::-1])
            annotated_saved = True

    # ========================================================
    # Outputs
    # ========================================================
    nodes_list = [
        {"label": k, "count": int(v)}
        for k, v in sorted(nodes_count.items(), key=lambda x: (-x[1], x[0]))
    ]

    summary = {
        "task_id": task_id,
        "video_root": str(video_root),
        "generator": "GroundingDINO",
        "prompt": prompt,
        "keywords": keywords,
        "params": {
            "max_frames": int(args.max_frames),
            "frame_stride": int(args.frame_stride),
            "box_thresh": float(args.box_thresh),
            "text_thresh": float(args.text_thresh),
        },
        "sampled_frames": len(frame_paths),
        "empty_frame_ratio": (empty_frames / max(1, len(frame_paths))),
        "nodes": nodes_list,
    }

    (video_root / "yolo_nodes.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (video_root / "yolo_scene_graph.json").write_text(
        json.dumps({**summary, "edges": []}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (video_root / "relations_timeline.json").write_text(
        json.dumps(relations_timeline, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n✅ Vid-B Finished")
    print("nodes saved       : yolo_nodes.json")
    print("scene graph saved : yolo_scene_graph.json")
    print("relations timeline: relations_timeline.json")


if __name__ == "__main__":
    main()
