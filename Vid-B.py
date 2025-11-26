# Vid-B-CLIP-BYTETRACK (task-agnostic, no GUI, no ximgproc)
# ------------------------------------------------
# Usage:
#   python Vid-B.py /path/to/VIDEO_ROOT
#
# Expect:
#   VIDEO_ROOT/raw.mp4 exists
#
# Outputs:
#   VIDEO_ROOT/yolo_nodes.json
#   VIDEO_ROOT/yolo_scene_graph.json
#   VIDEO_ROOT/pred_annotated_cliptrk.jpg
#   VIDEO_ROOT/relations_timeline.json   # <-- Vid-E 用它做时序推理

import os
import sys
import math
import json
from pathlib import Path
from collections import defaultdict, deque

import cv2
import torch
import numpy as np
from PIL import Image
import open_clip
from supervision import Detections
import importlib

# ============== Paths & Args ==============

VIDEO_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./camera_demo_fruit")
VIDEO_ROOT = VIDEO_ROOT.resolve()
video_path = str(VIDEO_ROOT / "raw.mp4")
assert (VIDEO_ROOT / "raw.mp4").exists(), f"找不到 {VIDEO_ROOT}/raw.mp4，请先合成视频"

MAX_FRAMES       = 300     # 至多处理多少帧
FRAME_STRIDE     = 5       # 每 N 帧采一帧
GRID_SIZE        = 4       # 4~6 之间可调
SCORE_THRESH     = 0.35
NMS_IOU          = 0.5
NEAR_K           = 0.25
HOLD_MIN_FRAMES  = 2
NEAR_MIN_FRAMES  = 3
SHOW_DEBUG_EVERY = 0       # 0=关闭; >0 每 N 帧输出调试图（目前只写文件不弹窗）

# ============== Classes & Prompts (任务无关) ==============
# 尽量覆盖你现有/计划的 task：切水果、倒麦片/水、抓苹果/遮挡等
CLASSES = [
    "person",
    "apple",
    "banana",
    "bowl",
    "cup",
    "mug",
    "table",
    "cutting board",
    "cereal box",
    "milk carton",
    "bottle",
    "knife",
    "fork",
    "spoon",
]

ALIAS = {
    "knife":         ["kitchen knife", "chef knife", "cutting knife", "sharp knife"],
    "fork":          ["dining fork", "table fork", "metal fork"],
    "spoon":         ["table spoon", "metal spoon"],
    "bowl":          ["ceramic bowl", "white bowl"],
    "table":         ["desk surface", "wooden table", "dining table"],
    "cutting board": ["chopping board", "wooden chopping board"],
    "cereal box":    ["cereal carton", "cereal package"],
    "milk carton":   ["milk box", "milk bottle"],
}

TEXT_TEMPLATES = [
    "a photo of a {}",
    "a close-up photo of a {}",
    "a {} on a kitchen table",
    "a hand holding a {}",
    "a {} used for preparing food"
]

# ============== Torch↔NumPy 自检 ==============
try:
    _ = torch.from_numpy(np.zeros((1, 1), dtype=np.float32))
except Exception as e:
    raise RuntimeError(
        f"[ENV ERROR] torch↔numpy 互操作失败：{e}\n"
        "建议版本：numpy≈1.26.x, torch≈2.x；并确保子进程使用同一解释器(sys.executable)。"
    )

# ============== Init CLIP ==============
# 为了跨机器稳定，这里直接用 CPU；如果你之后确认 GPU 可用，可以改成自动检测。
device = "cpu"

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")


def build_text_embeds(classes):
    phrases, owners = [], []
    for cls in classes:
        names = [cls] + ALIAS.get(cls, [])
        for n in names:
            for t in TEXT_TEMPLATES:
                phrases.append(t.format(n))
                owners.append(cls)
    with torch.no_grad():
        text_tokens = tokenizer(phrases).to(device)
        text_feats = clip_model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return phrases, owners, text_feats


PHRASES, OWNERS, TEXT_EMB = build_text_embeds(CLASSES)

# ============== Proposals: Grid（不依赖 ximgproc） ==============


def gen_proposals(img_bgr, grid=GRID_SIZE):
    H, W = img_bgr.shape[:2]
    boxes = []
    step_y, step_x = max(1, H // grid), max(1, W // grid)
    for i in range(grid):
        for j in range(grid):
            x1, y1 = j * step_x, i * step_y
            x2, y2 = min(W, x1 + step_x), min(H, y1 + step_y)
            if (x2 - x1) * (y2 - y1) >= 400:  # 过滤过小块
                boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32)

# ============== CLIP Scoring ==============


def clip_score_crops(img_bgr, boxes_xyxy):
    crops = []
    for (x1, y1, x2, y2) in boxes_xyxy.astype(int):
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crops.append(None)
            continue
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append(preprocess(Image.fromarray(crop)).unsqueeze(0))
    valid_idx = [i for i, c in enumerate(crops) if c is not None]
    if not valid_idx:
        return [], [], []
    imgs = torch.cat([crops[i] for i in valid_idx], dim=0).to(device)

    with torch.no_grad():
        image_feats = clip_model.encode_image(imgs)
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_feats @ TEXT_EMB.T)
        probs = logits.softmax(dim=-1)  # [N, num_phrases]

    owner_masks = []
    for ci, cls in enumerate(CLASSES):
        owner_masks.append(torch.tensor(
            [1 if OWNERS[j] == cls else 0 for j in range(len(OWNERS))],
            device=probs.device,
            dtype=torch.bool
        ))

    cls_scores, cls_ids = [], []
    for row in probs:
        per_cls = [
            row[owner_masks[ci]].max().item() if owner_masks[ci].any() else 0.0
            for ci in range(len(CLASSES))
        ]
        cls_scores.append(per_cls)
        cls_ids.append(int(np.argmax(per_cls)))

    out_boxes, out_cls, out_prob = [None] * len(boxes_xyxy), [None] * len(boxes_xyxy), [None] * len(boxes_xyxy)
    for k, i in enumerate(valid_idx):
        out_boxes[i] = boxes_xyxy[i]
        out_cls[i] = cls_ids[k]
        out_prob[i] = max(cls_scores[k])
    return out_boxes, out_cls, out_prob

# ============== NMS ==============


def nms_xyxy(boxes, scores, iou_th=0.5):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_th)[0]
        order = order[inds + 1]
    return keep

# ============== ByteTrack（兼容 supervision 旧 API） ==============

try:
    bt_mod = importlib.import_module("supervision.tracker.byte_tracker.core")
    BTClass = getattr(bt_mod, "ByteTrack")
    tracker = BTClass()
    for k, v in {
        "track_thresh": 0.5,
        "match_thresh": 0.8,
        "track_buffer": 30,
        "frame_rate": 30,
    }.items():
        if hasattr(tracker, k):
            setattr(tracker, k, v)
    print("✅ ByteTrack 初始化完成 ✅")
except Exception as e:
    raise ImportError("无法初始化 ByteTrack，请确认 supervision 版本>=0.22.0") from e

# ============== 关系统计 & 时间线 & 调试计数 ==============

pair_near_hist = defaultdict(lambda: deque(maxlen=20))   # frozenset({idA,idB}) -> [bool]
pair_hold_hist = defaultdict(lambda: deque(maxlen=20))   # (person_id, obj_id) -> [bool]
edges_count, nodes_count = {}, {}
timeline = []  # 每帧记录 {"frame": i, "holding":[...], "relations":[{...}]}

# 专门统计一些常见“工具/容器”的出现次数，方便 debug
TOOL_DEBUG = {name: 0 for name in ["knife", "fork", "spoon", "cup", "bowl", "cereal box", "apple"]}


def add_edge(a, r, b):
    if a == b:
        return
    edges_count[(a, r, b)] = edges_count.get((a, r, b), 0) + 1


def inside(pt, box):
    x1, y1, x2, y2 = box
    return (x1 <= pt[0] <= x2) and (y1 <= pt[1] <= y2)


# ============== 主循环 ==============

cap = cv2.VideoCapture(video_path)
assert cap.isOpened()
frame_idx = 0
processed_frames = 0
first_annot = None

while True and processed_frames < MAX_FRAMES:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_idx % FRAME_STRIDE != 0:
        frame_idx += 1
        continue

    H, W = frame.shape[:2]
    diag = math.hypot(W, H)

    # 1) proposals（网格）
    boxes = gen_proposals(frame, GRID_SIZE)
    if len(boxes) == 0:
        frame_idx += 1
        processed_frames += 1
        continue

    # 2) CLIP 评分
    bxs, cls_ids, probs = clip_score_crops(frame, boxes)
    det_boxes, det_scores, det_classes = [], [], []
    for b, cid, p in zip(bxs, cls_ids, probs):
        if b is None or cid is None or p is None:
            continue
        if p < SCORE_THRESH:
            continue
        det_boxes.append(b)
        det_scores.append(float(p))
        det_classes.append(int(cid))

    # 3) NMS（按类）
    final_boxes, final_scores, final_classes = [], [], []
    for cls_id in range(len(CLASSES)):
        idx = [i for i, c in enumerate(det_classes) if c == cls_id]
        if not idx:
            continue
        cls_boxes = [det_boxes[i] for i in idx]
        cls_scores = [det_scores[i] for i in idx]
        keep = nms_xyxy(cls_boxes, cls_scores, NMS_IOU)
        for k in keep:
            final_boxes.append(cls_boxes[k])
            final_scores.append(cls_scores[k])
            final_classes.append(cls_id)

    # 4) ByteTrack
    if len(final_boxes):
        xyxy = np.array(final_boxes, dtype=np.float32)
        scores = np.array(final_scores, dtype=np.float32)
        class_id = np.array(final_classes, dtype=int)
        det = Detections(xyxy=xyxy, confidence=scores, class_id=class_id)
        tracks = tracker.update_with_detections(det)
    else:
        try:
            tracks = tracker.update_with_detections(Detections.empty())
        except Exception:
            empty_det = Detections(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros((0,), dtype=np.float32),
                class_id=np.zeros((0,), dtype=int),
            )
            tracks = tracker.update_with_detections(empty_det)

    # 5) 节点 & 关系 + 时间线记录
    id_map = {}
    if len(tracks) and getattr(tracks, "tracker_id", None) is not None:
        for b, tid, cid, conf in zip(tracks.xyxy, tracks.tracker_id, tracks.class_id, tracks.confidence):
            label = CLASSES[int(cid)]
            id_map[int(tid)] = (label, b, float(conf))
            nodes_count[label] = nodes_count.get(label, 0) + 1

    frame_event = {
        "frame": processed_frames,
        "holding": [],      # ["knife", "apple"]
        "relations": []     # [{"type":"near","a":"cereal box","b":"bowl"}, ...]
    }

    ids = list(id_map.keys())
    for i in range(len(ids)):
        ii = ids[i]
        li, bi, _ = id_map[ii]
        xi = np.array([(bi[0] + bi[2]) / 2, (bi[1] + bi[3]) / 2])
        ai = max(0.0, (bi[2] - bi[0]) * (bi[3] - bi[1]))
        for j in range(i + 1, len(ids)):
            jj = ids[j]
            lj, bj, _ = id_map[jj]
            xj = np.array([(bj[0] + bj[2]) / 2, (bj[1] + bj[3]) / 2])
            aj = max(0.0, (bj[2] - bj[0]) * (bj[3] - bj[1]))

            # 粗略左右/上下
            if xi[0] + 50 < xj[0]:
                add_edge(li, "left_of", lj)
                frame_event["relations"].append({"type": "left_of", "a": li, "b": lj})
            if xi[1] + 50 < xj[1]:
                add_edge(li, "above", lj)
                frame_event["relations"].append({"type": "above", "a": li, "b": lj})

            # near（时序投票）
            near_now = np.linalg.norm(xi - xj) < NEAR_K * diag
            key_near = frozenset({ii, jj})
            pair_near_hist[key_near].append(near_now)
            if sum(pair_near_hist[key_near]) >= NEAR_MIN_FRAMES:
                add_edge(li, "near", lj)
                add_edge(lj, "near", li)
                frame_event["relations"].append({"type": "near", "a": li, "b": lj})
                frame_event["relations"].append({"type": "near", "a": lj, "b": li})

            # holding（时序投票）
            if li == "person" and lj != "person":
                hold_now = inside(xj, bi) and (aj < 0.35 * ai)
                pair_hold_hist[(ii, jj)].append(hold_now)
                if sum(pair_hold_hist[(ii, jj)]) >= HOLD_MIN_FRAMES:
                    add_edge("person", "holding", lj)
                    frame_event["holding"].append(lj)
            if lj == "person" and li != "person":
                hold_now = inside(xi, bj) and (ai < 0.35 * aj)
                pair_hold_hist[(jj, ii)].append(hold_now)
                if sum(pair_hold_hist[(jj, ii)]) >= HOLD_MIN_FRAMES:
                    add_edge("person", "holding", li)
                    frame_event["holding"].append(li)

    # holding / relations 去重
    frame_event["holding"] = sorted(list(set(frame_event["holding"])))
    # relations 可以简单去重
    uniq_rel = []
    seen_rel = set()
    for r in frame_event["relations"]:
        key = (r["type"], r["a"], r["b"])
        if key in seen_rel:
            continue
        seen_rel.add(key)
        uniq_rel.append(r)
    frame_event["relations"] = uniq_rel

    # 工具 debug 计数
    for name in TOOL_DEBUG.keys():
        if name in frame_event["holding"]:
            TOOL_DEBUG[name] += 1

    timeline.append(frame_event)

    # 第一次可视化一帧到文件
    if (processed_frames == 0) and len(id_map):
        vis = frame.copy()
        for tid, (lbl, b, sc) in id_map.items():
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{lbl}#{tid} {sc:.2f}",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(VIDEO_ROOT / "pred_annotated_cliptrk.jpg"), vis)

    frame_idx += 1
    processed_frames += 1

cap.release()

# ============== 输出 JSON ==============

nodes = [
    {"label": k, "count": v}
    for k, v in sorted(nodes_count.items(), key=lambda x: (-x[1], x[0]))
]
edges = [
    {"s": a, "r": r, "o": b, "count": c}
    for (a, r, b), c in sorted(edges_count.items(), key=lambda x: -x[1])
]

(VIDEO_ROOT / "yolo_nodes.json").write_text(
    json.dumps(
        {
            "video_root": str(VIDEO_ROOT),
            "generator": "CLIP+ByteTrack(grid)",
            "sampled_frames": processed_frames,
            "nodes": nodes,
        },
        indent=2,
    ),
    encoding="utf-8",
)

(VIDEO_ROOT / "yolo_scene_graph.json").write_text(
    json.dumps(
        {
            "video_root": str(VIDEO_ROOT),
            "generator": "CLIP+ByteTrack(grid)",
            "sampled_frames": processed_frames,
            "nodes": nodes,
            "edges": edges,
        },
        indent=2,
    ),
    encoding="utf-8",
)

(VIDEO_ROOT / "relations_timeline.json").write_text(
    json.dumps(timeline, indent=2), encoding="utf-8"
)

print("nodes saved       :", VIDEO_ROOT / "yolo_nodes.json")
print("scene graph saved :", VIDEO_ROOT / "yolo_scene_graph.json")
print("annotated preview :", VIDEO_ROOT / "pred_annotated_cliptrk.jpg")
print("relations timeline:", VIDEO_ROOT / "relations_timeline.json")
print("🔍 TOOL DEBUG COUNTS:", TOOL_DEBUG)
