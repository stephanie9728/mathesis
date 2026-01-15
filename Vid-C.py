#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Vid-C.py — Generic Error Detector (YAML-driven, window-stable)
#
# Usage:
#   python Vid-C.py <experiment_yaml> <video_root>
#
# Output:
#   <video_root>/vid_c_runlog.json
#
# Supports:
#   detector.type:
#     - missing_object
#     - external_flag
#     - always / never
#
# missing_object extras:
#   - present_count_ge: int (default 1)
#   - gate_presence_any: [..] (optional)
#   - require_empty_ratio_le: float (optional)
#   - min_total_nodes: int (optional; SOFT)
#   - confirm_k: int (default 1): require K consecutive windows missing before confirm
#   - state_file: str (default ".detector_state.json") stored under <task>/windows/
#   - label_aliases: {target_label: [alias1, alias2, ...]} (optional)
#       Example:
#         label_aliases:
#           black_cup: [cup]
#
# Rationale:
#   - "not enough context" should not hard-block triggering; otherwise windows become unstable.
#   - confirm_k provides stability across windows.

import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Union

import yaml


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def try_load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return load_json(p)
    except Exception:
        return None


def normalize_label(s: str) -> str:
    return (s or "").strip().lower().replace("-", "_").replace(" ", "_")


def load_scene_graph(video_root: Path) -> Dict[str, Any]:
    p = video_root / "yolo_scene_graph.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} (run Vid-B first)")
    return load_json(p)


def extract_labels_and_counts(graph: Dict[str, Any]) -> Tuple[List[str], Dict[str, int]]:
    nodes = graph.get("nodes", []) or []
    labels: List[str] = []
    counts: Dict[str, int] = {}
    for n in nodes:
        lb = normalize_label(n.get("label", ""))
        if not lb:
            continue
        labels.append(lb)
        try:
            counts[lb] = counts.get(lb, 0) + int(n.get("count", 1) or 1)
        except Exception:
            counts[lb] = counts.get(lb, 0) + 1
    return labels, counts


def best_match_count(counts: Dict[str, int], target: str) -> int:
    """
    "Soft" matching:
      - exact
      - substring both ways (t in lb) or (lb in t)
    """
    t = normalize_label(target)
    best = 0
    for lb, c in counts.items():
        if (t == lb) or (t in lb) or (lb in t):
            best = max(best, int(c))
    return best


def best_match_count_any(counts: Dict[str, int], targets: List[str]) -> Tuple[int, str]:
    """
    Return (best_count, best_target_matched)
    """
    best = 0
    best_t = ""
    for t in targets:
        c = best_match_count(counts, t)
        if c > best:
            best = c
            best_t = t
    return best, best_t


def gate_passed(labels: List[str], gate_any: List[str]) -> bool:
    if not gate_any:
        return True
    gate_norm = [normalize_label(x) for x in gate_any if str(x).strip()]
    for g in gate_norm:
        if any((g == lb) or (g in lb) or (lb in g) for lb in labels):
            return True
    return False


def find_task_windows_root(video_root: Path) -> Path:
    """
    video_root can be:
      - <task_id>
      - <task_id>/windows/w000090
    We store state under <task_id>/windows/.detector_state.json
    """
    if video_root.name.startswith("w") and video_root.parent.name == "windows":
        return video_root.parent
    return video_root / "windows"


def load_state(state_path: Path) -> Dict[str, Any]:
    st = try_load_json(state_path)
    if not st:
        return {"missing_streak": 0, "last_update": 0.0}
    if "missing_streak" not in st:
        st["missing_streak"] = 0
    return st


def save_state(state_path: Path, st: Dict[str, Any]):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(st, indent=2), encoding="utf-8")


def to_str_list(x: Any) -> List[str]:
    """
    Allow YAML to pass watch_object as:
      - "bottle"
      - ["black_cup", "cup"]
    """
    if x is None:
        return []
    if isinstance(x, list):
        out: List[str] = []
        for it in x:
            s = str(it).strip()
            if s:
                out.append(s)
        return out
    s = str(x).strip()
    return [s] if s else []


def expand_watch_targets(
    watch_list: List[str],
    label_aliases: Dict[str, Any],
) -> List[str]:
    """
    Expand watch targets using alias map:
      label_aliases:
        black_cup: [cup]
    Result: ["black_cup", "cup"]
    """
    aliases_norm: Dict[str, List[str]] = {}
    if isinstance(label_aliases, dict):
        for k, v in label_aliases.items():
            kk = normalize_label(str(k))
            vv = to_str_list(v)
            aliases_norm[kk] = [normalize_label(x) for x in vv if str(x).strip()]

    expanded: List[str] = []
    for w in watch_list:
        wn = normalize_label(w)
        if not wn:
            continue
        expanded.append(wn)
        for a in aliases_norm.get(wn, []):
            if a and a not in expanded:
                expanded.append(a)

    # de-dup keep order
    seen = set()
    out: List[str] = []
    for t in expanded:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def detect_missing_object(
    graph: Dict[str, Any],
    watch_targets: List[str],
    empty_frame_ratio: Optional[float],
    require_empty_ratio_le: float,
    min_total_nodes: int,
    gate_any: List[str],
    present_count_ge: int,
    confirm_k: int,
    state_path: Path,
) -> Tuple[bool, str, Dict[str, Any]]:
    labels, counts = extract_labels_and_counts(graph)
    total_nodes = len(labels)

    watch_count, matched_target = best_match_count_any(counts, watch_targets)
    present = (watch_count >= int(present_count_ge))

    cam_ok = True
    if empty_frame_ratio is not None:
        cam_ok = (float(empty_frame_ratio) <= float(require_empty_ratio_le))

    gate_ok = gate_passed(labels, gate_any)

    # NOTE: min_total_nodes is SOFT (won't block); we only record it
    enough_context = (total_nodes >= int(min_total_nodes))

    debug = {
        "watch_targets": watch_targets,
        "matched_target": matched_target,
        "watch_count": watch_count,
        "present_count_ge": int(present_count_ge),
        "present": present,
        "total_nodes": total_nodes,
        "min_total_nodes": int(min_total_nodes),
        "enough_context": enough_context,
        "empty_frame_ratio": empty_frame_ratio,
        "require_empty_ratio_le": float(require_empty_ratio_le),
        "cam_ok": cam_ok,
        "gate_presence_any": gate_any,
        "gate_ok": gate_ok,
        "confirm_k": int(confirm_k),
        "state_path": str(state_path),
        "labels_top": labels[:25],
    }

    # Hard gates: if fail, do NOT trigger and reset streak
    st = load_state(state_path)
    if not gate_ok:
        st["missing_streak"] = 0
        st["last_update"] = time.time()
        save_state(state_path, st)
        debug["missing_streak"] = st["missing_streak"]
        return False, "missing_object: gate not passed", debug

    if not cam_ok:
        st["missing_streak"] = 0
        st["last_update"] = time.time()
        save_state(state_path, st)
        debug["missing_streak"] = st["missing_streak"]
        return False, "missing_object: camera too empty (empty_frame_ratio high)", debug

    # Missing logic + streak
    if not present:
        st["missing_streak"] = int(st.get("missing_streak", 0)) + 1
    else:
        st["missing_streak"] = 0

    st["last_update"] = time.time()
    save_state(state_path, st)

    debug["missing_streak"] = st["missing_streak"]

    if not present:
        if st["missing_streak"] >= int(confirm_k):
            extra = ""
            if not enough_context:
                extra = " (context low but allowed)"
            # show primary intended target for readability (first in list)
            primary = watch_targets[0] if watch_targets else "watch_object"
            return True, f"missing_object: '{primary}' missing (streak={st['missing_streak']}/{confirm_k}){extra}", debug
        else:
            return False, f"missing_object: missing but waiting confirm_k (streak={st['missing_streak']}/{confirm_k})", debug

    return False, "missing_object: not triggered (object considered present)", debug


def detect_external_flag(video_root: Path, file: str, key: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Look for flag file in:
      1) window root (video_root/file)
      2) task root (video_root/../..  if video_root is .../windows/wXXXXXX)
    """
    tried: List[str] = []

    def try_path(p: Path):
        tried.append(str(p))
        data = try_load_json(p)
        if not data:
            return None
        return data

    # 1) window-local
    p1 = video_root / file
    data = try_path(p1)

    # 2) fallback: task root
    if data is None:
        # window dir: <task>/windows/w000090  -> task root is parents[2] == <task>
        # If someone passes <task> directly, parents[2] might not exist, so guard.
        task_root = None
        try:
            if video_root.name.startswith("w") and video_root.parent.name == "windows":
                task_root = video_root.parent.parent
        except Exception:
            task_root = None

        if task_root is not None:
            p2 = task_root / file
            data = try_path(p2)

    if data is None:
        return False, f"external_flag: missing {file}", {"file": file, "key": key, "tried": tried}

    val = bool(data.get(key, False))
    return val, f"external_flag: {key}={val}", {
        "file": file, "key": key, "value": data.get(key, None), "tried": tried
    }


def main():
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python Vid-C.py <experiment_yaml> <video_root>")

    yaml_path = Path(sys.argv[1]).resolve()
    video_root = Path(sys.argv[2]).resolve()

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    meta = cfg.get("meta", {}) or {}
    det = cfg.get("detector", {}) or {}

    det_type = (det.get("type") or "never").strip().lower()

    # Optional empty ratio
    yolo_nodes = try_load_json(video_root / "yolo_nodes.json")
    empty_frame_ratio = None
    if yolo_nodes and isinstance(yolo_nodes, dict):
        empty_frame_ratio = yolo_nodes.get("empty_frame_ratio", None)

    error_confirmed = False
    reason = ""
    debug: Dict[str, Any] = {"detector_type": det_type}

    try:
        if det_type in ("none", "never", ""):
            error_confirmed = False
            reason = "detector=never"

        elif det_type == "always":
            error_confirmed = True
            reason = "detector=always"

        elif det_type == "missing_object":
            # watch_object can be string or list
            watch_raw: Union[str, List[str]] = det.get("watch_object", "")
            if det.get("watch_object_from_meta"):
                meta_key = str(det.get("watch_object_from_meta"))
                # meta[meta_key] might be string; we keep raw and normalize later
                watch_raw = meta.get(meta_key, watch_raw)

            watch_list = to_str_list(watch_raw)

            if not watch_list:
                error_confirmed = False
                reason = "missing_object: watch_object not set"
            else:
                graph = load_scene_graph(video_root)
                gate_any = list(det.get("gate_presence_any", []) or [])
                require_empty_ratio_le = float(det.get("require_empty_ratio_le", 0.9))
                min_total_nodes = int(det.get("min_total_nodes", 1))
                present_count_ge = int(det.get("present_count_ge", 1))
                confirm_k = int(det.get("confirm_k", 1))
                state_file = str(det.get("state_file", ".detector_state.json"))
                label_aliases = det.get("label_aliases", {}) or {}

                windows_root = find_task_windows_root(video_root)
                state_path = windows_root / state_file

                watch_targets = expand_watch_targets(watch_list, label_aliases)

                error_confirmed, reason, d2 = detect_missing_object(
                    graph=graph,
                    watch_targets=watch_targets,
                    empty_frame_ratio=empty_frame_ratio,
                    require_empty_ratio_le=require_empty_ratio_le,
                    min_total_nodes=min_total_nodes,
                    gate_any=gate_any,
                    present_count_ge=present_count_ge,
                    confirm_k=confirm_k,
                    state_path=state_path,
                )
                debug.update(d2)
                debug["label_aliases"] = label_aliases

        elif det_type == "external_flag":
            file = str(det.get("file", "robot_error.json"))
            key = str(det.get("key", "error_confirmed"))
            error_confirmed, reason, d2 = detect_external_flag(video_root, file=file, key=key)
            debug.update(d2)

        else:
            error_confirmed = False
            reason = f"unknown detector type: {det_type}"

    except Exception as e:
        error_confirmed = False
        reason = f"detector exception: {e}"

    runlog = {
        "config": str(yaml_path),
        "task_id": meta.get("task_id"),
        "error_type": meta.get("error_type"),
        "object": meta.get("object"),
        "error_confirmed": bool(error_confirmed),
        "reason": reason,
        "empty_frame_ratio": empty_frame_ratio,
        "debug": debug,
        "ts": time.time(),
    }
    out = video_root / "vid_c_runlog.json"
    out.write_text(json.dumps(runlog, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ Vid-C detector: error_confirmed={error_confirmed} ({reason})")
    print("Runlog:", out)


if __name__ == "__main__":
    main()
