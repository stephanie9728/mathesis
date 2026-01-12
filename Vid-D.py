#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Vid-D.py (NO-VIS, ERROR-AWARE + FLAG OUTPUT)
# --------------------------------------------
# Usage:
#   python Vid-D.py <experiment_yaml> <video_root>
#
# Output:
#   <video_root>/explanation_plan.json
#   <video_root>/error_flag.json        <-- NEW (for Vid-E gating)
#
# Purpose:
#   - Load YAML meta
#   - Load Vid-B output
#   - Produce a structured explanation plan
#   - Emit a clear "error_confirmed" flag (pluggable for your real detector)

import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def error_type_to_observability(error_type: str) -> str:
    et = (error_type or "").strip().lower()
    if "belief" in et or "uncertainty" in et or "search" in et or "non_observable" in et:
        return "non_observable"
    if "motion" in et or "planning" in et or "execution" in et or "observable" in et:
        return "observable"
    return "unknown"


def summarize_nodes(nodes):
    nodes = nodes or []
    nodes = sorted(nodes, key=lambda x: (-int(x.get("count", 0)), str(x.get("label", ""))))
    top = nodes[:8]
    return [f"{n.get('label')}({n.get('count')})" for n in top]


def load_vidb_outputs(video_root: Path):
    nodes_path = video_root / "yolo_nodes.json"
    graph_path = video_root / "yolo_scene_graph.json"

    if nodes_path.exists():
        data = json.loads(nodes_path.read_text(encoding="utf-8"))
        return {
            "source": "yolo_nodes.json",
            "nodes": data.get("nodes", []),
            "empty_frame_ratio": data.get("empty_frame_ratio", None),
            "sampled_frames": data.get("sampled_frames", None),
            "task_id_from_b": data.get("task_id", None),
        }

    if graph_path.exists():
        data = json.loads(graph_path.read_text(encoding="utf-8"))
        return {
            "source": "yolo_scene_graph.json",
            "nodes": data.get("nodes", []),
            "empty_frame_ratio": None,
            "sampled_frames": None,
            "task_id_from_b": data.get("task_id", None),
        }

    raise FileNotFoundError("❌ Missing Vid-B outputs: yolo_nodes.json or yolo_scene_graph.json")


def build_rule_based_plan(meta, llm_hint, vidb):
    task_id = meta.get("task_id") or meta.get("task_name") or "unknown_task"
    task_name = meta.get("task_name", task_id)
    goal_en = meta.get("goal_en", "")
    error_type = meta.get("error_type", "")
    expected_tool = meta.get("expected_tool", None)

    observability = error_type_to_observability(error_type)

    nodes_summary = summarize_nodes(vidb.get("nodes"))
    empty_ratio = vidb.get("empty_frame_ratio", None)

    if observability == "non_observable":
        cause_summary = "I expected the object to be in a certain place, but it wasn’t there, so I wasn’t sure of its location."
        evidence = [
            "The first attempt did not find the expected object in the initial location.",
            "This suggests my prior assumption about the object’s location was incorrect.",
        ]
        recovery_steps = [
            "Stop the current attempt and reassess where the object might be.",
            "Search an alternative location (e.g., another drawer/area).",
            "Once found, grasp the object and present it to the camera.",
        ]
        immediate_script = "I expected it to be here, but I’m not finding it. I’ll check another place next."
        post_script = (
            "I initially expected it to be in the first place I checked, but it wasn’t there. "
            "After updating that assumption, I searched another location and then completed the task."
        )

    elif observability == "observable":
        cause_summary = "Something in the environment or my motion execution caused the failure during the task."
        evidence = [
            "A visible interruption occurred during motion (e.g., blockage or slippage).",
            "The robot paused and adjusted its action to continue safely.",
        ]
        recovery_steps = [
            "Pause to avoid compounding the failure.",
            "Adjust the action (replan path / adjust grasp / retry).",
            "Continue the motion and complete the placement/presentation.",
        ]
        immediate_script = "Something interfered with my motion, so I’m going to pause and adjust before continuing."
        post_script = (
            "An unexpected issue happened during the motion, so I paused and adjusted my action. "
            "After correcting it, I continued and finished the task."
        )

    else:
        cause_summary = "An issue occurred during the task, and I needed to recover."
        evidence = ["The task did not proceed as expected.", "A recovery action was required."]
        recovery_steps = ["Pause", "Try an alternative action", "Continue once stable"]
        immediate_script = "Something didn’t go as expected. I’ll adjust and continue."
        post_script = "Something didn’t go as expected, so I adjusted my approach and then continued."

    tool_phrase = ""
    if expected_tool and expected_tool != "null":
        tool_phrase = f" (target: {expected_tool})"

    if nodes_summary:
        evidence.append(f"Visible context included: {', '.join(nodes_summary)}")
    if empty_ratio is not None:
        evidence.append(f"Frames with no detections ratio: {empty_ratio:.2f}")

    plan = {
        "task_id": task_id,
        "task_name": task_name,
        "goal_en": goal_en,
        "error_type": error_type,
        "observability": observability,
        "inputs": {
            "llm_hint": llm_hint,
            "vidb_source": vidb.get("source"),
            "detected_objects_summary": nodes_summary,
            "empty_frame_ratio": empty_ratio,
        },
        "cause": {"summary": cause_summary + tool_phrase, "evidence": evidence[:4]},
        "recovery": {"summary": "A step-by-step plan to resolve the issue and complete the goal.", "steps": recovery_steps[:4]},
        "user_facing": {
            "immediate_script": immediate_script,
            "post_recovery_script": post_script,
        },
    }
    return plan


# ---------------- NEW: error flag logic ----------------

def _read_json_if_exists(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def decide_error_confirmed(meta: Dict[str, Any], video_root: Path) -> Dict[str, Any]:
    """
    Output a stable flag for Vid-E:
      error_confirmed: bool
      source: where it came from

    Priority (highest -> lowest):
    1) meta.error_confirmed (explicitly injected by your detector/controller)
    2) video_root/error_event.json or video_root/vid_c_runlog.json (if present)
    3) default:
       - if this is a window folder (..../windows/wXXXXXX): False
       - else (full run): True  (because your dataset/task is "an error scenario")
    """
    # 1) explicit override from YAML meta
    if "error_confirmed" in meta:
        return {
            "error_confirmed": bool(meta.get("error_confirmed")),
            "source": "meta.error_confirmed",
        }

    # 2) accept upstream detector files if you add them later
    #    (names are suggestions; your code can write either)
    error_event = _read_json_if_exists(video_root / "error_event.json")
    if error_event and "error_confirmed" in error_event:
        return {"error_confirmed": bool(error_event["error_confirmed"]), "source": "error_event.json"}

    vidc_runlog = _read_json_if_exists(video_root / "vid_c_runlog.json")
    if vidc_runlog:
        # tolerate a couple of key names
        for k in ("error_confirmed", "error_detected", "failure", "failed"):
            if k in vidc_runlog:
                return {"error_confirmed": bool(vidc_runlog[k]), "source": f"vid_c_runlog.json:{k}"}

    # 3) default behavior
    is_window = "windows" in [p.name for p in video_root.parents] or (video_root.name.startswith("w") and video_root.parent.name == "windows")
    if is_window:
        return {"error_confirmed": False, "source": "default(window)=False"}
    return {"error_confirmed": True, "source": "default(full)=True"}


def main():
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python Vid-D.py <experiment_yaml> <video_root>\n"
            "e.g.   python Vid-D.py tasks/experiment_missing_knife_two_drawers.yaml missing_knife_two_drawers"
        )

    yaml_path = Path(sys.argv[1]).resolve()
    video_root = Path(sys.argv[2]).resolve()

    print(f"📄 Vid-D config : {yaml_path}")
    print(f"📂 Vid-D video  : {video_root}")

    if not yaml_path.exists():
        raise FileNotFoundError(f"❌ YAML not found: {yaml_path}")
    if not video_root.exists():
        raise FileNotFoundError(f"❌ video_root not found: {video_root}")

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    meta = cfg.get("meta", {}) or {}
    llm_hint = (cfg.get("llm", {}) or {}).get("hint", "").strip()

    vidb = load_vidb_outputs(video_root)
    plan = build_rule_based_plan(meta, llm_hint, vidb)

    # save plan
    out_plan = video_root / "explanation_plan.json"
    out_plan.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    # NEW: save error flag for Vid-E gating
    flag = decide_error_confirmed(meta, video_root)
    flag_out = {
        "task_id": plan.get("task_id"),
        "error_confirmed": bool(flag["error_confirmed"]),
        "source": flag["source"],
        "observability": plan.get("observability"),
        "error_type": plan.get("error_type"),
        "ts": time.time() if "time" in globals() else None,  # safe if time not imported
    }
    # avoid importing time at top; keep it simple
    # write without ts if time isn't defined
    if flag_out["ts"] is None:
        flag_out.pop("ts", None)

    out_flag = video_root / "error_flag.json"
    out_flag.write_text(json.dumps(flag_out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("✅ Explanation plan saved:", out_plan)
    print("✅ Error flag saved       :", out_flag)
    print(f"   error_confirmed = {flag_out['error_confirmed']} ({flag_out['source']})")
    print(f"   observability  = {plan['observability']}")
    print("✅ Vid-D finished")


if __name__ == "__main__":
    main()
