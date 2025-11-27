#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vid-E (research version)
------------------------
- Reads:
    - experiment YAML (tasks/experiment_xxx.yaml)
    - scene graph (VIDEO_ROOT/yolo_scene_graph.json)  [optional, may be empty]
    - relations timeline (VIDEO_ROOT/relations_timeline.json) [optional]
- Uses:
    - error_config.* in YAML (error_type, expected_tool, etc.)
- Produces:
    - VIDEO_ROOT/vid_e_explanation.txt    (LLM-facing explanation for user)
    - VIDEO_ROOT/vid_e_runlog.json        (machine log for analysis)

Error types supported:
    - wrong_tool
    - inability_to_pour
    - object_occluded
    - grasp_failure      (预留，将来你可以用在别的任务上)
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import yaml

# ----------------- LLM wrapper (reuse your existing LLMPrompter) -----------------
try:
    from LLM.prompt import LLMPrompter
except Exception:
    LLMPrompter = None


def load_json_or_none(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def safe_get(dct, *keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def summarize_timeline_for_debug(timeline, tool_list):
    """
    timeline: list of {"frame": int, "holding": [...], "near_apple": [...]}
    Return:
        - counts: {tool: count_of_frames_where_holding}
        - inferred_tool: tool with max count (or None)
    """
    counts = {t: 0 for t in tool_list}
    if not isinstance(timeline, list):
        return counts, None

    for frame_ev in timeline:
        holding = frame_ev.get("holding", [])
        if not isinstance(holding, list):
            continue
        for t in tool_list:
            if t in holding:
                counts[t] += 1

    inferred = None
    max_cnt = 0
    for t, c in counts.items():
        if c > max_cnt:
            max_cnt = c
            inferred = t

    if max_cnt == 0:
        inferred = None
    return counts, inferred


def build_llm_prompter(cfg_llm: dict):
    if LLMPrompter is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found in env; will fall back to rule-based text.")
        return None
    model = cfg_llm.get("model", "gpt-4o-mini")
    return LLMPrompter(gpt_version=model, api_key=api_key)


def build_wrong_tool_prompt(goal_en, context_en, expected, used, timing_mode):
    when_text = {
        "none": "You are NOT supposed to produce any explanation in this condition, "
                "but for debugging we still simulate what you would say.",
        "immediate": "You are explaining the error right after it happens, before or while you recover.",
        "delayed": "You are explaining the error after you have already recovered and completed the task.",
    }.get(timing_mode, "You are explaining the error around the time it happens.")

    user = f"""
Task goal:
- {goal_en}

Task context:
- {context_en.strip() if context_en else ""}

What happened:
- The robot selected the WRONG TOOL for the task.
- The expected correct tool was: "{expected}".
- The tool that appears to have been used instead is: "{used if used else "an incorrect tool"}".

Instructions:
- Explain to a non-technical user what went wrong and why.
- Be honest that you picked up the wrong tool.
- Mention that this makes the task harder or unsafe.
- Describe, step by step, how you will fix it and continue.
- Be concise, friendly, and use 2–4 short paragraphs or bullet points.
"""
    sys = (
        "You are a robot assistant reflecting on your own mistake in a user study. "
        "Your explanation will be spoken to a human participant. "
        f"{when_text} "
        "Avoid technical terms like 'bounding boxes', 'scene graph', or 'classifier'. "
        "Speak in plain English, in the first person as the robot (e.g., 'I tried to...')."
    )
    return sys, user


def build_inability_to_pour_prompt(goal_en, context_en, err_cfg, timing_mode):
    expected = safe_get(err_cfg, "expected_outcome_en", default="")
    failure = safe_get(err_cfg, "failure_mode_en", default="")
    suggested = safe_get(err_cfg, "suggested_recovery_en", default="")

    user = f"""
Task goal:
- {goal_en}

Task context:
- {context_en.strip() if context_en else ""}

What happened:
- The robot attempted to pour, but the liquid or cereal did not come out as expected.
- Expected outcome: {expected}
- Observed failure: {failure}

Suggested recovery plan (high level):
- {suggested}

Instructions:
- Explain, in simple terms, why the pouring attempt failed (e.g., bad angle, grasp, height, or blockage).
- Emphasize that you noticed the pour did not work and that this is a limitation or difficulty, not the user's fault.
- Describe clearly what you will do next to recover (adjust angle, reposition, or ask for help) and then try again.
- Use 2–4 short paragraphs or bullet points, friendly, first-person voice.
"""
    sys = (
        "You are a robot assistant explaining a failed pouring attempt to a non-technical user. "
        "Focus on what went wrong, why pouring was difficult, and what your recovery plan is. "
        "Align your explanation with the given context but you can phrase things naturally."
    )
    return sys, user


def build_object_occluded_prompt(goal_en, context_en, err_cfg, timing_mode):
    occluded_obj = safe_get(err_cfg, "occluded_object", default="the object")
    occluder = safe_get(err_cfg, "occluder", default="an obstacle")
    expected = safe_get(err_cfg, "expected_behavior_en", default="")
    uncertainty_desc = safe_get(err_cfg, "uncertainty_description_en", default="")

    user = f"""
Task goal:
- {goal_en}

Task context:
- {context_en.strip() if context_en else ""}

What happened:
- At first, I could not fully see the {occluded_obj} because it was hidden behind the {occluder}.
- This made me uncertain about the exact position of the object and caused me to hesitate or search.
- Explanation of uncertainty: {uncertainty_desc}

Expected behavior:
- {expected}

Instructions:
- Explain to the user that the object was occluded from your viewpoint.
- Emphasize that you needed to look around or move to get a better view before acting.
- Describe how you re-planned, moved, and then successfully continued the task.
- Use 2–4 short paragraphs or bullet points, in a friendly first-person tone.
"""
    sys = (
        "You are a robot assistant explaining a temporary uncertainty due to occlusion. "
        "Describe why you could not see the object at first, how you handled the situation, "
        "and how you eventually found and grasped the object."
    )
    return sys, user


def build_grasp_failure_prompt(goal_en, context_en, timing_mode):
    user = f"""
Task goal:
- {goal_en}

Task context:
- {context_en.strip() if context_en else ""}

What happened:
- I attempted to grasp the object, but the grasp failed (for example, I slipped, or only touched the edge).
- I detected that the grasp was unsuccessful.

Instructions:
- Explain what might have caused the grasp to fail (e.g., shape, slipperiness, bad angle).
- Reassure the user that this sometimes happens and is part of robot limitations.
- Describe what you will do next to recover (re-approach, adjust fingers, ask for help).
- Use 2–4 short paragraphs or bullet points in a friendly, first-person style.
"""
    sys = (
        "You are a robot assistant explaining a failed grasp attempt. "
        "Do not blame the user. Focus on your own limitations and your recovery plan."
    )
    return sys, user


def main():
    if len(sys.argv) < 3:
        print("Usage: python Vid-E.py <experiment_yaml> <video_root> [timing_mode] [participant_id]")
        sys.exit(1)

    yaml_path = Path(sys.argv[1])
    video_root = Path(sys.argv[2])
    timing_mode = sys.argv[3] if len(sys.argv) > 3 else "immediate"  # "none" | "immediate" | "delayed"
    participant_id = sys.argv[4] if len(sys.argv) > 4 else "pilot"

    assert yaml_path.exists(), f"Config not found: {yaml_path}"
    assert video_root.exists(), f"Video root not found: {video_root}"

    cfg = yaml.safe_load(yaml_path.read_text())

    task_name = safe_get(cfg, "meta", "task_name", default=cfg["meta"].get("task_id", ""))
    error_type = safe_get(cfg, "meta", "error_type", default=None)
    goal_en = safe_get(cfg, "task", "goal_en", default="The robot is performing a manipulation task.")
    context_en = safe_get(cfg, "task", "context_en", default="")

    # 读取 perception 结果（如果有的话，就算是空也没关系）
    timeline_path = video_root / "relations_timeline.json"
    graph_path = video_root / "yolo_scene_graph.json"
    timeline = load_json_or_none(timeline_path) or []
    scene_graph = load_json_or_none(graph_path) or {"nodes": [], "edges": []}

    print("✅ Using experiment config:", yaml_path.name)
    print("✅ Video root           :", str(video_root))
    print("⏱  Explanation timing  :", timing_mode)
    print("👤 Participant / run ID :", participant_id)

    print("=== Vid-E Context ===")
    print("task_name :", task_name)
    print("goal_en   :", goal_en)
    print("error_type:", error_type)

    # ----------- 根据 error_type 读 YAML 的 error_config -----------
    err_cfg = safe_get(cfg, "error_config", default={}) or {}
    explanation_cfg = safe_get(cfg, "explanation", default={})
    llm_cfg = safe_get(explanation_cfg, "llm", default={})

    # 默认：不触发 error
    error_triggered = False
    trigger_reason = ""
    used_tool = None
    expected_tool = None
    tool_timeline_counts = {}
    tool_graph_counts = {}

    # ---------------- wrong_tool 分支 ----------------
    if error_type == "wrong_tool":
        wt_cfg = safe_get(err_cfg, "wrong_tool", default={}) or {}
        expected_tool = wt_cfg.get("expected_tool")
        tool_candidates = wt_cfg.get("tool_candidates", []) or []
        force_trigger = bool(wt_cfg.get("force_trigger", False))
        assumed_wrong_tool = wt_cfg.get("assumed_wrong_tool")

        print("expected  :", expected_tool)
        print("candidates:", tool_candidates)

        # 尝试从 timeline 推断 used_tool（如果 Vid-B 有写 holding）
        tool_timeline_counts, inferred_tool = summarize_timeline_for_debug(timeline, tool_candidates)
        print("timeline_counts  :", tool_timeline_counts)
        used_tool = inferred_tool

        # 如果视觉完全没用上，就 fallback 到 YAML 里的 assumed_wrong_tool
        if used_tool is None and assumed_wrong_tool:
            used_tool = assumed_wrong_tool
            trigger_reason = "Using assumed_wrong_tool from YAML."
        elif used_tool is not None and expected_tool is not None:
            trigger_reason = f"inferred used_tool={used_tool} vs expected_tool={expected_tool}"

        # 触发逻辑：
        # - 如果 expected_tool 和 used_tool 都有且不相等 => error
        # - 如果 force_trigger=True 且 expected_tool / assumed_wrong_tool 给全了 => error
        if expected_tool and used_tool and used_tool != expected_tool:
            error_triggered = True
        elif force_trigger and expected_tool and assumed_wrong_tool:
            error_triggered = True
            if not trigger_reason:
                trigger_reason = "Force-triggered from YAML (force_trigger=true)."

        print("error_triggered:", error_triggered)
        print("reason         :", trigger_reason)

    # ---------------- inability_to_pour 分支 ----------------
    elif error_type == "inability_to_pour":
        ip_cfg = safe_get(err_cfg, "inability_to_pour", default={}) or {}
        force_trigger = bool(ip_cfg.get("force_trigger", False))
        # 目前我们不依赖视觉，只要是这类实验视频就视为发生错误
        error_triggered = force_trigger
        trigger_reason = "Force-triggered inability_to_pour from YAML."

    # ---------------- object_occluded 分支 ----------------
    elif error_type == "object_occluded":
        oo_cfg = safe_get(err_cfg, "object_occluded", default={}) or {}
        force_trigger = bool(oo_cfg.get("force_trigger", False))
        error_triggered = force_trigger
        trigger_reason = "Force-triggered object_occluded from YAML."

    # ---------------- grasp_failure 分支（预留） ----------------
    elif error_type == "grasp_failure":
        gf_cfg = safe_get(err_cfg, "grasp_failure", default={}) or {}
        force_trigger = bool(gf_cfg.get("force_trigger", True))
        error_triggered = force_trigger
        trigger_reason = "Force-triggered grasp_failure from YAML."

    else:
        trigger_reason = f"Error type '{error_type}' not yet implemented for automatic triggering."

    # 如果 timing_mode == "none"，表示实验的 baseline 条件，直接不生成解释
    if timing_mode == "none":
        print("⚪ timing_mode=none => Baseline condition, no user-facing explanation generated.")
        explanation_text = (
            "No explanation was generated because this run is in the baseline "
            "no-explanation condition."
        )
        llm_used = False
    else:
        # 如果错误没有实际触发，就生成一个“无错误”的简短说明（调试/对照用）
        if not error_triggered:
            print("error_triggered:", error_triggered)
            print("reason         :", trigger_reason)
            explanation_text = (
                "In this run, I did not detect a clear error based on the configured error type. "
                "Therefore, I am not generating a detailed error-aware explanation for you this time."
            )
            llm_used = False
        else:
            # 这里才是真正调用 LLM 生成“研究用解释”的地方
            prompter = build_llm_prompter(llm_cfg)
            explanation_text = None
            llm_used = False

            if error_type == "wrong_tool":
                sys_prompt, user_prompt = build_wrong_tool_prompt(
                    goal_en, context_en, expected_tool, used_tool, timing_mode
                )
            elif error_type == "inability_to_pour":
                sys_prompt, user_prompt = build_inability_to_pour_prompt(
                    goal_en, context_en, err_cfg.get("inability_to_pour", {}), timing_mode
                )
            elif error_type == "object_occluded":
                sys_prompt, user_prompt = build_object_occluded_prompt(
                    goal_en, context_en, err_cfg.get("object_occluded", {}), timing_mode
                )
            elif error_type == "grasp_failure":
                sys_prompt, user_prompt = build_grasp_failure_prompt(
                    goal_en, context_en, timing_mode
                )
            else:
                sys_prompt = (
                    "You are a robot assistant. Explain what kind of issue occurred "
                    "in a friendly and concise way."
                )
                user_prompt = f"""
Task goal:
- {goal_en}

Context:
- {context_en}

Error type:
- {error_type}

Instructions:
- Briefly explain that there was some problem executing the task.
- Suggest what you will try next time to improve.
"""

            if prompter is not None:
                print("✅ Using LLM.prompt for explanations")
                try:
                    prompt_obj = {"system": sys_prompt, "user": user_prompt}
                    explanation_text, _raw = prompter.query(
                        prompt=prompt_obj,
                        sampling_params={
                            "temperature": llm_cfg.get("temperature", 0.3),
                            "max_tokens": llm_cfg.get("max_tokens", 220),
                        },
                        save=False,
                        save_dir=str(video_root),
                    )
                    explanation_text = explanation_text.strip()
                    llm_used = True
                except Exception as e:
                    print("⚠️ LLM call failed, fallback to rule-based text:", e)

            if not explanation_text:
                # 兜底：如果 LLM 不可用
                if error_type == "wrong_tool":
                    explanation_text = (
                        f"I was supposed to use the {expected_tool} to complete this task, "
                        f"but I ended up using a different tool instead. I will put it down, "
                        f"pick up the correct {expected_tool}, and then continue the task more carefully."
                    )
                elif error_type == "inability_to_pour":
                    explanation_text = (
                        "I tried to pour, but nothing came out. This likely happened because "
                        "my angle or grasp was not good. I will adjust my position or ask for your "
                        "help, then try pouring again more carefully."
                    )
                elif error_type == "object_occluded":
                    explanation_text = (
                        "At first I could not clearly see the object because it was hidden "
                        "behind another item. I needed to move and look from a different angle "
                        "before I could continue."
                    )
                else:
                    explanation_text = (
                        "There was a problem while I was executing the task. I will adjust my "
                        "actions and try again more carefully."
                    )
                llm_used = False

    # ---------------- 写文件 ----------------
    out_txt = video_root / "vid_e_explanation.txt"
    out_log = video_root / "vid_e_runlog.json"

    out_txt.write_text(explanation_text.strip(), encoding="utf-8")

    runlog = {
        "timestamp": datetime.now().isoformat(),
        "config": yaml_path.name,
        "video_root": str(video_root),
        "participant_id": participant_id,
        "timing_mode": timing_mode,
        "task_name": task_name,
        "goal_en": goal_en,
        "error_type": error_type,
        "error_triggered": error_triggered,
        "trigger_reason": trigger_reason,
        "expected_tool": expected_tool,
        "used_tool": used_tool,
        "tool_timeline_counts": tool_timeline_counts,
        "scene_nodes": len(scene_graph.get("nodes", [])),
        "scene_edges": len(scene_graph.get("edges", [])),
        "llm_used": llm_used,
        "explanation": explanation_text.strip(),
    }
    out_log.write_text(json.dumps(runlog, indent=2), encoding="utf-8")

    print("\n=== Vid-E Explanation (LLM or fallback) ===")
    print(explanation_text.strip())
    print(f"\nSaved explanation -> {out_txt}")

    print("\n=== Vid-E Summary ===")
    print("Config     :", yaml_path.name)
    print("Video Root :", str(video_root))
    print("Error Type :", error_type)
    print("Expected   :", expected_tool)
    print("Used Tool  :", used_tool)
    print("Triggered  :", int(error_triggered))
    print("LLM Used   :", llm_used)
    print("Runlog     :", str(out_log))


if __name__ == "__main__":
    main()
