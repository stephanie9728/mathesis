#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vid-E: LLM-based error explanation (purely config driven)

Usage:
  python Vid-E.py tasks/experiment_grasp_apple_uncertainty.yaml /path/to/VIDEO_ROOT

Assumptions:
  - VIDEO_ROOT contains outputs from Vid-B:
      yolo_nodes.json
      yolo_scene_graph.json
      relations_timeline.json
  - YAML config has the structure:

    meta:
      task_id: "grasp_apple_uncertainty"
      task_name: "Grasp apple: uncertainty about location"
      goal_en: "Grasp the apple ..."
      error_type: "uncertainty"      # or "wrong_tool" / "inability"
      explanation_timing: "immediate"

    tools:
      candidates: []                 # optional
      correct: null                  # optional

    llm:
      hint: >                        # long, task-specific hint
        ...

This script NO LONGER tries to auto-detect errors.
If meta.error_type is one of {"uncertainty", "wrong_tool", "inability"},
we *always* generate an error-aware explanation using the LLM.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
import openai  # 使用旧版 SDK 接口：openai.ChatCompletion.create

# 默认模型名（在 YAML 的 llm.model 里可以覆盖）
DEFAULT_MODEL = "gpt-4o-mini"

# 从环境变量读取 API Key（推荐做法）
openai.api_key = os.environ.get("OPENAI_API_KEY", "")

# ============================================================
# 1. 读 Vid-B 输出
# ============================================================

def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️ Failed to load {path}: {e}")
        return default


def summarize_nodes(nodes_json: Dict[str, Any], top_k: int = 6) -> str:
    nodes = nodes_json.get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        return "I did not detect any specific objects in the scene graph."

    nodes_sorted = sorted(
        nodes,
        key=lambda x: -x.get("count", 0)
    )[:top_k]

    lines = []
    for n in nodes_sorted:
        label = n.get("label", "object")
        count = n.get("count", 0)
        lines.append(f"- {label}: observed {count} times across frames.")
    return "From the scene graph, I detected:\n" + "\n".join(lines)


def summarize_relations(rel_json: Any, max_frames: int = 5) -> str:
    if not isinstance(rel_json, list) or not rel_json:
        return "I did not infer any explicit spatial relations over time."

    num_frames = len(rel_json)
    sample_frames = rel_json[:max_frames]
    frames_str = ", ".join(str(f.get("frame", i)) for i, f in enumerate(sample_frames))
    return (
        f"There is a relation timeline with {num_frames} frames "
        f"(showing relations for frames: {frames_str}). "
        "Currently no specific relations are listed, but this indicates how long the scene was observed."
    )


# ============================================================
# 2. 构造 LLM 提示词
# ============================================================

def build_system_prompt(task_name: str, goal_en: str, error_type: str, hint: str) -> str:
    return f"""
You are a helpful, polite assistive kitchen robot speaking to a non-expert human user.

Your goal is to naturally explain:
- What you were trying to do;
- What you currently see in the scene;
- What went wrong;
- Why this led to uncertainty or difficulty for you.

Speak in the first person ("I").
Use a calm, slightly detailed, and reassuring tone.
It is OK if the explanation is a bit longer than a few sentences.
Do NOT mention any technical system details, models, confidence scores, or probabilities.

Task name: {task_name}
Intended goal: {goal_en}
Configured error type: {error_type}

After your explanation, you MUST add a short recovery plan in the following format:

Next steps:
- bullet point
- bullet point
- bullet point (2–4 bullets total)

The "Next steps" should describe what you will do next to improve the situation and how you will act more safely or carefully.

Additional experiment-specific guidance:
{hint}
""".strip()




def build_user_prompt(
    nodes_summary: str,
    relations_summary: str,
    error_type: str,
    expected_tool: Optional[str],
    tools_section: Dict[str, Any],
) -> str:
    lines: List[str] = []

    lines.append("Here is a summary of what I detected from the video scene:")
    lines.append("")
    lines.append(nodes_summary)
    lines.append("")
    lines.append(relations_summary)
    lines.append("")
    lines.append(f"The experiment config sets error_type = '{error_type}'.")

    if expected_tool:
        lines.append(f"The config suggests that the expected tool is: '{expected_tool}'.")

    candidates = tools_section.get("candidates")
    if isinstance(candidates, list) and candidates:
        lines.append(f"Tool candidates mentioned in the config: {candidates}.")

    lines.append("")
    lines.append(
        "Please now produce a single, user-facing explanation describing the error "
        "condition in this specific run, in a way that would make sense to a non-expert user."
    )

    return "\n".join(lines)


# ============================================================
# 3. 调用 LLM（旧版 openai.ChatCompletion 接口）
# ============================================================

def call_llm(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    使用旧版 openai.ChatCompletion 接口。
    需要环境变量 OPENAI_API_KEY 已设置，或者提前设置 openai.api_key。
    """
    if not openai.api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please export it in your environment "
            "or set openai.api_key manually."
        )

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=256,
    )
    return resp["choices"][0]["message"]["content"].strip()


# ============================================================
# 4. 主入口
# ============================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python Vid-E.py <experiment.yaml> <VIDEO_ROOT>")
        sys.exit(1)

    yaml_path = Path(sys.argv[1]).resolve()
    video_root = Path(sys.argv[2]).resolve()

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    meta = cfg.get("meta", {})
    tools_cfg = cfg.get("tools", {})
    llm_cfg = cfg.get("llm", {})

    task_id = meta.get("task_id", "unknown_task")
    task_name = meta.get("task_name", task_id)
    goal_en = meta.get("goal_en", "The robot is performing a manipulation task.")
    error_type = meta.get("error_type", "none")
    explanation_timing = meta.get("explanation_timing", "immediate")

    expected_tool = meta.get("expected_tool") or tools_cfg.get("correct")
    llm_hint = llm_cfg.get("hint", "").strip()

    print("✅ Using experiment config:", yaml_path.name)
    print("✅ Video root           :", video_root)
    print("⏱  Explanation timing  :", explanation_timing)
    print("👤 Task / ID            :", task_name, "/", task_id)

    # -----------------------------
    # 4.1 读取 Vid-B 输出
    # -----------------------------
    nodes_json = load_json(video_root / "yolo_nodes.json", default={})
    scene_json = load_json(video_root / "yolo_scene_graph.json", default={})
    rel_json = load_json(video_root / "relations_timeline.json", default=[])

    nodes_summary = summarize_nodes(nodes_json)
    relations_summary = summarize_relations(rel_json)

    # -----------------------------
    # 4.2 决定是否“触发错误”
    # -----------------------------

    # 现在设计：只要 error_type 不是 "none"，就当成 error trial
    error_type_normalized = (error_type or "").strip().lower()

    if error_type_normalized and error_type_normalized != "none":
        error_triggered = True
        reason = (
            f"Config meta.error_type='{error_type}' → "
            f"treat this run as an error trial (no automatic detection)."
        )
    else:
        error_triggered = False
        reason = (
            f"Error type '{error_type}' means this run is treated as a non-error trial, "
            f"so no error-aware explanation is generated."
        )

    # -----------------------------
    # 4.3 调用 LLM 或输出 fallback
    # -----------------------------
    if error_triggered:
        model_name = llm_cfg.get("model", DEFAULT_MODEL)

        system_prompt = build_system_prompt(
            task_name=task_name,
            goal_en=goal_en,
            error_type=error_type,
            hint=llm_hint,
        )
        user_prompt = build_user_prompt(
            nodes_summary=nodes_summary,
            relations_summary=relations_summary,
            error_type=error_type,
            expected_tool=expected_tool,
            tools_section=tools_cfg,
        )

        print("✅ Using LLM (", model_name, ") for explanations")
        try:
            explanation_text = call_llm(system_prompt, user_prompt, model=model_name)
            llm_used = True
        except Exception as e:
            explanation_text = (
                "I was supposed to generate a detailed explanation using the LLM, "
                f"but there was an error when calling the model: {e}"
            )
            llm_used = False
    else:
        explanation_text = (
            "In this run, I did not treat the behavior as an error trial based on the "
            f"configured error_type = '{error_type}'. Therefore, I am not generating a "
            "detailed error-aware explanation."
        )

    # -----------------------------
    # 4.4 保存结果 & 打印 summary
    # -----------------------------
    out_txt = video_root / "vid_e_explanation.txt"
    out_txt.write_text(explanation_text, encoding="utf-8")

    runlog = {
        "config": str(yaml_path),
        "video_root": str(video_root),
        "task_id": task_id,
        "task_name": task_name,
        "goal_en": goal_en,
        "error_type": error_type,
        "expected_tool": expected_tool,
        "error_triggered": error_triggered,
        "reason": reason,
        "llm_used": llm_used,
        "nodes_summary": nodes_summary,
        "relations_summary": relations_summary,
    }
    (video_root / "vid_e_runlog.json").write_text(
        json.dumps(runlog, indent=2), encoding="utf-8"
    )

    print("\n=== Vid-E Explanation (LLM or fallback) ===")
    print(explanation_text)
    print(f"\nSaved explanation -> {out_txt}")

    print("\n=== Vid-E Summary ===")
    print("Config     :", yaml_path.name)
    print("Video Root :", video_root)
    print("Error Type :", error_type)
    if expected_tool:
        print("Expected   :", expected_tool)
    print("Triggered  :", int(error_triggered))
    print("LLM Used   :", llm_used)
    print("Runlog     :", video_root / "vid_e_runlog.json")


if __name__ == "__main__":
    main()
