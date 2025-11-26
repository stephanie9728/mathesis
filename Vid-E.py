# Vid-E: Research version (task-agnostic error reasoning + LLM explanation)
# ---------------------------------------------------------------------
# Usage:
#   python Vid-E.py EXP_YAML VIDEO_ROOT [TIMING] [PID]
#
# Inputs:
#   EXP_YAML   : e.g. tasks/experiment_cut_fruit_tool_error_mini.yaml
#   VIDEO_ROOT : e.g. camera_demo_fruit
#   TIMING     : "none" | "immediate" | "delayed"  (optional; default "immediate")
#   PID        : participant id or "pilot"         (optional)
#
# Requires:
#   VIDEO_ROOT/relations_timeline.json
#   VIDEO_ROOT/yolo_scene_graph.json   (optional but recommended)
#   VIDEO_ROOT/yolo_scene_explanation.txt (from Vid-C, optional)

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
import os

# ========== 1. Parse args ==========

if len(sys.argv) < 3:
    print("Usage: python Vid-E.py EXP_YAML VIDEO_ROOT [TIMING] [PID]")
    sys.exit(1)

EXP_PATH = Path(sys.argv[1]).resolve()
VIDEO_ROOT = Path(sys.argv[2]).resolve()
TIMING = sys.argv[3] if len(sys.argv) > 3 else "immediate"
PID = sys.argv[4] if len(sys.argv) > 4 else "pilot"

assert EXP_PATH.exists(), f"Config not found: {EXP_PATH}"
assert VIDEO_ROOT.exists(), f"Video root not found: {VIDEO_ROOT}"

print(f"✅ Using experiment config: {EXP_PATH.name}")
print(f"✅ Video root           : {VIDEO_ROOT}")
print(f"⏱  Explanation timing  : {TIMING}")
print(f"👤 Participant / run ID : {PID}")

# ========== 2. Load YAML config ==========

CFG = yaml.safe_load(EXP_PATH.read_text())
meta: Dict[str, Any] = CFG.get("meta", {})

task_name: str = meta.get("task_name", EXP_PATH.stem)
goal_en: str = meta.get("goal_en", "The robot is performing a manipulation task.")
error_type: str = meta.get("error_type", "none")

# For wrong_tool tasks (backward compatible with your existing YAML)
expected_tool: Optional[str] = meta.get("expected_tool")
tool_candidates: List[str] = meta.get("tool_candidates", [])

print("=== Vid-E Context ===")
print("task_name :", task_name)
print("goal_en   :", goal_en)
print("error_type:", error_type)
if error_type == "wrong_tool":
    print("expected  :", expected_tool)
    print("candidates:", tool_candidates)

# ========== 3. Load perception outputs (Vid-B + Vid-C) ==========

timeline_path = VIDEO_ROOT / "relations_timeline.json"
graph_path = VIDEO_ROOT / "yolo_scene_graph.json"
scene_expl_path = VIDEO_ROOT / "yolo_scene_explanation.txt"

timeline: List[Dict[str, Any]] = []
scene_graph: Dict[str, Any] = {"nodes": [], "edges": []}
scene_summary: str = ""

if timeline_path.exists():
    timeline = json.loads(timeline_path.read_text())
    print(f"📈 Loaded relations_timeline with {len(timeline)} frames")
else:
    print(f"⚠️ relations_timeline.json not found at {timeline_path}")

if graph_path.exists():
    scene_graph = json.loads(graph_path.read_text())
    print(f"📄 Loaded scene graph with {len(scene_graph.get('edges', []))} edges")
else:
    print(f"⚠️ yolo_scene_graph.json not found at {graph_path}")

if scene_expl_path.exists():
    scene_summary = scene_expl_path.read_text(encoding="utf-8").strip()
else:
    scene_summary = ""
    print(f"⚠️ yolo_scene_explanation.txt not found at {scene_expl_path}")

# ========== 4. Helper: infer used tool / relations ==========

def infer_used_tool_from_timeline_and_graph(
    timeline: List[Dict[str, Any]],
    tool_candidates: List[str],
    scene_graph: Dict[str, Any],
) -> (Optional[str], Dict[str, int], Dict[str, int]):
    """
    For wrong_tool tasks: count how often each candidate tool appears
    as 'holding' in the timeline, and how often it appears in scene graph edges.
    """
    # timeline counts
    t_counts = {t: 0 for t in tool_candidates}
    for frame in timeline:
        holding = frame.get("holding", [])
        for lbl in holding:
            if lbl in t_counts:
                t_counts[lbl] += 1

    # graph counts (person --holding--> tool)
    g_counts = {t: 0 for t in tool_candidates}
    for e in scene_graph.get("edges", []):
        if e.get("r") == "holding" and e.get("s") == "person":
            obj = e.get("o")
            if obj in g_counts:
                g_counts[obj] += e.get("count", 1)

    # pick best candidate from timeline first; if all zero, fall back to graph
    used_tool = None
    if t_counts and max(t_counts.values()) > 0:
        used_tool = max(t_counts.items(), key=lambda kv: kv[1])[0]
    elif g_counts and max(g_counts.values()) > 0:
        used_tool = max(g_counts.items(), key=lambda kv: kv[1])[0]

    return used_tool, t_counts, g_counts

# ========== 5. Error detection (now only wrong_tool; can extend later) ==========

error_triggered: bool = False
trigger_reason: str = ""
used_tool: Optional[str] = None

if error_type == "wrong_tool" and expected_tool and tool_candidates:
    used_tool, t_counts, g_counts = infer_used_tool_from_timeline_and_graph(
        timeline, tool_candidates, scene_graph
    )

    print("inferred used_tool:", used_tool)
    print("timeline_counts   :", t_counts)
    print("graph_counts      :", g_counts)

    if used_tool is None:
        error_triggered = False
        trigger_reason = "Cannot determine used_tool or expected_tool."
    else:
        if used_tool != expected_tool:
            error_triggered = True
            trigger_reason = f"used_tool={used_tool} != expected_tool={expected_tool}"
        else:
            error_triggered = False
            trigger_reason = "used_tool matches expected_tool."
else:
    # For now, other error types are not implemented; can be extended later.
    error_triggered = False
    trigger_reason = f"Error type '{error_type}' not yet implemented for automatic triggering."

print("error_triggered:", error_triggered)
print("reason         :", trigger_reason)

# ========== 6. LLM interface (research version) ==========

llm = None
try:
    from LLM.prompt import LLMPrompter

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    llm = LLMPrompter(gpt_version="gpt-4o-mini", api_key=api_key)
    print("✅ Using LLM.prompt (gpt-4o-mini) for explanations")
except Exception as e:
    print("⚠️ LLM.prompt not available or API key missing; will fall back to template explanation only.")
    llm = None

# ========== 7. Build explanation prompt ==========

def build_llm_prompt_for_wrong_tool(
    goal: str,
    used_tool: str,
    expected_tool: str,
    scene_summary: str,
    timing: str,
) -> Dict[str, str]:
    """
    Construct system/user prompts for wrong_tool explanations.
    """
    system_prompt = (
        "You are a friendly household robot explaining your own mistake to a non-technical user. "
        "You must:\n"
        "1) Briefly describe what went wrong in everyday language (no technical terms like 'scene graph' or 'timeline').\n"
        "2) Explain why the behavior was suboptimal or incorrect, in 1–2 sentences.\n"
        "3) Present a concrete recovery plan in 3–4 short bullet points.\n"
        "4) Write in first-person as the robot (e.g., 'I picked up...').\n"
        "5) Be concise but reassuring, and stay within 150–200 words.\n"
    )

    # timing hint purely for style (not for correctness)
    if timing == "immediate":
        timing_clause = "I am telling you this right away, while I am still in the middle of the task."
    elif timing == "delayed":
        timing_clause = "I am explaining this now that I have already recovered from the mistake."
    else:
        timing_clause = "I am explaining this to help you understand what happened."

    user_prompt = (
        f"Task goal:\n{goal}\n\n"
        f"Observed tool usage:\n"
        f"- I used: {used_tool}\n"
        f"- I should have used: {expected_tool}\n\n"
        f"Scene description (from my perception modules):\n"
        f"{scene_summary if scene_summary else '(No detailed scene summary available.)'}\n\n"
        f"Context about timing:\n{timing_clause}\n\n"
        "Please generate a user-facing explanation in English that:\n"
        "- Names the incorrect tool and the correct tool.\n"
        "- Explains why the incorrect tool choice is problematic.\n"
        "- Clearly lays out what I will do next to correct the situation.\n"
        "- Avoids mentioning internal labels like 'error_type' or 'expected_tool'.\n"
    )

    return {"system": system_prompt, "user": user_prompt}

# ========== 8. Generate final explanation text ==========

explanation_text: str = ""

if error_triggered and error_type == "wrong_tool" and used_tool and expected_tool:
    if llm is not None:
        prompts = build_llm_prompt_for_wrong_tool(
            goal=goal_en,
            used_tool=used_tool,
            expected_tool=expected_tool,
            scene_summary=scene_summary,
            timing=TIMING,
        )
        try:
            text, _meta = llm.query(
                prompt=prompts,
                sampling_params={"temperature": 0.4, "max_tokens": 220},
                save=False,
                save_dir=str((VIDEO_ROOT / "..").resolve()),
            )
            explanation_text = text.strip()
        except Exception as e:
            print("⚠️ LLM call failed, falling back to template:", e)
            explanation_text = (
                f"I was supposed to use the {expected_tool} to complete this task, "
                f"but I mistakenly used the {used_tool} instead. "
                "This tool is not ideal for the action I needed to perform. "
                "I will now switch to the correct tool and redo the step more carefully."
            )
    else:
        explanation_text = (
            f"I was supposed to use the {expected_tool} to complete this task, "
            f"but I mistakenly used the {used_tool} instead. "
            "This tool is not ideal for the action I needed to perform. "
            "I will now switch to the correct tool and redo the step more carefully."
        )
else:
    # No error triggered or unsupported type -> 简短说明，不调 LLM
    explanation_text = (
        "In this run, I did not detect a clear mismatch between the tool I used and the "
        "tool specified in the task. Because of that, I am not generating a detailed "
        "error explanation for you this time."
    )

print("\n=== Vid-E Explanation (LLM or fallback) ===")
print(explanation_text)

# ========== 9. Save explanation & run log ==========

exp_out_path = VIDEO_ROOT / "vid_e_explanation.txt"
exp_out_path.write_text(explanation_text, encoding="utf-8")

runlog = {
    "config": EXP_PATH.name,
    "video_root": str(VIDEO_ROOT),
    "timing": TIMING,
    "pid": PID,
    "task_name": task_name,
    "goal_en": goal_en,
    "error_type": error_type,
    "expected_tool": expected_tool,
    "tool_candidates": tool_candidates,
    "used_tool": used_tool,
    "error_triggered": error_triggered,
    "trigger_reason": trigger_reason,
    "llm_used": bool(error_triggered and error_type == "wrong_tool" and used_tool and expected_tool and llm),
}

runlog_path = VIDEO_ROOT / "vid_e_runlog.json"
runlog_path.write_text(json.dumps(runlog, indent=2), encoding="utf-8")

print(f"\nSaved explanation -> {exp_out_path}")
print("\n=== Vid-E Summary ===")
print("Config     :", EXP_PATH.name)
print("Video Root :", VIDEO_ROOT)
print("Error Type :", error_type)
print("Used Tool  :", used_tool)
print("Expected   :", expected_tool)
print("Triggered  :", 1 if error_triggered else 0)
print("LLM Used   :", runlog['llm_used'])
print("Runlog     :", runlog_path)
