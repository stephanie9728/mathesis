#!/usr/bin/env python
# Vid-E: research version – LLM-based failure explanation (English)

import os
import sys
import json
import time
from pathlib import Path

import yaml

# ========= 1. 参数解析 =========

if len(sys.argv) < 3:
    print("Usage: python Vid-E.py <experiment_yaml> <video_root>")
    sys.exit(1)

EXP_PATH = Path(sys.argv[1]).resolve()
VIDEO_ROOT = Path(sys.argv[2]).resolve()

assert EXP_PATH.exists(), f"Experiment YAML not found: {EXP_PATH}"
assert VIDEO_ROOT.exists(), f"Video root not found: {VIDEO_ROOT}"

print(f"✅ Using experiment config: {EXP_PATH.name}")
print(f"✅ Video root           : {VIDEO_ROOT}")

# ========= 2. 读 YAML 配置 =========

CFG = yaml.safe_load(EXP_PATH.read_text())
meta = CFG.get("meta", {})
tools_cfg = CFG.get("tools", {})
llm_cfg = CFG.get("llm", {})

task_id   = meta.get("task_id", EXP_PATH.stem)
task_name = meta.get("task_name", task_id)
goal_en   = meta.get("goal_en", "No goal description provided.")
error_type = meta.get("error_type", "wrong_tool")
expected_tool = meta.get("expected_tool", tools_cfg.get("correct"))

tool_candidates = tools_cfg.get("candidates", ["knife", "fork", "spoon"])
llm_hint = llm_cfg.get("hint", "")

print("=== Vid-E Context ===")
print("task_name :", task_name)
print("goal_en   :", goal_en)
print("error_type:", error_type)
print("expected  :", expected_tool)
print("candidates:", tool_candidates)

# ========= 3. 读取 timeline + scene graph =========

timeline_path = VIDEO_ROOT / "relations_timeline.json"
scene_graph_path = VIDEO_ROOT / "yolo_scene_graph.json"

frames = []
if timeline_path.exists():
    raw = json.loads(timeline_path.read_text())
    if isinstance(raw, dict) and "frames" in raw:
        frames = raw["frames"]
    elif isinstance(raw, list):
        frames = raw
    else:
        frames = []
    print(f"📈 Loaded relations_timeline with {len(frames)} frames")
else:
    print(f"⚠️ No relations_timeline.json found at {timeline_path}")

scene_graph = {}
edges = []
if scene_graph_path.exists():
    scene_graph = json.loads(scene_graph_path.read_text())
    edges = scene_graph.get("edges", [])
    print(f"📄 Loaded scene graph with {len(edges)} edges")
else:
    print(f"⚠️ No yolo_scene_graph.json found at {scene_graph_path}")

# ========= 4. 从 timeline / scene graph 推断 used_tool =========

def infer_used_tool_from_timeline(frames, candidates):
    if not frames:
        return None, {}

    counts = {t: 0 for t in candidates}
    for fr in frames:
        holding = fr.get("holding", []) or []
        near_apple = fr.get("near_apple", []) or []
        for t in candidates:
            if t in holding:
                counts[t] += 2   # holding 权重大
            if t in near_apple:
                counts[t] += 1

    best_tool = None
    best_score = 0
    for t, c in counts.items():
        if c > best_score:
            best_tool, best_score = t, c

    if best_score == 0:
        return None, counts
    return best_tool, counts

def infer_used_tool_from_graph(edges, candidates):
    if not edges:
        return None, {}

    counts = {t: 0 for t in candidates}
    for e in edges:
        s = e.get("s")
        r = e.get("r")
        o = e.get("o")
        c = int(e.get("count", 1))

        # 近似认为 tool near apple 说明它是被用来操作苹果
        if r == "near":
            if s == "apple" and o in counts:
                counts[o] += c
            if o == "apple" and s in counts:
                counts[s] += c

    best_tool = None
    best_score = 0
    for t, v in counts.items():
        if v > best_score:
            best_tool, best_score = t, v

    if best_score == 0:
        return None, counts
    return best_tool, counts

used_tool, timeline_counts = infer_used_tool_from_timeline(frames, tool_candidates)
if used_tool is None:
    used_tool, graph_counts = infer_used_tool_from_graph(edges, tool_candidates)
else:
    graph_counts = {}

print("inferred used_tool:", used_tool)
print("timeline_counts   :", timeline_counts)
if graph_counts:
    print("graph_counts      :", graph_counts)

# ========= 5. 判断是否触发错误 =========

error_triggered = False
reason = ""

if error_type == "wrong_tool":
    if used_tool is not None and expected_tool is not None:
        if used_tool != expected_tool:
            error_triggered = True
            reason = f"used_tool={used_tool} != expected_tool={expected_tool}"
        else:
            reason = "used_tool matches expected_tool – no wrong-tool error."
    else:
        reason = "Cannot determine used_tool or expected_tool."
else:
    # 其他类型（如 inability / uncertainty），暂时默认都触发一次解释
    error_triggered = True
    reason = f"Non-wrong_tool error_type={error_type}, explanation always triggered (research mode)."

print("error_triggered:", error_triggered)
print("reason         :", reason)

# ========= 6. 组装给 LLM 的 prompt =========

# 从 scene graph 中抽一点 object / relation，给 LLM 当 context
nodes_for_llm = []
if scene_graph:
    for n in scene_graph.get("nodes", [])[:10]:
        nodes_for_llm.append(f'{n.get("label","?")}×{n.get("count",0)}')

rel_for_llm = []
for e in edges[:15]:
    rel_for_llm.append(f'{e.get("s")} —{e.get("r")}→ {e.get("o")} (×{e.get("count",1)})')

scene_brief = "none"
if nodes_for_llm or rel_for_llm:
    scene_brief = (
        "Objects: " + ", ".join(nodes_for_llm) + "\n"
        "Relations: " + ("; ".join(rel_for_llm) if rel_for_llm else "none")
    )

# ========= 7. 调用 LLM 生成英文解释 =========

explanation_text = ""
llm_used = False

if error_triggered:
    try:
        from LLM.prompt import LLMPrompter

        class ThrottledPrompter(LLMPrompter):
            def __init__(self, *args, rpm_cap=4, **kwargs):
                super().__init__(*args, **kwargs)
                self.min_interval = 60.0 / max(1, rpm_cap)
                self._last = 0.0

            def query(self, *a, **k):
                wait = self.min_interval - (time.time() - self._last)
                if wait > 0:
                    time.sleep(wait)
                out = super().query(*a, **k)
                self._last = time.time()
                return out

        api_key = os.environ.get("OPENAI_API_KEY")
        assert api_key, "OPENAI_API_KEY environment variable not set."

        prompter = ThrottledPrompter(gpt_version="gpt-4o-mini", api_key=api_key)

        system_prompt = (
            "You are an assistive kitchen robot talking to a non-expert user. "
            "Your job is to explain what went wrong in the last attempt and "
            "what you will do to fix it.\n\n"
            "Constraints:\n"
            "- Answer in clear, natural English.\n"
            "- Be friendly and reassuring, and avoid any technical details "
            "  (no mention of models, neural networks, confidence scores, etc.).\n\n"
            "Output format:\n"
            "1. First give 2–3 sentences that explain:\n"
            "   - what task you were trying to complete;\n"
            "   - what actually went wrong;\n"
            "   - why this happened (e.g., wrong tool, could not pour the cereal, "
            "     misjudged where the apple was, etc.).\n"
            "2. Then add a short heading “Next I will:” and give 2–4 bullet points "
            "   describing your concrete recovery plan and how you will avoid the same error next time.\n"
            "3. Speak in the first person (“I”) and address the user directly.\n"
        )

        # 研究者给的额外提示（来自 YAML）
        hint_block = f"Researcher hint: {llm_hint}\n" if llm_hint else ""

        user_prompt = (
            f"{hint_block}"
            f"Task name: {task_name}\n"
            f"High-level goal: {goal_en}\n"
            f"Error type: {error_type}\n"
            f"Expected tool (if applicable): {expected_tool}\n"
            f"Tool I actually used (if detected): {used_tool}\n"
            f"Internal error check: {reason}\n\n"
            f"Perception summary:\n{scene_brief}\n\n"
            "Please generate a user-facing explanation and a recovery plan "
            "following the format described in the system prompt."
        )

        payload = {"system": system_prompt, "user": user_prompt}
        explanation_text, _ = prompter.query(
            prompt=payload,
            sampling_params={"temperature": 0.2, "max_tokens": 220},
            save=False,
            save_dir=str(VIDEO_ROOT)
        )
        llm_used = True

    except Exception as e:
        print("⚠️ LLM error, falling back to template:", e)
        explanation_text = (
            f"I was supposed to {goal_en}, but something went wrong. "
            f"My internal check says: {reason}. "
            "I will switch to the correct tool and try again more carefully."
        )

else:
    explanation_text = (
        "No clear error was detected in this run. "
        "The tool I used matches the expected tool, so I did not generate a user-facing error explanation."
    )

explanation_text = explanation_text.strip()

# ========= 8. 保存 & 打印 =========

out_txt = VIDEO_ROOT / "vid_e_explanation.txt"
out_txt.write_text(explanation_text, encoding="utf-8")

print("\n=== Vid-E Explanation (LLM) ===")
print(explanation_text)
print("\nSaved explanation ->", out_txt)

# 记录一个简洁的 runlog（方便之后论文分析）
runlog_path = VIDEO_ROOT / "vid_e_runlog.json"
runlog = {
    "config": EXP_PATH.name,
    "video_root": str(VIDEO_ROOT),
    "task_id": task_id,
    "task_name": task_name,
    "goal_en": goal_en,
    "error_type": error_type,
    "expected_tool": expected_tool,
    "tool_candidates": tool_candidates,
    "used_tool": used_tool,
    "timeline_counts": timeline_counts,
    "graph_counts": graph_counts,
    "error_triggered": error_triggered,
    "reason": reason,
    "llm_used": llm_used,
    "explanation_text": explanation_text,
    "timestamp": time.time(),
}

runlog_path.write_text(json.dumps(runlog, indent=2), encoding="utf-8")

print("\n=== Vid-E Summary ===")
print("Config     :", EXP_PATH.name)
print("Video Root :", VIDEO_ROOT)
print("Error Type :", error_type)
print("Used Tool  :", used_tool)
print("Expected   :", expected_tool)
print("Triggered  :", int(bool(error_triggered)))
print("LLM Used   :", llm_used)
print("Runlog     :", runlog_path)
