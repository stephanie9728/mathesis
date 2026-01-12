#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vid-E (TEXT-FIRST, ERROR-GATED)
================================
- timing == none  -> never generate explanation
- timing != none  -> generate explanation ONLY IF error_confirmed == True
- Writes:
    vid_e_explanation.txt
    explanation_text.txt
    vid_e_runlog.json
- Controller owns playback timing.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml
import openai

DEFAULT_MODEL = "gpt-4o-mini"
MAX_WORDS = 25

openai.api_key = os.environ.get("OPENAI_API_KEY", "")
SPEAK_MODE = os.environ.get("VIDE_SPEAK_MODE", "never").lower()  # kept for compatibility


# ============================================================
# Prompts
# ============================================================

def build_system_prompt_non_observable(explanation_timing: str) -> str:
    tense_rule = (
        "Use future tense for the recovery action (e.g., 'I will adjust...')."
        if explanation_timing == "immediate"
        else
        "Use past tense for the recovery action (e.g., 'I adjusted...')."
    )
    return f"""
You are generating a VERY SHORT spoken robot explanation for a live in-person experiment.

Context awareness:
- You know the task goal, the scene, and the objects involved.
- You know what action you were trying to perform and why.
- An execution problem has already been confirmed by the system.

Error type: NON-OBSERVABLE.
The user CANNOT directly see the reason for the failure from the robot's motion alone.

Your explanation MUST:
- Briefly describe the situation and the failed action.
- Attribute the failure to a non-visible physical or interaction property.
- State the corrective behavior.

Output exactly TWO short sentences using this template:
"I couldn’t {{action}} the {{object}} because {{cause}}. I {{recovery_action}}."

Rules:
- The cause MUST describe a non-visible property (e.g., flatness, instability, slippage).
- Do NOT mention sensors, models, uncertainty, probabilities, or internal decision-making.
- Do NOT explain timing or error detection.
- Total length ≤ {MAX_WORDS} words.
- {tense_rule}
- Output ONLY the two sentences.
""".strip()


def build_system_prompt_observable(explanation_timing: str) -> str:
    tense_rule = (
        "Use future tense for the recovery action (e.g., 'I will go around...')."
        if explanation_timing == "immediate"
        else
        "Use past tense for the recovery action (e.g., 'I went around...')."
    )
    return f"""
You are generating a VERY SHORT spoken robot explanation for a live in-person experiment.

Context awareness:
- You know the task goal, the scene, and the visible objects.
- You know what action you were trying to perform.
- An execution problem has already been confirmed by the system.

Error type: OBSERVABLE.
The user CAN see the reason for the problem from the scene or the robot’s behavior.

Your explanation MUST:
- Briefly confirm the visible situation.
- State the corrective behavior without inferring hidden causes.

Output exactly TWO short sentences using this template:
"I couldn’t {{action}} the {{object}} because {{cause}}. I {{recovery_action}}."

Rules:
- The cause MUST describe a visible situation or event in the scene.
- Do NOT infer hidden properties or internal reasoning.
- Do NOT explain timing or error detection.
- Total length ≤ {MAX_WORDS} words.
- {tense_rule}
- Output ONLY the two sentences.
""".strip()


def build_user_prompt(task_name: str,
                      action: str,
                      object_name: str,
                      possible_causes: List[str],
                      recovery_actions: List[str]) -> str:
    return f"""
Task: {task_name}
Action verb: {action}
Object: {object_name}

Choose EXACTLY ONE cause from the list and copy it VERBATIM:
{", ".join(possible_causes)}

Choose EXACTLY ONE recovery action from the list and copy it VERBATIM:
{", ".join(recovery_actions)}

Return only the two-sentence template.
""".strip()


# ============================================================
# LLM
# ============================================================

def call_llm(system_prompt: str, user_prompt: str, model: str) -> str:
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=80,
    )
    return resp["choices"][0]["message"]["content"].strip()


# ============================================================
# Validation / postprocess / fallback
# ============================================================

def normalize_two_sentences(text: str) -> str:
    t = " ".join(text.split())
    t = t.replace("couldn't", "couldn’t")

    # keep only first two sentences
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p for p in parts if p]
    if len(parts) >= 2:
        t = parts[0].rstrip() + " " + parts[1].rstrip()
    elif len(parts) == 1:
        t = parts[0].rstrip()
        if " I " in t and not t.endswith("."):
            t += "."
        if t.count(".") == 0:
            t = t.rstrip(".") + ". I will try a different approach."
    else:
        t = ""

    if t and not re.search(r"[.!?]$", t):
        t += "."
    return t


def is_acceptable(text: str) -> bool:
    if not text:
        return False
    w = text.split()
    if len(w) > MAX_WORDS:
        return False
    if not text.lower().startswith("i couldn"):
        return False
    # quick check that we have two sentences-ish
    return text.count(".") >= 1


def fallback_explanation(action: str, object_name: str, timing: str) -> str:
    action = action or "perform"
    object_name = object_name or "object"
    if timing == "immediate":
        return f"I couldn’t {action} the {object_name} because it didn’t work as expected. I will try a different approach."
    else:
        return f"I couldn’t {action} the {object_name} because it didn’t work as expected. I tried a different approach."


def normalize_timing(explanation_timing: str) -> str:
    t = (explanation_timing or "immediate").lower()
    if t == "post":
        return "post_recovery"
    if t not in {"none", "immediate", "post_recovery"}:
        return "immediate"
    return t


def read_error_flag(video_root: Path) -> Dict[str, Any]:
    """
    Read error_flag.json produced by Vid-D.
    If missing, default to error_confirmed=True for non-window roots,
    but error_confirmed=False for window roots to avoid early explanations.
    """
    flag_path = video_root / "error_flag.json"
    if flag_path.exists():
        try:
            return json.loads(flag_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # default if missing
    is_window = "windows" in [p.name for p in video_root.parents] or (video_root.name.startswith("w") and video_root.parent.name == "windows")
    if is_window:
        return {"error_confirmed": False, "source": "Vid-E default(window)=False (missing error_flag.json)"}
    return {"error_confirmed": True, "source": "Vid-E default(full)=True (missing error_flag.json)"}


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python Vid-E.py <experiment.yaml> <VIDEO_ROOT>")
        sys.exit(1)

    yaml_path = Path(sys.argv[1]).resolve()
    video_root = Path(sys.argv[2]).resolve()

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    meta = cfg.get("meta", {}) or {}
    llm_cfg = cfg.get("llm", {}) or {}

    task_name = meta.get("task_name", meta.get("task_id", "unknown_task"))
    error_type = (meta.get("error_type") or "").lower()
    explanation_timing = normalize_timing(meta.get("explanation_timing"))

    action = meta.get("action", "perform")
    object_name = meta.get("object", "object")

    possible_causes = llm_cfg.get("possible_causes", ["it did not work as expected"])
    recovery_actions = llm_cfg.get("recovery_actions", ["try a different approach"])
    model_name = llm_cfg.get("model", DEFAULT_MODEL)

    # --------- NEW: error gating ----------
    flag = read_error_flag(video_root)
    error_confirmed = bool(flag.get("error_confirmed", False))
    flag_source = flag.get("source", "unknown")

    print("✅ Vid-E (ERROR-GATED)")
    print("   Task        :", task_name)
    print("   Timing      :", explanation_timing)
    print("   Error type  :", error_type)
    print("   Action      :", action)
    print("   Object      :", object_name)
    print("   error_conf  :", error_confirmed, f"({flag_source})")
    print("   Speak mode  :", SPEAK_MODE)

    # timing==none MUST be preserved
    if explanation_timing == "none":
        explanation_text = ""
        llm_used = False
        reason = "timing==none -> no explanation"
        _write_outputs(video_root, yaml_path, task_name, error_type, explanation_timing, action, object_name,
                       error_confirmed, flag_source, llm_used, reason, explanation_text)
        print("\n=== Vid-E Output ===")
        print("[NO EXPLANATION]")
        raise SystemExit(10)

    # error gate: if not confirmed, do not explain
    if not error_confirmed:
        explanation_text = ""
        llm_used = False
        reason = "error_confirmed==False -> no explanation"
        _write_outputs(video_root, yaml_path, task_name, error_type, explanation_timing, action, object_name,
                       error_confirmed, flag_source, llm_used, reason, explanation_text)
        print("\n=== Vid-E Output ===")
        print("[NO EXPLANATION]")
        raise SystemExit(10)

    # else: error_confirmed -> generate explanation
    system_prompt = (
        build_system_prompt_non_observable(explanation_timing)
        if error_type in {"non_observable", "non-observable", "nonobservable"}
        else build_system_prompt_observable(explanation_timing)
    )
    user_prompt = build_user_prompt(
        task_name=task_name,
        action=action,
        object_name=object_name,
        possible_causes=possible_causes,
        recovery_actions=recovery_actions,
    )

    tts_timing = {}
    llm_used = False
    reason = ""
    explanation_text = ""

    try:
        raw = call_llm(system_prompt, user_prompt, model_name)
        raw = normalize_two_sentences(raw)
        raw = " ".join(raw.split()[:MAX_WORDS])

        if is_acceptable(raw):
            explanation_text = raw
            llm_used = True
            reason = "LLM output accepted."
        else:
            explanation_text = fallback_explanation(action, object_name, explanation_timing)
            llm_used = False
            reason = "LLM output invalid -> fallback used."

    except Exception as e:
        explanation_text = fallback_explanation(action, object_name, explanation_timing)
        llm_used = False
        reason = f"LLM failed: {e} -> fallback used."

    _write_outputs(video_root, yaml_path, task_name, error_type, explanation_timing, action, object_name,
                   error_confirmed, flag_source, llm_used, reason, explanation_text, tts_timing)

    print("\n=== Vid-E Output ===")
    print(explanation_text if explanation_text else "[NO EXPLANATION]")
    raise SystemExit(0 if explanation_text else 10)


def _write_outputs(
    video_root: Path,
    yaml_path: Path,
    task_name: str,
    error_type: str,
    explanation_timing: str,
    action: str,
    object_name: str,
    error_confirmed: bool,
    flag_source: str,
    llm_used: bool,
    reason: str,
    explanation_text: str,
    tts_timing: Optional[Dict[str, Any]] = None,
):
    out_txt = video_root / "vid_e_explanation.txt"
    stable_txt = video_root / "explanation_text.txt"
    out_txt.write_text(explanation_text, encoding="utf-8")
    stable_txt.write_text(explanation_text, encoding="utf-8")

    runlog = {
        "config": str(yaml_path),
        "task_name": task_name,
        "error_type": error_type,
        "explanation_timing": explanation_timing,
        "action": action,
        "object": object_name,
        "error_confirmed": error_confirmed,
        "error_flag_source": flag_source,
        "llm_used": llm_used,
        "reason": reason,
        "explanation": explanation_text,
        "tts": tts_timing or {},
        "speak_mode": SPEAK_MODE,
        "ts": time.time(),
    }
    (video_root / "vid_e_runlog.json").write_text(json.dumps(runlog, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
