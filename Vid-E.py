#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vid-E.py (robust)
- Reads: YAML config + video_root
- Checks: vid_c_runlog.json:error_confirmed
- If error confirmed -> generate short spoken explanation via OpenAI (or fallback template)
- Writes:
    video_root/explanation_text.txt
    <stable_root>/explanation_text.txt   (stable_root = workdir/task_id)
- Exit codes:
    0  => ERROR FOUND (explanation generated)
    10 => NO ERROR (no explanation)
    1  => unexpected failure (still tries to write something)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import yaml

# ------------------- Tunables -------------------
DEFAULT_MODEL = os.environ.get("VIDE_MODEL", "gpt-4o-mini")

# word constraints for spoken explanation
MAX_WORDS_MIN = int(os.environ.get("VIDE_WORDS_MIN", "50"))
MAX_WORDS_MAX = int(os.environ.get("VIDE_WORDS_MAX", "80"))
MAX_WORDS = MAX_WORDS_MAX  # hard cap for truncation

# sentence constraints
EXPLANATION_SENT_MIN = int(os.environ.get("VIDE_SENT_MIN", "3"))
EXPLANATION_SENT_MAX = int(os.environ.get("VIDE_SENT_MAX", "3"))  # default exactly 3

# kept for compatibility with older pipelines
SPEAK_MODE = os.environ.get("VIDE_SPEAK_MODE", "never").lower()

# disallow generic filler explanations (add your own here)
BANNED_PHRASES = [
    "did not work as expected",
    "didn't work as expected",
    "try a different approach",
    "different approach",
    "something went wrong",

    # kill the old robotic template
    "continue with the next step",
    "wasn’t where i expected",
    "wasn't where i expected",
]

RC_ERROR_FOUND = 0
RC_NO_ERROR = 10

# ------------------- Helpers -------------------

def log(*args):
    print(*args, flush=True)

def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def count_words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))

def count_sentences(s: str) -> int:
    # Good enough for spoken TTS: split on . ! ?
    parts = [p.strip() for p in re.split(r"[.!?]+", s.strip()) if p.strip()]
    return len(parts)

def has_banned(s: str) -> bool:
    t = s.lower()
    return any(p in t for p in BANNED_PHRASES)

def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # keep sentence boundaries, but normalize internal whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def enforce_hard_cap(s: str) -> str:
    # Hard truncate words to avoid runaway output
    words = s.split()
    if len(words) > MAX_WORDS:
        s = " ".join(words[:MAX_WORDS]).strip()
    return s

def valid_explanation(s: str) -> Tuple[bool, str]:
    s = normalize_whitespace(s)
    w = count_words(s)
    n = count_sentences(s)

    if has_banned(s):
        return False, "banned_phrase"
    if not (EXPLANATION_SENT_MIN <= n <= EXPLANATION_SENT_MAX):
        return False, f"bad_sentence_count(n={n})"
    if not (MAX_WORDS_MIN <= w <= MAX_WORDS_MAX):
        return False, f"bad_word_count(w={w})"
    return True, "ok"

def extract_task_meta(cfg: Dict[str, Any]) -> Dict[str, str]:
    meta = cfg.get("meta", {}) or {}
    return {
        "task_id": str(meta.get("task_id", "")),
        "task_name": str(meta.get("task_name", "")),
        "error_type": str(meta.get("error_type", "")),  # observable / non_observable
        "action": str(meta.get("action", "")),
        "object": str(meta.get("object", "")),
        "explanation_timing": str(meta.get("explanation_timing", "immediate")),
        "participant_id": str(meta.get("participant_id", "")),
    }

def build_system_prompt_non_observable(explanation_timing: str) -> str:
    # NOTE: immediate => attention cue allowed/required; post_recovery => NO attention cue
    if explanation_timing == "immediate":
        return f"""
Generate a brief, natural spoken explanation for a live in-person robot study.

Error type: NON-OBSERVABLE.
The user cannot directly see the cause from the robot’s motion or the scene.

Output exactly THREE short spoken sentences (total {MAX_WORDS_MIN}-{MAX_WORDS_MAX} words).

Sentence 1:
- Start with a very short attention cue (1–3 words), e.g., "One moment," / "Just a second,".
- State what you were trying to do (action + object) and that it didn’t work.

Sentence 2:
- Give ONE non-visible cause in everyday language (a hidden property or a non-visible outcome).
- The cause must be something the user could not directly see at that moment.

Sentence 3:
- Clearly state what you will do next to fix it, in concrete everyday language.
- Use future tense in Sentence 3 (e.g., "I will ...").

Rules:
- Be calm and helpful.
- Do NOT mention sensors, models, uncertainty, probabilities, or internal decision-making.
- Do NOT mention timing or error detection.
- Do NOT use generic filler like "something went wrong" or "next step".
- Output ONLY the three sentences.
""".strip()

    # post_recovery (or anything else) => retrospective, no attention cue
    return f"""
Generate a brief, natural spoken explanation for a live in-person robot study.

Error type: NON-OBSERVABLE.
The user cannot directly see the cause from the robot’s motion or the scene.

Output exactly THREE short spoken sentences (total {MAX_WORDS_MIN}-{MAX_WORDS_MAX} words).

Sentence 1:
- Do NOT use an attention cue.
- Briefly state what you were trying to do (action + object) and that it didn’t work at first.

Sentence 2:
- Give ONE non-visible cause in everyday language (a hidden property or a non-visible outcome).
- The cause must be something the user could not directly see at that moment.

Sentence 3:
- Briefly state what you did to fix it, in concrete everyday language.
- Use past tense in Sentence 3 (e.g., "I adjusted ...", "I tried again ...", "I checked ...").

Rules:
- Speak as a short retrospective explanation (a quick recap), not an interruption.
- Do NOT mention sensors, models, uncertainty, probabilities, or internal decision-making.
- Do NOT mention timing or error detection.
- Do NOT use generic filler like "something went wrong" or "next step".
- Output ONLY the three sentences.
""".strip()


def build_system_prompt_observable(explanation_timing: str) -> str:
    # NOTE: immediate => attention cue allowed/required; post_recovery => NO attention cue
    if explanation_timing == "immediate":
        return f"""
Generate a brief, natural spoken explanation for a live in-person robot study.

Error type: OBSERVABLE.
The user can see the cause from the scene or the robot’s behavior.

Output exactly THREE short spoken sentences (total {MAX_WORDS_MIN}-{MAX_WORDS_MAX} words).

Sentence 1:
- Start with a very short attention cue (1–3 words), e.g., "One moment," / "Just a second,".
- State what you were trying to do (action + object) and that it didn’t work.

Sentence 2:
- Describe ONE visible cause using only what can be seen (no hidden inferences).

Sentence 3:
- Clearly state what you will do next to fix it, in concrete everyday language.
- Use future tense in Sentence 3 (e.g., "I will ...").

Rules:
- Be calm and helpful.
- Do NOT mention sensors, models, uncertainty, probabilities, or internal decision-making.
- Do NOT mention timing or error detection.
- Do NOT use generic filler like "something went wrong" or "next step".
- Output ONLY the three sentences.
""".strip()

    # post_recovery (or anything else) => retrospective, no attention cue
    return f"""
Generate a brief, natural spoken explanation for a live in-person robot study.

Error type: OBSERVABLE.
The user can see the cause from the scene or the robot’s behavior.

Output exactly THREE short spoken sentences (total {MAX_WORDS_MIN}-{MAX_WORDS_MAX} words).

Sentence 1:
- Do NOT use an attention cue.
- Briefly state what you were trying to do (action + object) and that it didn’t work at first.

Sentence 2:
- Describe ONE visible cause using only what can be seen (no hidden inferences).

Sentence 3:
- Briefly state what you did to fix it, in concrete everyday language.
- Use past tense in Sentence 3 (e.g., "I adjusted ...", "I moved ...", "I tried again ...").

Rules:
- Speak as a short retrospective explanation (a quick recap), not an interruption.
- Do NOT mention sensors, models, uncertainty, probabilities, or internal decision-making.
- Do NOT mention timing or error detection.
- Do NOT use generic filler like "something went wrong" or "next step".
- Output ONLY the three sentences.
""".strip()


def build_user_prompt(meta: Dict[str, str], llm_cfg: Dict[str, Any]) -> str:
    # We keep user-side context grounded in YAML, not in guesses.
    goal = (llm_cfg.get("goal_en") or "").strip()
    scene = (llm_cfg.get("scene_description") or "").strip()
    ctx = (llm_cfg.get("task_context") or "").strip()

    # action/object come from meta
    action = meta.get("action", "").strip()
    obj = meta.get("object", "").strip()

    # For your apple task: encourage mentioning visible items if provided in scene_description.
    return f"""
Task goal: {goal}
Action: {action}
Target object: {obj}

Scene: {scene}

Context: {ctx}

Write the explanation now.
""".strip()

def fallback_explanation(meta: Dict[str, str]) -> str:
    """
    Safe fallback if OpenAI is unavailable or output fails validation.
    Keeps it human-friendly and consistent with observability + tense.
    """
    timing = meta.get("explanation_timing", "immediate")
    error_type = meta.get("error_type", "")
    action = meta.get("action", "get")
    obj = meta.get("object", "object")

    if timing == "immediate":
        fix = "I’ll adjust my view and try again."
    else:
        fix = "I adjusted my view and tried again."

    if "non" in error_type:
        # non-observable: hidden outcome/property
        return f"One moment— I couldn’t {action} the {obj}. It wasn’t where I needed it to be. {fix}"
    else:
        # observable: visible situation
        return f"One moment— I couldn’t {action} the {obj}. I can’t see it clearly from this angle. {fix}"

def try_openai_generate(system_prompt: str, user_prompt: str, model: str) -> Optional[str]:
    """
    Uses OpenAI Python SDK (Responses API).
    Returns output_text or None.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        return None

    # As per OpenAI docs, use client.responses.create and read response.output_text :contentReference[oaicite:1]{index=1}
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = getattr(resp, "output_text", None)
        if not text:
            # sometimes SDK returns dict-like
            text = resp.get("output_text") if isinstance(resp, dict) else None
        return text.strip() if text else None
    except Exception:
        return None

def generate_explanation(meta: Dict[str, str], cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (explanation_text, runlog_updates)
    """
    runlog: Dict[str, Any] = {}
    timing = meta.get("explanation_timing", "immediate")
    error_type = (meta.get("error_type") or "").lower()

    llm_cfg = cfg.get("llm", {}) or {}
    model = str(llm_cfg.get("model") or DEFAULT_MODEL)

    if "non" in error_type:
        sys_prompt = build_system_prompt_non_observable(speak_phase)
    else:
        sys_prompt = build_system_prompt_observable(speak_phase)


    user_prompt = build_user_prompt(meta, llm_cfg)

    runlog["llm_model"] = model
    runlog["words_min"] = MAX_WORDS_MIN
    runlog["words_max"] = MAX_WORDS_MAX
    runlog["sent_min"] = EXPLANATION_SENT_MIN
    runlog["sent_max"] = EXPLANATION_SENT_MAX

    # Try up to 4 attempts; accept only valid outputs
    for attempt in range(1, 5):
        raw = try_openai_generate(sys_prompt, user_prompt, model=model)
        if not raw:
            runlog[f"attempt_{attempt}"] = "no_response"
            continue

        raw = normalize_whitespace(raw)
        raw = enforce_hard_cap(raw)

        ok, reason = valid_explanation(raw)
        runlog[f"attempt_{attempt}"] = reason
        if ok:
            runlog["selected_attempt"] = attempt
            return raw, runlog

    # fallback
    fb = fallback_explanation(meta)
    fb = enforce_hard_cap(normalize_whitespace(fb))

    # If fallback violates strict constraints (rare), lightly adjust (don’t crash)
    runlog["fallback_used"] = True
    return fb, runlog

# ------------------- Main -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_path", help="tasks/_tmp_run.yaml or experiment YAML")
    ap.add_argument("video_root", help="path to video root or window root")
    args = ap.parse_args()

    yaml_path = Path(args.yaml_path).resolve()
    video_root = Path(args.video_root).resolve()

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    meta = extract_task_meta(cfg)

    task_id = meta.get("task_id", "")
    task_name = meta.get("task_name", "")
    explanation_timing = meta.get("explanation_timing", "immediate")
    speak_phase = cfg["meta"].get("speak_phase", explanation_timing)

    error_type = meta.get("error_type", "")

    # Read detector result
    runlog_c = read_json(video_root / "vid_c_runlog.json")
    error_confirmed = bool(runlog_c.get("error_confirmed", False))

    # Pretty header
    log("✅ Vid-E (ERROR-GATED)")
    log("   Task        :", task_name if task_name else task_id)
    log("   Timing      :", explanation_timing)
    log("   Error type  :", error_type)
    log("   Action      :", meta.get("action", ""))
    log("   Object      :", meta.get("object", ""))
    log("   error_conf  :", error_confirmed, "(vid_c_runlog.json:error_confirmed)")
    log("   Speak mode  :", SPEAK_MODE)

    explanation_text = ""

    if not error_confirmed:
        explanation_text = "[NO EXPLANATION]"
        # write stable + window output
        out_txt = video_root / "explanation_text.txt"
        out_txt.write_text(explanation_text, encoding="utf-8")

        if task_id:
            stable_root = video_root
            # If video_root is .../task/windows/w000090, stable root should be .../task
            # We detect by searching upward for folder named task_id.
            for p in video_root.parents:
                if p.name == task_id:
                    stable_root = p
                    break
            (stable_root / "explanation_text.txt").write_text(explanation_text, encoding="utf-8")

        log("\n=== Vid-E Output ===")
        log(explanation_text)
        sys.exit(RC_NO_ERROR)

    # Error confirmed -> generate explanation
    explanation_text, runlog_e = generate_explanation(meta, cfg)

    # Final sanitize: ensure no empty
    explanation_text = explanation_text.strip() if explanation_text else ""
    if not explanation_text:
        explanation_text = fallback_explanation(meta)

    # Write outputs
    out_txt = video_root / "explanation_text.txt"
    out_txt.write_text(explanation_text, encoding="utf-8")

    stable_root = video_root
    if task_id:
        for p in video_root.parents:
            if p.name == task_id:
                stable_root = p
                break
    (stable_root / "explanation_text.txt").write_text(explanation_text, encoding="utf-8")

    # Save runlog
    runlog = {
        "task_id": task_id,
        "video_root": str(video_root),
        "error_confirmed": error_confirmed,
        "explanation_timing": explanation_timing,
        "error_type": error_type,
        **runlog_e,
    }
    (video_root / "vid_e_runlog.json").write_text(json.dumps(runlog, indent=2), encoding="utf-8")

    log("\n=== Vid-E Output ===")
    log(explanation_text)

    sys.exit(RC_ERROR_FOUND)

if __name__ == "__main__":
    main()
