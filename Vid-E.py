#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vid-E.py (NO-FALLBACK, self-repair)
- Reads: YAML config + video_root
- Checks: vid_c_runlog.json:error_confirmed
- If error confirmed -> generate short spoken explanation via OpenAI (no template fallback)
- Writes:
    video_root/explanation_text.txt
    <stable_root>/explanation_text.txt   (stable_root = workdir/task_id)
    video_root/vid_e_runlog.json
- Exit codes:
    0  => ERROR FOUND (explanation generated)
    10 => NO ERROR (no explanation)
    1  => unexpected failure (still writes something)
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
SPEAK_MODE = os.environ.get("VIDE_SPEAK_MODE", "speak").lower()

# disallow generic filler explanations (add your own here)
BANNED_PHRASES = [
    "did not work as expected",
    "didn't work as expected",
    "try a different approach",
    "different approach",
    "something went wrong",
    "continue with the next step",
    "wasn’t where i expected",
    "wasn't where i expected",
]

RC_ERROR_FOUND = 0
RC_NO_ERROR = 10
RC_FAIL = 1

# ------------------- Helpers -------------------

def log(*args):
    print(*args, flush=True)

def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

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
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def enforce_hard_cap(s: str) -> str:
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
        # allow controller/run_task to pass speak_phase explicitly
        "speak_phase": str(meta.get("speak_phase", "")),
    }

def _norm_phase(s: str) -> str:
    s = (s or "").strip().lower()
    return s if s in ("immediate", "post_recovery") else "immediate"

def build_system_prompt_non_observable(explanation_timing: str) -> str:
    explanation_timing = _norm_phase(explanation_timing)
    if explanation_timing == "immediate":
        return f"""
Generate a brief, natural spoken explanation for a live in-person robot study.

Error type: NON-OBSERVABLE (the user cannot directly see the cause).

Output EXACTLY THREE spoken sentences (total {MAX_WORDS_MIN}-{MAX_WORDS_MAX} words).

Sentence 1: what you tried to do (action + object) and that it didn’t work.
Sentence 2: ONE non-visible cause in everyday language (something the user could not see then).
Sentence 3: what you will do next to fix it (future tense, concrete everyday language).

Rules:
- Calm and helpful.
- Do NOT mention sensors, models, uncertainty, probabilities, or internal decision-making.
- Do NOT mention timing or error detection.
- Do NOT use generic filler like "something went wrong" or "different approach".
- Output ONLY the three sentences.
""".strip()

    return f"""
Generate a brief, natural spoken explanation for a live in-person robot study.

Error type: NON-OBSERVABLE (the user cannot directly see the cause).

Output EXACTLY THREE spoken sentences (total {MAX_WORDS_MIN}-{MAX_WORDS_MAX} words).

Sentence 1: quick recap that you tried (action + object) and it didn’t work at first (NO attention cue).
Sentence 2: ONE non-visible cause in everyday language.
Sentence 3: what you did to fix it (past tense, concrete everyday language).

Rules:
- Retrospective, not an interruption.
- Do NOT mention sensors, models, uncertainty, probabilities, or internal decision-making.
- Do NOT mention timing or error detection.
- Do NOT use generic filler like "something went wrong" or "different approach".
- Output ONLY the three sentences.
""".strip()

def build_system_prompt_observable(explanation_timing: str) -> str:
    explanation_timing = _norm_phase(explanation_timing)
    if explanation_timing == "immediate":
        return f"""
Generate a brief, natural spoken explanation for a live in-person robot study.

Error type: OBSERVABLE (the user can see the cause from the scene/behavior).

Output EXACTLY THREE spoken sentences (total {MAX_WORDS_MIN}-{MAX_WORDS_MAX} words).

Sentence 1: what you tried to do (action + object) and that it didn’t work.
Sentence 2: ONE visible cause using only what can be seen (no hidden inferences).
Sentence 3: what you will do next to fix it (future tense, concrete everyday language).

Rules:
- Calm and helpful.
- Do NOT mention sensors, models, uncertainty, probabilities, or internal decision-making.
- Do NOT mention timing or error detection.
- Do NOT use generic filler like "something went wrong" or "different approach".
- Output ONLY the three sentences.
""".strip()

    return f"""
Generate a brief, natural spoken explanation for a live in-person robot study.

Error type: OBSERVABLE (the user can see the cause from the scene/behavior).

Output EXACTLY THREE spoken sentences (total {MAX_WORDS_MIN}-{MAX_WORDS_MAX} words).

Sentence 1: quick recap that you tried (action + object) and it didn’t work at first (NO attention cue).
Sentence 2: ONE visible cause using only what can be seen (no hidden inferences).
Sentence 3: what you did to fix it (past tense, concrete everyday language).

Rules:
- Retrospective, not an interruption.
- Do NOT mention sensors, models, uncertainty, probabilities, or internal decision-making.
- Do NOT mention timing or error detection.
- Do NOT use generic filler like "something went wrong" or "different approach".
- Output ONLY the three sentences.
""".strip()

def build_user_prompt(meta: Dict[str, str], llm_cfg: Dict[str, Any]) -> str:
    goal = (llm_cfg.get("goal_en") or "").strip()
    scene = (llm_cfg.get("scene_description") or "").strip()
    ctx = (llm_cfg.get("task_context") or "").strip()

    action = (meta.get("action", "") or "").strip()
    obj = (meta.get("object", "") or "").strip()

    return f"""
Task goal: {goal}
Action: {action}
Target object: {obj}

Scene: {scene}

Context: {ctx}

Write the explanation now.
""".strip()

def build_repair_user_prompt(base_user_prompt: str, bad_text: str, reason: str) -> str:
    return f"""
{base_user_prompt}

Your last attempt was rejected: {reason}
Rejected text:
{bad_text}

Rewrite it to satisfy ALL constraints. Output ONLY the three sentences.
""".strip()

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

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = getattr(resp, "output_text", None)
        if not text and isinstance(resp, dict):
            text = resp.get("output_text")
        return text.strip() if text else None
    except Exception:
        return None

def generate_explanation(meta: Dict[str, str], cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (explanation_text, runlog_updates)
    NO fallback: must be produced by LLM or raise.
    """
    llm_cfg = cfg.get("llm", {}) or {}
    model = str(llm_cfg.get("model") or DEFAULT_MODEL)

    # Prefer explicit speak_phase; else fall back to explanation_timing
    speak_phase = _norm_phase(meta.get("speak_phase") or meta.get("explanation_timing") or "immediate")

    error_type = (meta.get("error_type") or "").strip().lower()
    sys_prompt = (
        build_system_prompt_non_observable(speak_phase)
        if ("non" in error_type)
        else build_system_prompt_observable(speak_phase)
    )

    base_user_prompt = build_user_prompt(meta, llm_cfg)

    runlog: Dict[str, Any] = {
        "llm_model": model,
        "words_min": MAX_WORDS_MIN,
        "words_max": MAX_WORDS_MAX,
        "sent_min": EXPLANATION_SENT_MIN,
        "sent_max": EXPLANATION_SENT_MAX,
        "speak_phase_used": speak_phase,
    }

    user_prompt = base_user_prompt
    last_text = ""
    last_reason = ""

    # 1st try normal, then self-repair with explicit rejection reason
    for attempt in range(1, 7):
        raw = try_openai_generate(sys_prompt, user_prompt, model=model)
        if not raw:
            runlog[f"attempt_{attempt}"] = "no_response"
            last_reason = "no_response"
            continue

        raw = enforce_hard_cap(normalize_whitespace(raw))
        ok, reason = valid_explanation(raw)
        runlog[f"attempt_{attempt}"] = reason

        if ok:
            runlog["selected_attempt"] = attempt
            return raw, runlog

        last_text = raw
        last_reason = reason
        user_prompt = build_repair_user_prompt(base_user_prompt, bad_text=raw, reason=reason)

    runlog["hard_fail"] = True
    runlog["last_reject_reason"] = last_reason
    runlog["last_reject_text"] = last_text
    raise RuntimeError(f"LLM failed to produce a valid explanation after retries: {last_reason}")

def find_stable_root(video_root: Path, task_id: str) -> Path:
    if not task_id:
        return video_root
    stable_root = video_root
    for p in video_root.parents:
        if p.name == task_id:
            stable_root = p
            break
    return stable_root

def write_outputs(video_root: Path, stable_root: Path, text: str):
    (video_root / "explanation_text.txt").write_text(text, encoding="utf-8")
    (stable_root / "explanation_text.txt").write_text(text, encoding="utf-8")

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
    # Keep printing speak_phase-like info, but generation uses meta.speak_phase preferred
    speak_phase = _norm_phase((meta.get("speak_phase") or explanation_timing))

    error_type = meta.get("error_type", "")

    stable_root = find_stable_root(video_root, task_id)

    # Read detector result
    runlog_c = read_json(video_root / "vid_c_runlog.json")
    error_confirmed = bool(runlog_c.get("error_confirmed", False))

    # Pretty header
    log("✅ Vid-E (ERROR-GATED, NO-FALLBACK)")
    log("   Task        :", task_name if task_name else task_id)
    log("   Timing      :", explanation_timing)
    log("   Speak phase :", speak_phase)
    log("   Error type  :", error_type)
    log("   Action      :", meta.get("action", ""))
    log("   Object      :", meta.get("object", ""))
    log("   error_conf  :", error_confirmed, "(vid_c_runlog.json:error_confirmed)")
    log("   Speak mode  :", SPEAK_MODE)

    # No error -> write marker, exit 10
    if not error_confirmed:
        explanation_text = "[NO EXPLANATION]"
        write_outputs(video_root, stable_root, explanation_text)

        log("\n=== Vid-E Output ===")
        log(explanation_text)
        sys.exit(RC_NO_ERROR)

    # Error confirmed -> generate explanation (no fallback)
    explanation_text = ""
    runlog_e: Dict[str, Any] = {}
    rc = RC_ERROR_FOUND

    try:
        explanation_text, runlog_e = generate_explanation(meta, cfg)
        explanation_text = explanation_text.strip()
        if not explanation_text:
            raise RuntimeError("LLM returned empty text")
        rc = RC_ERROR_FOUND
    except Exception as e:
        # Still write *something*, but not a template explanation
        explanation_text = "[LLM_EXPLANATION_FAILED]"
        runlog_e = {"exception": repr(e), "hard_fail": True}
        rc = RC_FAIL

    # Write outputs
    write_outputs(video_root, stable_root, explanation_text)

    # Save runlog
    runlog = {
        "task_id": task_id,
        "video_root": str(video_root),
        "stable_root": str(stable_root),
        "error_confirmed": error_confirmed,
        "explanation_timing": explanation_timing,
        "speak_phase": speak_phase,
        "error_type": error_type,
        **runlog_e,
    }
    (video_root / "vid_e_runlog.json").write_text(json.dumps(runlog, indent=2), encoding="utf-8")

    log("\n=== Vid-E Output ===")
    log(explanation_text)

    sys.exit(rc)

if __name__ == "__main__":
    main()
