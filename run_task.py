#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import subprocess
from pathlib import Path
import shutil
import argparse
import yaml

py = sys.executable

RC_ERROR_FOUND = 0
RC_NO_ERROR = 10


def copy_window_frames(src_frames: Path, dst_frames: Path, window_end: int, window_size: int) -> int:
    """
    window_end: 1-based frame index
    frames on disk: 0-based filenames 000000.png ...
    returns number of copied frames
    """
    dst_frames.mkdir(parents=True, exist_ok=True)
    start = max(1, window_end - window_size + 1)

    copied = 0
    for i in range(start - 1, window_end):
        src = src_frames / f"{i:06d}.png"
        if src.exists():
            shutil.copy2(src, dst_frames / src.name)
            copied += 1
    return copied


def run_stage(cmd, cwd: Path, must_succeed: bool = True) -> int:
    p = subprocess.run(cmd, cwd=str(cwd), check=False)
    if must_succeed and p.returncode != 0:
        raise SystemExit(p.returncode)
    return p.returncode


def count_frames(frames_dir: Path) -> int:
    return len(list(frames_dir.glob("*.png")))


def safe_send_say(say, text: str, task_id: str, phase: str) -> None:
    try:
        say.send_say(text, task_id, phase)
    except Exception as e:
        print(f"⚠️ send_say failed (ignored): {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_id")
    ap.add_argument("timing")       # none / immediate / post_recovery
    ap.add_argument("participant")
    ap.add_argument("--speak_phase", default="", help="immediate or post_recovery (controls prompt style)")

    ap.add_argument("--mode", default="full", choices=["full", "window"])
    ap.add_argument("--window_end", type=int, default=0, help="1-based frame index (e.g., 90 means up to 000089.png)")
    ap.add_argument("--window_size", type=int, default=30)

    # optional marker
    ap.add_argument("--final_window", action="store_true",
                    help="treat this window as the final fallback window")

    # remote audio config (PC1 endpoint)
    ap.add_argument("--pc1_audio", default="tcp://10.163.18.91:5557",
                    help="PC1 audio listener endpoint (PULL on PC1)")

    args = ap.parse_args()

    task_id = args.task_id
    timing = args.timing
    participant = args.participant
    def already_spoken(video_root: Path, phase: str) -> bool:
        return (video_root / f".spoken_{phase}.flag").exists()

    def mark_spoken(video_root: Path, phase: str) -> None:
        (video_root / f".spoken_{phase}.flag").write_text("1", encoding="utf-8")


    # ---------------- Audio: PC2 only sends SAY to PC1 ----------------
    say = None
    if timing != "none":
        try:
            from audio_send_say import SayClient
            say = SayClient(args.pc1_audio)
        except Exception as e:
            # audio is non-critical; keep pipeline running
            print(f"⚠️ Audio disabled (SayClient init failed): {e}", flush=True)
            say = None

    ROOT = Path(__file__).parent.resolve()
    TASKS_DIR = ROOT / "tasks"

    # ---------- 1) experiment YAML ----------
    yaml_path = TASKS_DIR / f"experiment_{task_id}.yaml"
    if not yaml_path.exists():
        raise ValueError(f"Experiment YAML not found: {yaml_path}")

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    # ---------- 2) override meta ----------
    cfg.setdefault("meta", {})
    cfg["meta"]["task_id"] = task_id
    cfg["meta"]["explanation_timing"] = timing
    cfg["meta"]["participant_id"] = participant
    cfg["meta"]["final_window"] = bool(args.final_window)
    if args.speak_phase:
        cfg["meta"]["speak_phase"] = args.speak_phase


    yaml_tmp = TASKS_DIR / "_tmp_run.yaml"
    yaml_tmp.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")

    # ---------- 3) roots ----------
    base_video_root = ROOT / task_id
    base_frames = base_video_root / "frames"
    if not base_frames.exists():
        raise RuntimeError(f"No frames found in {base_frames}")

    total_frames = count_frames(base_frames)
    if total_frames == 0:
        raise RuntimeError(f"No .png frames in {base_frames}")

    # WINDOW MODE
    if args.mode == "window":
        if args.window_end <= 0:
            raise SystemExit("window mode requires --window_end > 0")

        # Clamp window_end so we never create empty windows
        if args.window_end > total_frames:
            print(f"⚠️ window_end={args.window_end} > total_frames={total_frames}, clamp to {total_frames}")
            args.window_end = total_frames

        window_root = base_video_root / "windows" / f"w{args.window_end:06d}"
        window_frames = window_root / "frames"

        if window_root.exists():
            shutil.rmtree(window_root)
        window_root.mkdir(parents=True, exist_ok=True)

        copied = copy_window_frames(base_frames, window_frames, args.window_end, args.window_size)
        video_root = window_root
    else:
        copied = total_frames
        video_root = base_video_root

    print(f"✅ Task        : {task_id}")
    print(f"✅ Timing      : {timing}")
    print(f"✅ Participant : {participant}")
    print(f"✅ Video Root  : {video_root}")
    if args.mode == "window":
        print(f"🪟 Window      : end={args.window_end}, size={args.window_size}, copied={copied}, final={bool(args.final_window)}")

        if copied == 0:
            print("⚠️ Empty window after copy -> return NO_ERROR (10)")
            raise SystemExit(RC_NO_ERROR)

    # ---------- 4) Vid-B → Vid-E ----------
    run_stage([py, "Vid-B.py", str(video_root)], cwd=ROOT, must_succeed=True)

    # detector-only Vid-C reads YAML for detector config
    run_stage([py, "Vid-C.py", str(yaml_tmp), str(video_root)], cwd=ROOT, must_succeed=True)

    run_stage([py, "Vid-D.py", str(yaml_tmp), str(video_root)], cwd=ROOT, must_succeed=True)

    # Vid-E: rc=10 is NORMAL (no error detected)
    rc_e = run_stage([py, "Vid-E.py", str(yaml_tmp), str(video_root)], cwd=ROOT, must_succeed=False)

    # ---------- 5) When to speak ----------
    # Convention used here:
    # - immediate: speak when error is found (this run)
    # - post_recovery: speak ONLY when this run is flagged as final_window
    def maybe_speak(txt: str):
        if not say:
            return
        if not txt or txt.strip() == "" or txt.strip() == "[NO EXPLANATION]":
            return

        # speak-once gate
        phase = "immediate" if timing == "immediate" else "post_recovery"
        if already_spoken(base_video_root, phase):
            return

        if timing == "immediate":
            safe_send_say(say, txt, task_id, "immediate")
            mark_spoken(base_video_root, "immediate")
        elif timing == "post_recovery" and bool(args.final_window):
            safe_send_say(say, txt, task_id, "post_recovery")
            mark_spoken(base_video_root, "post_recovery")


    if rc_e == RC_ERROR_FOUND:
        txt_path = video_root / "explanation_text.txt"
        txt = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""

        if txt and txt != "[NO EXPLANATION]":
            print(txt)

        # Speak according to timing policy
        maybe_speak(txt)

        raise SystemExit(RC_ERROR_FOUND)

    if rc_e == RC_NO_ERROR:
        raise SystemExit(RC_NO_ERROR)

    raise SystemExit(rc_e)


if __name__ == "__main__":
    main()
