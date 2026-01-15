#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
controller_record_and_run.py (full, fixed)

What this controller does (PC2):
- ZMQ PULL: receives START / SEG_DONE / END / HELLO_ACK / AUDIO_DONE
- TCP server: receives length-prefixed PNG frames from PC1 and saves to <workdir>/<task_id>/frames
- Window analysis: periodically calls run_task.py in --mode window and latches explanation_text (if any)
- Optional remote audio: sends SAY to PC1 audio listener via ZMQ PUSH
- Optional orchestration blocking: can delay PLAY_ACK until AUDIO_DONE (block_on_audio)

Fixes vs your pasted version:
- run_window_analyze() call signature is correct (workdir, task_id, timing, participant, frames_dir, window_end, window_size, analyze_timeout_sec)
- Uses current_task_id consistently (no undefined task_id in maybe_analyze_and_latch)
- Writes frames as 000000.png, 000001.png, ...
- Adds external_flag helper: on SEG_DONE A for a configured task, writes robot_error.json in task root
  so Vid-C can use detector=external_flag.

Usage example:
  export VIDE_SPEAK_MODE=speak
  python3 controller_record_and_run.py --workdir /home/ru53kem/Projects/mathesis --zmq_port 5555 --tcp_port 5001 \
    --pc1_audio_endpoint tcp://10.163.18.91:5557 --audio_ack_fallback_sec 12
"""

import argparse
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Set, List

import yaml
import zmq


# ----------------------------- logging -----------------------------

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------- protocol parsing -----------------------------

def parse_start(msg: str, default_participant: str) -> Optional[Tuple[str, str, str]]:
    # START <task_id> <participant> <timing>
    parts = msg.split()
    if len(parts) < 4 or parts[0] != "START":
        return None
    task_id = parts[1]
    participant = parts[2] or default_participant
    timing = parts[3]
    return task_id, participant, timing


def parse_hello_ack(msg: str) -> Optional[Tuple[str, str]]:
    # HELLO_ACK <task_id> <endpoint>
    parts = msg.split(maxsplit=2)
    if len(parts) == 3 and parts[0] == "HELLO_ACK":
        return parts[1], parts[2]
    return None


def parse_seg_done(msg: str) -> Optional[Tuple[str, str]]:
    # SEG_DONE <task_id> <A|B>
    parts = msg.split()
    if len(parts) >= 3 and parts[0] == "SEG_DONE":
        return parts[1], parts[2]
    return None


def parse_end(msg: str) -> Optional[str]:
    # END <task_id>
    parts = msg.split()
    if len(parts) >= 2 and parts[0] == "END":
        return parts[1]
    return None


def parse_audio_done(msg: str) -> Optional[Tuple[str, str]]:
    # AUDIO_DONE <task_id> <phase>
    parts = msg.split()
    if len(parts) >= 3 and parts[0] == "AUDIO_DONE":
        return parts[1], parts[2]
    return None


# ----------------------------- TCP receiver (length-prefixed PNG) -----------------------------

class TCPReceiver:
    """
    Accepts a single TCP connection from PC1 and receives frames.
    Expected payload per frame:
      [4-byte big-endian length][png_bytes]
    """
    def __init__(self, host: str, port: int, accept_timeout: float = 0.2):
        self.host = host
        self.port = port
        self.accept_timeout = accept_timeout

        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((self.host, self.port))
        self.srv.listen(1)
        self.srv.settimeout(self.accept_timeout)

        self.conn: Optional[socket.socket] = None
        self.conn_addr = None

    def accept_if_needed(self):
        if self.conn is not None:
            return
        try:
            c, addr = self.srv.accept()
        except socket.timeout:
            return
        except Exception as e:
            log(f"⚠️ TCP accept error: {e}")
            return
        self.conn = c
        self.conn_addr = addr
        self.conn.settimeout(0.2)
        log(f"🔗 TCP connected from {addr}")

    def close_conn(self):
        if self.conn is None:
            return
        try:
            self.conn.close()
        except Exception:
            pass
        self.conn = None
        self.conn_addr = None

    def _recv_exact(self, n: int) -> Optional[bytes]:
        buf = b""
        while len(buf) < n:
            try:
                chunk = self.conn.recv(n - len(buf))
            except socket.timeout:
                return None
            except Exception:
                return None
            if not chunk:
                return b""  # closed
            buf += chunk
        return buf

    def recv_one_png(self) -> Optional[bytes]:
        """
        Returns:
          - bytes: PNG payload
          - None: no data yet (timeout)
        Side effects:
          - closes conn on socket closed
        """
        if self.conn is None:
            return None

        hdr = self._recv_exact(4)
        if hdr is None:
            return None
        if hdr == b"":
            log("⚠️ TCP recv: socket closed (closing conn)")
            self.close_conn()
            return None

        (length,) = struct.unpack(">I", hdr)
        if length <= 0 or length > 50_000_000:
            log(f"⚠️ Bad frame length={length} (closing conn)")
            self.close_conn()
            return None

        payload = self._recv_exact(length)
        if payload is None:
            return None
        if payload == b"":
            log("⚠️ TCP recv: socket closed mid-frame (closing conn)")
            self.close_conn()
            return None

        return payload


# ----------------------------- helpers -----------------------------

def clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def read_explanation_text(win_root: Path) -> str:
    txt_path = win_root / "explanation_text.txt"
    if not txt_path.exists():
        return ""
    txt = txt_path.read_text(encoding="utf-8", errors="replace").strip()
    if not txt or txt == "[NO EXPLANATION]":
        return ""
    return txt


def run_window_analyze(
    workdir: Path,
    task_id: str,
    timing: str,
    participant: str,
    frames_dir: Path,
    window_end: int,
    window_size: int,
    analyze_timeout_sec: float,
) -> str:
    """
    Create a window dir <workdir>/<task_id>/windows/wXXXXXX, copy last window_size frames,
    run run_task.py in window mode. If error latched (rc==0), return explanation_text.
    """
    win_root = (workdir / task_id / "windows" / f"w{window_end:06d}")
    win_frames = win_root / "frames"

    if win_root.exists():
        shutil.rmtree(win_root)
    win_frames.mkdir(parents=True, exist_ok=True)

    start = max(0, window_end - window_size)
    copied = 0
    for i in range(start, window_end):
        src = frames_dir / f"{i:06d}.png"
        if src.exists():
            shutil.copy2(src, win_frames / src.name)
            copied += 1

    if copied == 0:
        return ""

    cmd = [
        "python3", "run_task.py",
        task_id, timing, participant,
        "--speak_phase", timing,      # immediate / post_recovery
        "--mode", "window",
        "--window_end", str(window_end),
        "--window_size", str(window_size),
    ]
    log(f"🧠 WINDOW ANALYZE: {' '.join(cmd)}")

    try:
        p = subprocess.run(cmd, cwd=str(workdir), timeout=float(analyze_timeout_sec), check=False)
    except subprocess.TimeoutExpired:
        log(f"⏰ WINDOW ANALYZE timeout at window_end={window_end}")
        return ""

    # Convention: rc=0 => error found (writes explanation_text.txt); rc=10 => normal/no error
    if p.returncode == 0:
        txt = read_explanation_text(win_root)
        if txt:
            log(f"🚨 ERROR LATCHED at frame={window_end} (len={len(txt)})")
            return txt
        return ""

    if p.returncode == 10:
        log(f"✅ No error latched at window_end={window_end}")
        return ""

    log(f"⚠️ WINDOW ANALYZE rc={p.returncode} at window_end={window_end}")
    return ""


def write_robot_error_flag(video_root: Path, task_id: str, error_confirmed: bool, reason: str = ""):
    p = video_root / "robot_error.json"
    data = {
        "task_id": task_id,
        "error_confirmed": bool(error_confirmed),
        "reason": reason,
        "ts": time.time(),
    }
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"🧾 Wrote robot_error.json: error_confirmed={error_confirmed} reason={reason}")


def clear_robot_error_flag(video_root: Path):
    p = video_root / "robot_error.json"
    if p.exists():
        try:
            p.unlink()
            log("🧹 Cleared robot_error.json")
        except Exception as e:
            log(f"⚠️ Failed to clear robot_error.json: {e}")


# ----------------------------- main controller -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--zmq_port", type=int, default=5555)
    ap.add_argument("--tcp_port", type=int, default=5001)
    ap.add_argument("--participant", default="P01")

    ap.add_argument("--window_size", type=int, default=30)
    ap.add_argument("--window_stride", type=int, default=30)
    ap.add_argument("--window_warmup", type=int, default=60)
    ap.add_argument("--analyze_timeout_sec", type=float, default=30.0)

    ap.add_argument("--pc1_audio_endpoint", default="", help="tcp://PC1:5557 (optional)")
    ap.add_argument("--audio_ack_fallback_sec", type=float, default=12.0)

    ap.add_argument("--block_on_audio", action="store_true",
                    help="If set, delay PLAY_ACK for trigger phase until AUDIO_DONE or timeout.")

    ap.add_argument("--cfg", default="experiments.yaml")
    ap.add_argument("--force_error_on_seg_a_task",
                    action="append",
                    default=[],
                    help="Repeatable. For these task_ids, write robot_error.json on SEG_DONE A (for external_flag).")

    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    os.chdir(workdir)

    # Discover tasks
    tasks_dir = workdir / "tasks"
    task_ids: Set[str] = set()
    if tasks_dir.exists():
        for p in tasks_dir.glob("experiment_*.yaml"):
            task_ids.add(p.stem.replace("experiment_", ""))
    log(f"✅ Discovered tasks: {sorted(task_ids)}")

    # ZMQ sockets
    ctx = zmq.Context.instance()

    pull = ctx.socket(zmq.PULL)
    pull.bind(f"tcp://*:{args.zmq_port}")
    log(f"🟢 ZMQ listening on tcp://*:{args.zmq_port}")

    # ACK socket to PC1 (set when HELLO_ACK arrives)
    ack_push = None
    ack_endpoint = None

    def ensure_ack_socket(endpoint: str):
        nonlocal ack_push, ack_endpoint
        if endpoint == ack_endpoint and ack_push is not None:
            return
        try:
            if ack_push is not None:
                ack_push.close(0)
        except Exception:
            pass
        ack_push = ctx.socket(zmq.PUSH)
        ack_push.connect(endpoint)
        ack_endpoint = endpoint
        log(f"✅ ACK target set: {endpoint}")

    def send_play_ack(task_id: str, phase: str):
        if ack_push is None:
            log(f"⚠️ Cannot send PLAY_ACK (no ack endpoint yet): task={task_id} phase={phase}")
            return
        msg = f"PLAY_ACK {task_id} {phase}"
        ack_push.send_string(msg)
        log(f"↩️ Sent ACK: {msg}")

    # Optional remote audio sender (PC2 -> PC1 audio listener)
    audio_push = None
    if args.pc1_audio_endpoint:
        audio_push = ctx.socket(zmq.PUSH)
        audio_push.connect(args.pc1_audio_endpoint)
        log(f"🔈 Remote audio enabled. PC1 audio endpoint: {args.pc1_audio_endpoint}")
        log(f"🕒 Audio ACK fallback: {float(args.audio_ack_fallback_sec)}s")
    else:
        log("🔇 Remote audio disabled (no --pc1_audio_endpoint)")

    def send_say(task_id: str, phase: str, text: str):
        if audio_push is None:
            return
        txt = (text or "").strip()
        if not txt:
            return
        b64 = base64.b64encode(txt.encode("utf-8")).decode("ascii")
        audio_push.send_string(f"SAY {task_id} {phase} {b64}")

    # TCP receiver
    tcp = TCPReceiver("0.0.0.0", args.tcp_port)
    log(f"🟢 TCP listening on 0.0.0.0:{args.tcp_port}")

    # Runtime state
    running = False
    current_task_id: Optional[str] = None
    current_participant: Optional[str] = None
    current_timing: Optional[str] = None

    frames_dir: Optional[Path] = None
    frame_count = 0

    # Latching
    latched: Dict[str, bool] = {}
    latched_text: Dict[str, str] = {}
    played_phase: Dict[Tuple[str, str], bool] = {}

    # Optional block-on-audio bookkeeping
    pending_audio_ts: Dict[Tuple[str, str], float] = {}

    def is_current_task(tid: str) -> bool:
        return running and (current_task_id == tid)

    def playback_trigger(timing: str) -> Optional[Tuple[str, str]]:
        # returns (trigger_segment_label, phase)
        if timing == "none":
            return None
        if timing == "immediate":
            return ("A", "immediate")
        if timing == "post_recovery":
            return ("B", "post_recovery")
        return ("A", "immediate")

    def maybe_latch_and_maybe_speak_now(tid: str, new_text: str):
        if latched.get(tid, False):
            return
        latched[tid] = True
        latched_text[tid] = new_text

        if current_timing == "immediate":
            phase = "immediate"
            if played_phase.get((tid, phase), False):
                return
            played_phase[(tid, phase)] = True
            log(f"🗣️ ERROR LATCHED -> speak now (task={tid} phase={phase})")
            send_say(tid, phase, new_text)

    def service_audio_fallback():
        if not args.block_on_audio:
            return
        if not pending_audio_ts:
            return
        now = time.time()
        to_ack = []
        for (tid, phase), ts0 in list(pending_audio_ts.items()):
            if now - ts0 >= float(args.audio_ack_fallback_sec):
                to_ack.append((tid, phase))
        for tid, phase in to_ack:
            pending_audio_ts.pop((tid, phase), None)
            log(f"⏱️ AUDIO_DONE timeout -> fallback PLAY_ACK (task={tid} phase={phase})")
            send_play_ack(tid, phase)

    last_analyzed_end = -1

    def maybe_analyze_and_latch():
        nonlocal last_analyzed_end
        if not running or frames_dir is None or current_task_id is None or current_timing is None or current_participant is None:
            return
        if frame_count < int(args.window_warmup):
            return
        # analyze on stride
        if (frame_count - int(args.window_warmup)) % int(args.window_stride) != 0:
            return

        window_end = frame_count
        if window_end <= last_analyzed_end:
            return
        last_analyzed_end = window_end

        explanation = run_window_analyze(
            workdir=workdir,
            task_id=current_task_id,
            timing=current_timing,
            participant=current_participant,
            frames_dir=frames_dir,
            window_end=window_end,
            window_size=int(args.window_size),
            analyze_timeout_sec=float(args.analyze_timeout_sec),
        )
        if explanation:
            maybe_latch_and_maybe_speak_now(current_task_id, explanation)

    # Poll loop
    log(f"🪟 Window: size={args.window_size}, stride={args.window_stride}, warmup={args.window_warmup}")
    log("✅ Controller ready. Waiting for TCP + triggers...")

    poller = zmq.Poller()
    poller.register(pull, zmq.POLLIN)

    force_on_a: Set[str] = set(args.force_error_on_seg_a_task or [])

    while True:
        socks = dict(poller.poll(10))

        # 1) ZMQ messages
        if pull in socks and socks[pull] == zmq.POLLIN:
            msg = pull.recv_string().strip()
            log(f"📩 ZMQ received: {msg}")

            hello = parse_hello_ack(msg)
            if hello:
                _, endpoint = hello
                ensure_ack_socket(endpoint)
                continue

            ad = parse_audio_done(msg)
            if ad:
                tid, phase = ad
                if not is_current_task(tid):
                    log(f"⚠️ Ignored AUDIO_DONE for task={tid} (running={running}, current={current_task_id})")
                    continue
                if pending_audio_ts.pop((tid, phase), None) is not None:
                    log(f"✅ AUDIO_DONE received -> send PLAY_ACK for task={tid} phase={phase}")
                    send_play_ack(tid, phase)
                else:
                    log(f"ℹ️ AUDIO_DONE received but no pending wait: task={tid} phase={phase}")
                continue

            end_tid = parse_end(msg)
            if end_tid:
                if not is_current_task(end_tid):
                    log(f"⚠️ Ignored END for task={end_tid} (running={running}, current={current_task_id})")
                    continue
                log(f"🏁 END received for task={end_tid}")
                running = False
                current_task_id = None
                current_participant = None
                current_timing = None
                frames_dir = None
                frame_count = 0
                last_analyzed_end = -1
                pending_audio_ts.clear()
                tcp.close_conn()
                continue

            seg = parse_seg_done(msg)
            if seg:
                tid, seg_label = seg
                if not is_current_task(tid):
                    log(f"⚠️ Ignored SEG_DONE for task={tid} seg={seg_label} (running={running}, current={current_task_id})")
                    continue

                timing = current_timing or "immediate"
                log(f"🧭 SEG_DONE arrived: task={tid} seg={seg_label} timing={timing}")

                # Optional external_flag: mark error on segment A
                if seg_label == "A" and tid in force_on_a:
                    video_root = workdir / tid
                    write_robot_error_flag(
                        video_root=video_root,
                        task_id=tid,
                        error_confirmed=True,
                        reason="forced_on_seg_a",
                    )

                trig = playback_trigger(timing)

                if trig is None:
                    send_play_ack(tid, f"after{seg_label}")
                    continue

                trig_seg, trig_phase = trig

                # Non-trigger segments => ACK now
                if seg_label != trig_seg:
                    send_play_ack(tid, f"after{seg_label}")
                    continue

                # Trigger segment behavior
                if timing == "immediate":
                    # Immediate: speak when latched; don't block orchestration
                    send_play_ack(tid, f"after{seg_label}")
                    continue

                # post_recovery trigger at B:
                text = (latched_text.get(tid, "") or "").strip()
                if text and not played_phase.get((tid, trig_phase), False):
                    played_phase[(tid, trig_phase)] = True
                    log(f"🗣️ post_recovery trigger -> speak now (task={tid} phase={trig_phase})")
                    send_say(tid, trig_phase, text)

                    if args.block_on_audio:
                        pending_audio_ts[(tid, trig_phase)] = time.time()
                        continue

                send_play_ack(tid, f"after{seg_label}")
                continue

            parsed = parse_start(msg, args.participant)
            if not parsed:
                log("❌ Invalid command. Expected: START/SEG_DONE/HELLO_ACK/AUDIO_DONE/END")
                continue

            if running:
                log(f"⚠️ Experiment already running (current={current_task_id}) — START ignored")
                continue

            task_id, participant, timing = parsed
            if timing not in {"none", "immediate", "post_recovery"}:
                log(f"❌ Invalid timing condition: {timing}")
                continue
            if task_ids and task_id not in task_ids:
                log(f"❌ Unknown task_id: {task_id} (known: {sorted(task_ids)})")
                continue

            running = True
            current_task_id = task_id
            current_participant = participant
            current_timing = timing

            latched[task_id] = False
            latched_text[task_id] = ""
            played_phase.clear()
            pending_audio_ts.clear()
            last_analyzed_end = -1

            video_root = workdir / task_id
            frames_dir = video_root / "frames"
            video_root.mkdir(parents=True, exist_ok=True)

            # If using external_flag detector, clear stale file at START
            clear_robot_error_flag(video_root)

            clean_dir(frames_dir)
            frame_count = 0

            log(f"🧪 START: task={task_id}, timing={timing}, participant={participant}")
            log(f"📁 Streaming frames to: {frames_dir}")

            tcp.accept_if_needed()
            continue

        # 2) audio fallback timers (only in block_on_audio mode)
        service_audio_fallback()

        # 3) TCP frame receive + maybe analyze
        if running and frames_dir is not None:
            if tcp.conn is None:
                tcp.accept_if_needed()

            payload = tcp.recv_one_png()
            if payload is None:
                time.sleep(0.02)
                continue

            # save frame
            (frames_dir / f"{frame_count:06d}.png").write_bytes(payload)
            frame_count += 1

            maybe_analyze_and_latch()


if __name__ == "__main__":
    main()
