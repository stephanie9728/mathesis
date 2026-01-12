#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import shutil
import socket
import struct
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Set

import zmq
import yaml


# ============================================================
# Utilities
# ============================================================

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def recvall(conn: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data += chunk
    return data


def clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def warn_port_in_use(port: int):
    log(f"❌ ZMQ bind failed: tcp://*:{port} (Address already in use)")
    log("👉 Likely another controller instance is still running.")
    log("Try one of these on PC2:")
    log(f"  - ss -ltnp | grep {port}")
    log(f"  - lsof -i :{port}")
    log("Then kill the PID (e.g. kill <PID>) and restart.")


def try_b64decode(s: str) -> str:
    try:
        return base64.b64decode(s.encode("ascii")).decode("utf-8").strip()
    except Exception:
        return ""


# ============================================================
# TCP PNG receiver (streaming)
# ============================================================

class TcpPngServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server: Optional[socket.socket] = None
        self.conn: Optional[socket.socket] = None
        self.addr = None

    def start_listen(self):
        if self.server is not None:
            return
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        log(f"🟢 TCP listening on {self.host}:{self.port}")

    def accept_if_needed(self):
        if self.conn is not None:
            return
        assert self.server is not None
        self.conn, self.addr = self.server.accept()
        log(f"🔗 TCP connected from {self.addr}")

    def close_conn(self):
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
        self.conn = None

    def recv_one_png(self) -> Optional[bytes]:
        """Return PNG bytes, or None if connection broken."""
        try:
            assert self.conn is not None
            hdr = recvall(self.conn, 4)
            size = struct.unpack(">I", hdr)[0]
            payload = recvall(self.conn, size)
            return payload
        except Exception as e:
            log(f"⚠️ TCP recv error: {e} (closing conn)")
            self.close_conn()
            return None


# ============================================================
# Trigger parsing
# ============================================================

def parse_start(msg: str, default_participant: str) -> Optional[Tuple[str, str, str]]:
    """
    START <task_id> <participant> <timing>
    timing: none / immediate / post_recovery
    compat: 'post' -> 'post_recovery'
    """
    parts = msg.split()
    if len(parts) < 4 or parts[0].upper() != "START":
        return None
    task_id = parts[1]
    participant = parts[2] if parts[2] else default_participant
    timing = parts[3].lower()
    if timing == "post":
        timing = "post_recovery"
    return task_id, participant, timing


def parse_hello_ack(msg: str) -> Optional[Tuple[str, str]]:
    """
    HELLO_ACK <task_id> <ack_endpoint>
    """
    parts = msg.split()
    if len(parts) != 3 or parts[0].upper() != "HELLO_ACK":
        return None
    return parts[1], parts[2]


def parse_seg_done(msg: str) -> Optional[Tuple[str, str]]:
    """
    SEG_DONE <task_id> <A|B>
    """
    parts = msg.split()
    if len(parts) != 3 or parts[0].upper() != "SEG_DONE":
        return None
    return parts[1], parts[2].upper()


def parse_end(msg: str) -> Optional[str]:
    """
    END <task_id>
    """
    parts = msg.split()
    if len(parts) == 2 and parts[0].upper() == "END":
        return parts[1]
    return None


def parse_audio_done(msg: str) -> Optional[Tuple[str, str]]:
    """
    AUDIO_DONE <task_id> <phase>
    """
    parts = msg.split()
    if len(parts) != 3 or parts[0].upper() != "AUDIO_DONE":
        return None
    return parts[1], parts[2]


# Optional (future): ERROR_DETECTED <task_id> <base64_text>
def parse_error_detected(msg: str) -> Optional[Tuple[str, str]]:
    parts = msg.split(maxsplit=2)
    if len(parts) != 3 or parts[0].upper() != "ERROR_DETECTED":
        return None
    return parts[1], parts[2]


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcp_host", default="0.0.0.0")
    ap.add_argument("--tcp_port", type=int, default=5001)
    ap.add_argument("--zmq_port", type=int, default=5555)
    ap.add_argument("--python", default="python3")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--participant", default="P01")

    # window analysis
    ap.add_argument("--window_size", type=int, default=30)
    ap.add_argument("--window_stride", type=int, default=30)
    ap.add_argument("--window_warmup", type=int, default=30, help="min frames before first window analysis")
    ap.add_argument("--analyze_timeout_sec", type=float, default=120.0)

    # remote audio
    ap.add_argument("--pc1_audio_endpoint", default="tcp://10.163.18.91:5557",
                    help="PC1 PULL endpoint for SAY messages, e.g. tcp://10.163.18.91:5557")

    # AUDIO_DONE fallback
    ap.add_argument("--audio_ack_fallback_sec", type=float, default=8.0,
                    help="If AUDIO_DONE not received within this many seconds after SAY, send PLAY_ACK anyway")

    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    tasks_dir = workdir / "tasks"

    log(f"🔈 Remote audio enabled. PC1 audio endpoint: {args.pc1_audio_endpoint}")
    log(f"🪟 Window: size={args.window_size}, stride={args.window_stride}, warmup={args.window_warmup}")
    log(f"🕒 Audio ACK fallback: {args.audio_ack_fallback_sec:.1f}s")

    # Discover tasks (optional gating)
    task_ids: Set[str] = set()
    if tasks_dir.exists():
        for p in tasks_dir.glob("experiment_*.yaml"):
            cfg = yaml.safe_load(p.read_text())
            tid = cfg.get("meta", {}).get("task_id")
            if tid:
                task_ids.add(tid)
    log(f"✅ Discovered tasks: {sorted(task_ids)}")

    ctx = zmq.Context.instance()

    # PULL: receives triggers from PC1 AND AUDIO_DONE
    pull = ctx.socket(zmq.PULL)
    try:
        pull.bind(f"tcp://*:{args.zmq_port}")
    except zmq.ZMQError as e:
        if "Address already in use" in str(e):
            warn_port_in_use(args.zmq_port)
            raise SystemExit(2)
        raise
    log(f"🟢 ZMQ listening on tcp://*:{args.zmq_port}")

    # ACK back to PC1 orchestration (created after HELLO_ACK)
    ack_push: Optional[zmq.Socket] = None
    ack_target: Optional[str] = None

    def ensure_ack_socket(endpoint: str):
        nonlocal ack_push, ack_target
        if ack_target == endpoint and ack_push is not None:
            return
        ack_target = endpoint
        if ack_push is None:
            ack_push = ctx.socket(zmq.PUSH)
            ack_push.setsockopt(zmq.LINGER, 0)
            ack_push.setsockopt(zmq.IMMEDIATE, 1)
        ack_push.connect(ack_target)
        log(f"✅ ACK target set: {ack_target}")
        time.sleep(0.15)

    def send_play_ack(task_id: str, phase: str):
        if ack_push is None:
            log("⚠️ No ACK socket yet; skip PLAY_ACK")
            return
        msg = f"PLAY_ACK {task_id} {phase}"
        try:
            ack_push.send_string(msg)
            log(f"↩️ Sent ACK: {msg}")
        except Exception as e:
            log(f"⚠️ Failed to send PLAY_ACK: {e}")

    # Remote audio sender (PC2 -> PC1 audio listener)
    audio_push = ctx.socket(zmq.PUSH)
    audio_push.setsockopt(zmq.LINGER, 0)
    audio_push.setsockopt(zmq.IMMEDIATE, 1)
    audio_push.connect(args.pc1_audio_endpoint)
    log(f"✅ Connected to PC1 audio listener: {args.pc1_audio_endpoint}")
    time.sleep(0.15)

    def send_say(task_id: str, phase: str, text: str):
        payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
        msg = f"SAY {task_id} {phase} {payload}"
        try:
            audio_push.send_string(msg)
            log(f"📤 Sent SAY to PC1 audio: task={task_id} phase={phase} (len={len(text)})")
        except Exception as e:
            log(f"⚠️ Failed to send SAY: {e}")

    # pending play_ack until AUDIO_DONE arrives
    # store timestamp so we can fallback
    pending_ack_ts: Dict[Tuple[str, str], float] = {}

    # TCP receiver (streaming)
    tcp = TcpPngServer(args.tcp_host, args.tcp_port)
    tcp.start_listen()

    # Current run gating (prevents stale SEG_DONE)
    running: bool = False
    current_task_id: Optional[str] = None
    current_participant: Optional[str] = None
    current_timing: Optional[str] = None
    frames_dir: Optional[Path] = None
    frame_count: int = 0  # number of frames written so far

    # Latch explanation on detection; play later
    latched_text: Dict[str, str] = {}
    latched: Dict[str, bool] = {}
    played_phase: Dict[Tuple[str, str], bool] = {}

    # Remember timing per task
    last_timing_by_task: Dict[str, str] = {}

    def is_current_task(tid: str) -> bool:
        return running and (current_task_id == tid)

    # Which SEG_DONE triggers playback for each condition
    def playback_trigger(timing: str) -> Optional[Tuple[str, str]]:
        """
        return (seg_label, phase) or None if should never speak
        """
        if timing == "immediate":
            return ("A", "afterA")
        if timing == "post_recovery":
            return ("B", "afterB")
        return None

    def run_window_analyze(task_id: str, timing: str, participant: str, window_end_1based: int) -> Optional[str]:
        """
        Calls:
          python run_task.py <task_id> <timing> <participant> --mode window --window_end N --window_size W
        Contract:
          - rc=0 => stdout includes explanation (we take last "content-like" line)
          - rc=10 => no explanation
        """
        cmd = [
            args.python, "run_task.py", task_id, timing, participant,
            "--mode", "window",
            "--window_end", str(window_end_1based),
            "--window_size", str(args.window_size),
        ]
        log("🧠 WINDOW ANALYZE: " + " ".join(cmd))
        try:
            p = subprocess.run(
                cmd,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=float(args.analyze_timeout_sec),
                check=False,
            )
        except subprocess.TimeoutExpired:
            log(f"⚠️ Window analyze timeout (> {args.analyze_timeout_sec}s) at window_end={window_end_1based}")
            return None

        if p.returncode == 0:
            out = (p.stdout or "").strip()
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            cand = [ln for ln in lines if not ln.startswith("✅") and not ln.startswith("🪟") and not ln.startswith("🎉")]
            explanation = (cand[-1] if cand else (lines[-1] if lines else "")).strip()
            if explanation and explanation != "[NO EXPLANATION]":
                return explanation
            return None

        if p.returncode == 10:
            return None

        err_tail = (p.stderr or "").strip()[-400:]
        out_tail = (p.stdout or "").strip()[-400:]
        log(f"⚠️ Window analyze failed rc={p.returncode}\nSTDERR(last400): {err_tail}\nSTDOUT(last400): {out_tail}")
        return None

    def maybe_analyze_and_latch():
        nonlocal frame_count
        if not running or not current_task_id or not current_timing or not current_participant:
            return
        if current_timing not in {"immediate", "post_recovery"}:
            return
        if latched.get(current_task_id, False):
            return
        if frame_count < int(args.window_warmup):
            return
        if args.window_stride <= 0:
            return
        if (frame_count % int(args.window_stride)) != 0:
            return

        explanation = run_window_analyze(current_task_id, current_timing, current_participant, frame_count)
        if explanation:
            latched[current_task_id] = True
            latched_text[current_task_id] = explanation
            log(f"🚨 ERROR LATCHED at frame={frame_count} (len={len(explanation)}) — will play at trigger segment")
        else:
            log(f"✅ No error latched at window_end={frame_count}")

    def final_window_fallback_latch_if_needed(task_id: str) -> bool:
        """
        At playback trigger, if not latched yet, run one last window analyze using latest frame_count.
        Return True if latched after fallback.
        """
        if latched.get(task_id, False):
            return True
        if frame_count <= 0:
            return False
        if not current_timing or not current_participant:
            return False

        log(f"🧩 Final-window fallback at trigger: trying window_end={frame_count}")
        explanation = run_window_analyze(task_id, current_timing, current_participant, frame_count)
        if explanation:
            latched[task_id] = True
            latched_text[task_id] = explanation
            log(f"🚨 ERROR LATCHED by FINAL WINDOW (len={len(explanation)})")
            return True
        log("🧩 Final-window fallback: still no explanation")
        return False

    def start_audio_wait(task_id: str, phase: str, text: str):
        """
        Send SAY and start waiting for AUDIO_DONE. Fallback timer will send PLAY_ACK if needed.
        """
        send_say(task_id, phase, text)
        pending_ack_ts[(task_id, phase)] = time.time()

    def service_audio_ack_fallback():
        """
        If AUDIO_DONE is missing too long, ACK anyway to avoid blocking PC1.
        """
        if not pending_ack_ts:
            return
        now = time.time()
        to_ack = []
        for (tid, phase), ts0 in pending_ack_ts.items():
            if now - ts0 >= float(args.audio_ack_fallback_sec):
                to_ack.append((tid, phase))
        for tid, phase in to_ack:
            pending_ack_ts.pop((tid, phase), None)
            log(f"⏱️ AUDIO_DONE timeout -> fallback PLAY_ACK (task={tid} phase={phase})")
            send_play_ack(tid, phase)

    log("✅ Controller ready. Waiting for TCP + triggers...")

    poller = zmq.Poller()
    poller.register(pull, zmq.POLLIN)

    while True:
        # Poll ZMQ briefly so we can also read TCP frames + run fallback timers
        socks = dict(poller.poll(10))

        # 1) Handle ZMQ messages
        if pull in socks and socks[pull] == zmq.POLLIN:
            msg = pull.recv_string().strip()
            log(f"📩 ZMQ received: {msg}")

            # HELLO_ACK: set ack endpoint anytime
            hello = parse_hello_ack(msg)
            if hello:
                _, endpoint = hello
                ensure_ack_socket(endpoint)
                continue

            # AUDIO_DONE: only accept for current task; then ACK
            ad = parse_audio_done(msg)
            if ad:
                tid, phase = ad
                if not is_current_task(tid):
                    log(f"⚠️ Ignored AUDIO_DONE for task={tid} (running={running}, current={current_task_id})")
                    continue
                if pending_ack_ts.pop((tid, phase), None) is not None:
                    log(f"✅ AUDIO_DONE received -> send PLAY_ACK for task={tid} phase={phase}")
                    send_play_ack(tid, phase)
                else:
                    log(f"ℹ️ AUDIO_DONE received but no pending wait: task={tid} phase={phase}")
                continue

            # END: end run
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
                pending_ack_ts.clear()
                continue

            # Optional: ERROR_DETECTED latch via message (future)
            ed = parse_error_detected(msg)
            if ed:
                tid, b64 = ed
                if not is_current_task(tid):
                    log(f"⚠️ Ignored ERROR_DETECTED for task={tid} (running={running}, current={current_task_id})")
                    continue
                if not latched.get(tid, False):
                    text = try_b64decode(b64)
                    if text:
                        latched[tid] = True
                        latched_text[tid] = text
                        log(f"🚨 ERROR_DETECTED latched via ZMQ (len={len(text)}) — will play at trigger segment")
                else:
                    log(f"ℹ️ ERROR_DETECTED ignored; already latched task={tid}")
                continue

            # SEG_DONE: decide if this is the playback trigger; if so, play latched or fallback
            seg = parse_seg_done(msg)
            if seg:
                tid, seg_label = seg
                if not is_current_task(tid):
                    log(f"⚠️ Ignored SEG_DONE for task={tid} seg={seg_label} (running={running}, current={current_task_id})")
                    continue

                timing = current_timing or last_timing_by_task.get(tid, "immediate")
                trig = playback_trigger(timing)
                log(f"🧭 SEG_DONE arrived: task={tid} seg={seg_label} timing={timing}")

                if trig is None:
                    # none: just ACK the corresponding phase to keep orchestration unblocked
                    phase = f"after{seg_label}"
                    send_play_ack(tid, phase)
                    continue

                trig_seg, trig_phase = trig
                # Always ACK non-trigger segments immediately
                if seg_label != trig_seg:
                    phase = f"after{seg_label}"
                    send_play_ack(tid, phase)
                    continue

                # This is the trigger segment (immediate->A, post_recovery->B)
                if played_phase.get((tid, trig_phase), False):
                    log(f"ℹ️ Trigger {trig_phase} already played; ACK")
                    send_play_ack(tid, trig_phase)
                    continue

                # Final-window fallback if not latched yet
                final_window_fallback_latch_if_needed(tid)

                text = latched_text.get(tid, "").strip()
                if text:
                    played_phase[(tid, trig_phase)] = True
                    log(f"🗣️ Play explanation at trigger {trig_phase}: task={tid}")
                    start_audio_wait(tid, trig_phase, text)
                    # wait AUDIO_DONE (or fallback timer) before ACK
                    continue

                # Still no explanation => do not speak, ACK immediately
                log(f"🔇 No latched explanation even after final-window fallback -> ACK {trig_phase}")
                send_play_ack(tid, trig_phase)
                continue

            # START: begin run, start streaming frames, start window analysis
            parsed = parse_start(msg, args.participant)
            if not parsed:
                log("❌ Invalid command. Expected: "
                    "START <task_id> <participant> <timing> | "
                    "SEG_DONE <task_id> <A|B> | "
                    "HELLO_ACK <task_id> <endpoint> | "
                    "AUDIO_DONE <task_id> <phase> | "
                    "END <task_id>")
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
            last_timing_by_task[task_id] = timing

            # reset state for this task
            latched[task_id] = False
            latched_text[task_id] = ""
            pending_ack_ts.clear()
            played_phase.clear()

            # prepare directories
            video_root = workdir / task_id
            frames_dir = video_root / "frames"
            video_root.mkdir(parents=True, exist_ok=True)
            clean_dir(frames_dir)
            frame_count = 0

            log(f"🧪 START: task={task_id}, timing={timing}, participant={participant}")
            log(f"📁 Streaming frames to: {frames_dir}")

            tcp.accept_if_needed()
            continue

        # 2) Service audio fallback timers every loop
        service_audio_ack_fallback()

        # 3) If running, read one TCP frame per loop and maybe analyze
        if running and frames_dir is not None:
            if tcp.conn is None:
                tcp.accept_if_needed()

            payload = tcp.recv_one_png()
            if payload is None:
                time.sleep(0.02)
                continue

            (frames_dir / f"{frame_count:06d}.png").write_bytes(payload)
            frame_count += 1

            # Window analysis / latch (immediate & post_recovery)
            maybe_analyze_and_latch()


if __name__ == "__main__":
    main()
