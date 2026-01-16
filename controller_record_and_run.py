#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
controller_record_and_run.py (PC2) — FULL WORKING VERSION (clean + no NameError)

PC2 controller:
- ZMQ PULL: receives START / SEG_DONE / END / HELLO_ACK / AUDIO_DONE
- TCP server: receives length-prefixed PNG frames and saves to <workdir>/<task_id>/frames
- Window analysis: periodically runs run_task.py in window mode to generate explanation_text.txt
- Audio: controller sends SAY to PC1 audio listener (ZMQ PUSH)
- ACK: controller sends PLAY_ACK <task_id> <afterA|afterB> to PC1 via endpoint from HELLO_ACK

Policies:
1) immediate:
   - As soon as text is latched -> send SAY immediate (once)
   - Gate segB by holding PLAY_ACK afterA until AUDIO_DONE immediate (or timeout fallback)

2) post_recovery:
   - A->B contiguous: send PLAY_ACK afterA immediately on SEG_DONE A
   - Prefer speak at SEG_DONE B if latched_text exists (once)
   - If text not ready at B: ACK afterB immediately, mark pending_speak,
     enqueue one extra analyze; when text arrives -> speak ASAP
   - If --block_on_audio: hold PLAY_ACK afterB until AUDIO_DONE post_recovery (or timeout fallback)

3) none:
   - never speak, ACK immediately
"""

import argparse
import base64
import os
import queue
import shutil
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Dict

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
    return parts[1], (parts[2] or default_participant), parts[3].strip()


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
    Accepts a single TCP connection from PC1 and receives frames:
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


# ----------------------------- filesystem helpers -----------------------------

def clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def read_explanation_text(workdir: Path, task_id: str, window_end: int) -> str:
    """
    run_task window mode writes:
      <workdir>/<task_id>/windows/w{window_end:06d}/explanation_text.txt
    """
    txt_path = workdir / task_id / "windows" / f"w{window_end:06d}" / "explanation_text.txt"
    if not txt_path.exists():
        return ""
    txt = txt_path.read_text(encoding="utf-8", errors="replace").strip()
    if not txt or txt == "[NO EXPLANATION]":
        return ""
    return txt


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
                    help="For post_recovery: delay PLAY_ACK afterB until AUDIO_DONE post_recovery or timeout.")
    ap.add_argument("--immediate_gate_fallback_sec", type=float, default=12.0,
                    help="For immediate: if waiting to ACK afterA for AUDIO_DONE, release after this timeout.")

    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    os.chdir(workdir)

    # -------------------- ZMQ setup --------------------
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
        ack_push.setsockopt(zmq.LINGER, 0)
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

    # Remote audio sender (PC2 -> PC1 audio listener)
    audio_push = None
    if args.pc1_audio_endpoint:
        audio_push = ctx.socket(zmq.PUSH)
        audio_push.setsockopt(zmq.LINGER, 0)
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
        log(f"📤 SAY sent: task={task_id} phase={phase} len={len(txt)}")

    # -------------------- TCP setup --------------------
    tcp = TCPReceiver("0.0.0.0", args.tcp_port)
    log(f"🟢 TCP listening on 0.0.0.0:{args.tcp_port}")

    # -------------------- runtime state --------------------
    running = False
    current_task_id: Optional[str] = None
    current_participant: Optional[str] = None
    current_timing: Optional[str] = None

    frames_dir: Optional[Path] = None
    frame_count = 0  # saved frames count; also used as 1-based window_end after increment

    # Latching / speaking state
    latched: Dict[str, bool] = {}
    latched_text: Dict[str, str] = {}
    played_phase: Dict[Tuple[str, str], bool] = {}

    # post_recovery optional block-on-audio bookkeeping (afterB gating)
    pending_audio_ts: Dict[Tuple[str, str], float] = {}

    # immediate AND-gate bookkeeping (afterA gating)
    seg_a_done: Dict[str, bool] = {}
    audio_done_immediate: Dict[str, bool] = {}
    need_ack_afterA: Dict[str, bool] = {}
    pending_afterA_ts: Dict[str, float] = {}

    # post_recovery pending speak (B arrives before text ready)
    post_pending_speak: Dict[str, bool] = {}
    post_pending_speak_ts: Dict[str, float] = {}
    final_window_requested: Dict[str, bool] = {}
    seg_b_done: Dict[str, bool] = {}

    def is_current_task(tid: str) -> bool:
        return running and (current_task_id == tid)

    # -------------------- analysis worker (non-blocking + killable) --------------------
    analyze_q: "queue.Queue[Tuple[str, int]]" = queue.Queue(maxsize=4)

    proc_lock = threading.Lock()
    current_analyze_proc: Optional[subprocess.Popen] = None

    def maybe_latch_and_speak(tid: str, txt: str):
        """
        Called when window analyze produces an explanation text.
        - latch once
        - immediate: speak immediately once
        - post_recovery: latch; if B already happened and pending -> speak ASAP
        """
        if not txt:
            return

        if not latched.get(tid, False):
            latched[tid] = True
            latched_text[tid] = txt
        else:
            # keep latest text if you want; usually identical; harmless
            latched_text[tid] = txt

        timing = (current_timing or "")
        if timing == "immediate":
            if not played_phase.get((tid, "immediate"), False):
                played_phase[(tid, "immediate")] = True
                log(f"🗣️ ERROR LATCHED -> speak now (task={tid} phase=immediate)")
                send_say(tid, "immediate", txt)
            return

        if timing == "post_recovery":
            if seg_b_done.get(tid, False) and post_pending_speak.get(tid, False):
                if not played_phase.get((tid, "post_recovery"), False):
                    played_phase[(tid, "post_recovery")] = True
                    post_pending_speak[tid] = False
                    post_pending_speak_ts.pop(tid, None)
                    log(f"🗣️ post_recovery pending -> speak ASAP when text becomes ready (task={tid})")
                    send_say(tid, "post_recovery", txt)

    def analyze_worker():
        nonlocal current_analyze_proc
        while True:
            tid, window_end = analyze_q.get()
            try:
                # drop if task changed / ended
                if not is_current_task(tid) or current_timing is None or current_participant is None:
                    continue

                cmd = [
                    "python3", "run_task.py",
                    tid, current_timing, current_participant,
                    "--speak_phase", current_timing,
                    "--mode", "window",
                    "--window_end", str(int(window_end)),
                    "--window_size", str(int(args.window_size)),
                    "--pc1_audio", "",  # controller owns audio
                ]
                log(f"🧠 WINDOW ANALYZE: {' '.join(cmd)}")

                with proc_lock:
                    current_analyze_proc = subprocess.Popen(cmd, cwd=str(workdir))

                try:
                    rc = current_analyze_proc.wait(timeout=float(args.analyze_timeout_sec))
                except subprocess.TimeoutExpired:
                    log(f"⏰ WINDOW ANALYZE timeout at window_end={window_end}")
                    # terminate -> kill
                    with proc_lock:
                        p = current_analyze_proc
                    if p is not None:
                        try:
                            p.terminate()
                            p.wait(timeout=0.5)
                        except Exception:
                            try:
                                p.kill()
                            except Exception:
                                pass
                    rc = -1
                finally:
                    with proc_lock:
                        current_analyze_proc = None

                if rc == 0 and is_current_task(tid):
                    txt = read_explanation_text(workdir, tid, window_end)
                    if txt:
                        log(f"🚨 ERROR LATCHED at window_end={window_end} (len={len(txt)})")
                        maybe_latch_and_speak(tid, txt)

            finally:
                analyze_q.task_done()

    threading.Thread(target=analyze_worker, daemon=True).start()

    # -------------------- analysis enqueue logic --------------------
    last_enqueued_end = -1

    def maybe_enqueue_analyze():
        nonlocal last_enqueued_end
        if not running or frames_dir is None or current_task_id is None:
            return

        # If immediate already latched, you can optionally stop analyzing to reduce load:
        # if current_timing == "immediate" and latched.get(current_task_id, False):
        #     return

        if frame_count < int(args.window_warmup):
            return
        if (frame_count - int(args.window_warmup)) % int(args.window_stride) != 0:
            return

        window_end = int(frame_count)  # 1-based
        if window_end <= last_enqueued_end:
            return
        last_enqueued_end = window_end

        try:
            analyze_q.put_nowait((current_task_id, window_end))
            log(f"🧠 Enqueued WINDOW ANALYZE window_end={window_end}")
        except queue.Full:
            log("⚠️ Analyze queue full; skip this window to avoid blocking")

    # -------------------- fallbacks / timers --------------------
    def service_post_audio_fallback():
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

    def service_immediate_gate_fallback():
        if not running or (current_timing or "") != "immediate" or current_task_id is None:
            return
        tid = current_task_id
        if not need_ack_afterA.get(tid, False):
            return
        ts0 = pending_afterA_ts.get(tid, 0.0)
        if ts0 and (time.time() - ts0) >= float(args.immediate_gate_fallback_sec):
            need_ack_afterA[tid] = False
            pending_afterA_ts.pop(tid, None)
            log(f"⏱️ Immediate gate timeout -> fallback PLAY_ACK afterA (task={tid})")
            send_play_ack(tid, "afterA")

    # -------------------- startup info --------------------
    log(f"🪟 Window: size={args.window_size}, stride={args.window_stride}, warmup={args.window_warmup}")
    log("✅ Controller ready. Waiting for TCP + triggers...")

    poller = zmq.Poller()
    poller.register(pull, zmq.POLLIN)

    # -------------------- helpers: END cleanup --------------------
    def kill_analysis_and_drain_queue():
        nonlocal current_analyze_proc
        # stop running proc
        with proc_lock:
            p = current_analyze_proc
        if p is not None:
            log("🛑 Stopping running window analyze (terminate->kill)")
            try:
                p.terminate()
                p.wait(timeout=0.5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
            with proc_lock:
                current_analyze_proc = None

        # drain queue
        drained = 0
        try:
            while True:
                analyze_q.get_nowait()
                analyze_q.task_done()
                drained += 1
        except queue.Empty:
            pass
        if drained:
            log(f"🧹 Drained {drained} queued analyze jobs")

    # -------------------- main loop --------------------
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

                # ---- immediate: release afterA gate if waiting ----
                if phase == "immediate":
                    audio_done_immediate[tid] = True
                    if seg_a_done.get(tid, False) and need_ack_afterA.get(tid, False):
                        need_ack_afterA[tid] = False
                        pending_afterA_ts.pop(tid, None)
                        log(f"✅ Gate open (SEG_DONE A && AUDIO_DONE immediate) -> PLAY_ACK afterA (task={tid})")
                        send_play_ack(tid, "afterA")

                # ---- post_recovery: if we blocked afterB on audio, release ----
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
                kill_analysis_and_drain_queue()

                # Reset state (hard reset)
                running = False
                current_task_id = None
                current_participant = None
                current_timing = None
                frames_dir = None
                frame_count = 0
                last_enqueued_end = -1

                # clear dictionaries
                pending_audio_ts.clear()
                seg_a_done.clear()
                audio_done_immediate.clear()
                need_ack_afterA.clear()
                pending_afterA_ts.clear()

                post_pending_speak.clear()
                post_pending_speak_ts.clear()
                final_window_requested.clear()
                seg_b_done.clear()

                latched.clear()
                latched_text.clear()
                played_phase.clear()

                tcp.close_conn()
                continue

            seg = parse_seg_done(msg)
            if seg:
                tid, seg_label = seg
                if not is_current_task(tid):
                    log(f"⚠️ Ignored SEG_DONE for task={tid} seg={seg_label} (running={running}, current={current_task_id})")
                    continue

                timing = (current_timing or "immediate").strip()
                log(f"🧭 SEG_DONE arrived: task={tid} seg={seg_label} timing={timing}")

                # timing == none: ACK immediately
                if timing == "none":
                    send_play_ack(tid, f"after{seg_label}")
                    continue

                # timing == immediate: hold afterA only if we already sent SAY and not AUDIO_DONE
                if timing == "immediate":
                    if seg_label == "A":
                        seg_a_done[tid] = True

                        has_audio = played_phase.get((tid, "immediate"), False)
                        if has_audio and not audio_done_immediate.get(tid, False):
                            need_ack_afterA[tid] = True
                            pending_afterA_ts[tid] = time.time()
                            log(f"⏸️ Hold PLAY_ACK afterA until AUDIO_DONE immediate (task={tid})")
                            continue

                        send_play_ack(tid, "afterA")
                        continue

                    # SEG_DONE B: ACK immediately
                    send_play_ack(tid, f"after{seg_label}")
                    continue

                # timing == post_recovery
                if timing == "post_recovery":
                    if seg_label == "A":
                        send_play_ack(tid, "afterA")
                        continue

                    # seg_label == B
                    seg_b_done[tid] = True
                    text = (latched_text.get(tid, "") or "").strip()

                    if text and not played_phase.get((tid, "post_recovery"), False):
                        played_phase[(tid, "post_recovery")] = True
                        log(f"🗣️ post_recovery -> speak at SEG_DONE B (task={tid})")
                        send_say(tid, "post_recovery", text)

                        if args.block_on_audio:
                            pending_audio_ts[(tid, "post_recovery")] = time.time()
                            log(f"⏸️ Hold PLAY_ACK afterB until AUDIO_DONE post_recovery (task={tid})")
                            continue

                        send_play_ack(tid, "afterB")
                        continue

                    # no text yet at B -> ACK immediately, mark pending, enqueue one extra analyze
                    post_pending_speak[tid] = True
                    post_pending_speak_ts[tid] = time.time()

                    if not final_window_requested.get(tid, False):
                        final_window_requested[tid] = True
                        window_end = max(1, int(frame_count))
                        try:
                            analyze_q.put_nowait((tid, window_end))
                            log(f"🧠 post_recovery: text not ready at B -> enqueue extra analyze window_end={window_end} (task={tid})")
                        except queue.Full:
                            log("⚠️ Analyze queue full; cannot enqueue extra analyze at B")

                    send_play_ack(tid, "afterB")
                    continue

                # fallback
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
            timing = (timing or "").strip()

            if timing not in {"none", "immediate", "post_recovery"}:
                log(f"❌ Invalid timing condition: {timing}")
                continue

            # Start
            running = True
            current_task_id = task_id
            current_participant = participant
            current_timing = timing

            # per-task init
            latched[task_id] = False
            latched_text[task_id] = ""

            seg_a_done[task_id] = False
            audio_done_immediate[task_id] = False
            need_ack_afterA[task_id] = False
            pending_afterA_ts.pop(task_id, None)

            seg_b_done[task_id] = False
            post_pending_speak[task_id] = False
            post_pending_speak_ts.pop(task_id, None)
            final_window_requested[task_id] = False

            # global-ish clears
            played_phase.clear()
            pending_audio_ts.clear()

            last_enqueued_end = -1

            video_root = workdir / task_id
            frames_dir = video_root / "frames"
            video_root.mkdir(parents=True, exist_ok=True)

            clean_dir(frames_dir)
            frame_count = 0

            log(f"🧪 START: task={task_id}, timing={timing}, participant={participant}")
            log(f"📁 Streaming frames to: {frames_dir}")

            tcp.accept_if_needed()
            continue

        # 2) fallbacks / timers
        service_post_audio_fallback()
        service_immediate_gate_fallback()

        # 3) TCP frame receive + enqueue analyze
        if running and frames_dir is not None:
            if tcp.conn is None:
                tcp.accept_if_needed()

            payload = tcp.recv_one_png()
            if payload is None:
                time.sleep(0.02)
                continue

            (frames_dir / f"{frame_count:06d}.png").write_bytes(payload)
            frame_count += 1
            maybe_enqueue_analyze()


if __name__ == "__main__":
    main()
