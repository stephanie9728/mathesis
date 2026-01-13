#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import zmq


@dataclass
class SayClient:
    pc1_endpoint: str  # e.g. tcp://10.163.18.91:5557

    def __post_init__(self):
        ctx = zmq.Context.instance()
        self.push = ctx.socket(zmq.PUSH)
        # give socket time to flush on process exit (helps short scripts)
        self.push.setsockopt(zmq.LINGER, 1000)  # ms
        self.push.connect(self.pc1_endpoint)
        self._last_t = 0.0
        self._last_text = ""

    def send_say(
        self,
        text: str,
        task_id: str,
        phase: str,
        interrupt: bool = True,
        debounce_sec: float = 0.15,
        flush_sleep_sec: float = 0.0,
        timing_log: Optional[Dict[str, Any]] = None,
    ) -> bool:
        text = (text or "").strip()
        if not text:
            return False

        now = time.time()
        if interrupt and (now - self._last_t) < debounce_sec and text == self._last_text:
            return False
        self._last_t, self._last_text = now, text

        b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        msg = f"SAY {task_id} {phase} {b64}"
        self.push.send_string(msg)

        if flush_sleep_sec and flush_sleep_sec > 0:
            time.sleep(flush_sleep_sec)

        if timing_log is not None:
            timing_log["audio_mode"] = "remote_pc1_openai"
            timing_log["tts_call_time"] = now
            timing_log["tts_text_len"] = len(text)
            timing_log["pc1_endpoint"] = self.pc1_endpoint

        return True
