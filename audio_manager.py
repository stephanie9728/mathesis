# audio_manager.py
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List


@dataclass
class EspeakConfig:
    voice: str = "en"
    rate: int = 170
    volume: int = 130
    pitch: Optional[int] = None
    debounce_sec: float = 0.15
    kill_timeout_sec: float = 0.5


class AudioManager:
    def __init__(
        self,
        voice: str = "en",
        rate: int = 170,
        volume: int = 130,
        pitch: Optional[int] = None,
        debounce_sec: float = 0.15,
        kill_timeout_sec: float = 0.5,
    ):
        self.cfg = EspeakConfig(
            voice=voice,
            rate=rate,
            volume=volume,
            pitch=pitch,
            debounce_sec=debounce_sec,
            kill_timeout_sec=kill_timeout_sec,
        )
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._last_call_time: float = 0.0
        self._last_text: str = ""

        # prefer espeak-ng, fallback espeak; plus hard fallback paths
        self._espeak = (
            shutil.which("espeak-ng")
            or shutil.which("espeak")
            or ("/usr/bin/espeak-ng" if Path("/usr/bin/espeak-ng").exists() else None)
            or ("/usr/bin/espeak" if Path("/usr/bin/espeak").exists() else None)
        )
        if not self._espeak:
            raise RuntimeError("espeak backend not found. Need 'espeak-ng' or 'espeak' on PC2.")

    def speak(self, text: str, interrupt: bool = True, timing_log: Optional[Dict[str, Any]] = None) -> bool:
        text = (text or "").strip()
        if not text:
            return False

        with self._lock:
            now = time.time()
            if interrupt and (now - self._last_call_time) < self.cfg.debounce_sec and text == self._last_text:
                return False
            self._last_call_time = now
            self._last_text = text

            if interrupt:
                self._stop_locked()

            t_call = time.time()
            if timing_log is not None:
                timing_log["audio_mode"] = "tts_espeak"
                timing_log["tts_call_time"] = t_call
                timing_log["tts_text_len"] = len(text)

            cmd = self._build_espeak_cmd(text)
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            t_start = time.time()
            if timing_log is not None:
                timing_log["tts_start_time"] = t_start
                timing_log["tts_latency_sec"] = t_start - t_call

        return True

    def speak_blocking(self, text: str, interrupt: bool = True, timing_log: Optional[Dict[str, Any]] = None) -> bool:
        ok = self.speak(text=text, interrupt=interrupt, timing_log=timing_log)
        if not ok:
            return False
        proc = None
        with self._lock:
            proc = self._proc
        if proc is not None:
            proc.wait()
            if timing_log is not None:
                timing_log["tts_end_time"] = time.time()
        return True

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def is_playing(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def _build_espeak_cmd(self, text: str, wav_out: Optional[str] = None) -> List[str]:
        cmd = [self._espeak, "-v", self.cfg.voice, "-s", str(self.cfg.rate), "-a", str(self.cfg.volume)]
        if self.cfg.pitch is not None:
            cmd += ["-p", str(self.cfg.pitch)]
        if wav_out:
            cmd += ["-w", wav_out]
        cmd.append(text)
        return cmd

    def _stop_locked(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=self.cfg.kill_timeout_sec)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
