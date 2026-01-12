#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, zmq
from datetime import datetime
from audio_manager import AudioManager

def log(m): 
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--voice", default="en")
    ap.add_argument("--rate", type=int, default=170)
    args = ap.parse_args()

    audio = AudioManager(voice=args.voice, rate=args.rate)

    ctx = zmq.Context.instance()
    pull = ctx.socket(zmq.PULL)
    pull.bind(f"tcp://*:{args.port}")
    log(f"🟢 PC2 TTS listening on tcp://*:{args.port}")

    while True:
        raw = pull.recv()
        msg = json.loads(raw.decode("utf-8"))
        cmd = msg.get("cmd", "").upper()

        if cmd == "SAY":
            text = msg.get("text", "")
            interrupt = bool(msg.get("interrupt", True))
            meta = {k: msg.get(k) for k in ["task_id","participant","timing","phase"]}
            log(f"🗣️ SAY {meta} interrupt={int(interrupt)} len={len(text)}")
            audio.speak(text=text, interrupt=interrupt)
        elif cmd == "STOP":
            log("🛑 STOP")
            audio.stop()
        else:
            log(f"⚠️ Unknown cmd: {cmd} {msg}")

if __name__ == "__main__":
    main()
