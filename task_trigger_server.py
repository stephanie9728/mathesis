#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import zmq
from datetime import datetime
from pathlib import Path


class TaskRunner:
    def __init__(self, python_exec: str, workdir: Path):
        self.python_exec = python_exec
        self.workdir = workdir
        self.proc: subprocess.Popen | None = None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, participant: str, timing: str):
        if self.running():
            print("⚠️ RUNNING already, ignore START")
            return

        cmd = [self.python_exec, "run_task.py", "--timing", timing, "--participant", participant]
        print("🚀 Launch:", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, cwd=str(self.workdir))

    def stop(self):
        if self.running():
            print("🛑 Terminate running task...")
            self.proc.terminate()
        else:
            print("ℹ️ No running task to stop")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--timing", default="immediate")
    parser.add_argument("--participant", default="P01")
    args = parser.parse_args()

    runner = TaskRunner(args.python, Path(args.workdir).resolve())

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PULL)
    sock.bind(f"tcp://*:{args.port}")

    print(f"🟢 PC2 trigger listening on tcp://*:{args.port}")
    print(f"   workdir={runner.workdir}")
    print(f"   will run: {args.python} run_task.py --timing {args.timing} --participant {args.participant}")

    while True:
        msg = sock.recv_string().strip()
        ts = datetime.now().strftime("%H:%M:%S")
        up = msg.upper()

        print(f"📩 [{ts}] {msg}")

        if up.startswith("START"):
            runner.start(participant=args.participant, timing=args.timing)

        elif up.startswith("STOP"):
            runner.stop()

        else:
            print("⚠️ Unknown command. Use START or STOP.")


if __name__ == "__main__":
    main()
