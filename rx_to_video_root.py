#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import socket
import struct
import time
from pathlib import Path

import cv2
import numpy as np


def recvall(conn, n):
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data += chunk
    return data


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5001)
    p.add_argument("--video_root", required=True, help="e.g. camera_demo_fruit")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max_frames", type=int, default=0, help="0 means infinite")
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()

    video_root = Path(args.video_root).resolve()
    frames_dir = video_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 记录一个简单的 session 文件，方便下游脚本定位
    (video_root / "session.txt").write_text(f"start_time={time.time()}\n")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[RX] Listening on {args.host}:{args.port}")
    conn, addr = srv.accept()
    print("[RX] Connected from:", addr)

    i = 0
    t0 = time.time()
    try:
        while True:
            size = struct.unpack(">I", recvall(conn, 4))[0]
            payload = recvall(conn, size)
            img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            # 保存为 png 序列
            out_path = frames_dir / f"{i:06d}.png"
            cv2.imwrite(str(out_path), img)
            i += 1

            if i % 60 == 0:
                fps = i / max(1e-6, (time.time() - t0))
                print(f"[RX] saved {i} frames, avg fps={fps:.1f} -> {frames_dir}")

            if args.preview:
                cv2.imshow("rx_preview", img)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if args.max_frames > 0 and i >= args.max_frames:
                break

    except Exception as e:
        print("[RX] Error:", repr(e))
    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            srv.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[RX] Done. frames:", i, "dir:", frames_dir)


if __name__ == "__main__":
    main()
