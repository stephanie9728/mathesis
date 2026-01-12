#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import struct
import cv2
import numpy as np

HOST = "0.0.0.0"
PORT = 5001

def recvall(conn, n):
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data += chunk
    return data

def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"Listening on {HOST}:{PORT} ...")

    conn, addr = srv.accept()
    print("Connected from:", addr)

    while True:
        size = struct.unpack(">I", recvall(conn, 4))[0]
        payload = recvall(conn, size)

        img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        # img 是 BGR，直接接你的算法
        cv2.imshow("pc2_recv", img)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    conn.close()
    srv.close()

if __name__ == "__main__":
    main()
