# tools/frames_to_mp4_apple.py
import cv2
from pathlib import Path

VIDEO_ROOT = Path("camera_demo_apple")
frames_dir = VIDEO_ROOT / "frames"
out_path = VIDEO_ROOT / "raw.mp4"

frames = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
assert frames, f"No frames found in {frames_dir}"

# 读第一帧确定分辨率
img0 = cv2.imread(str(frames[0]))
h, w = img0.shape[:2]
fps = 30  # 你录制时的帧率，如果知道是别的就改成实际值

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

for f in frames:
    img = cv2.imread(str(f))
    if img is None:
        continue
    writer.write(img)

writer.release()
print(f"✅ wrote: {out_path}  frames: {len(frames)} size: ({w}, {h}) fps: {fps}")
