# Vid-A：录制视频 → 抽帧 (2 FPS)
import cv2, time, os
from pathlib import Path

# ==== 只改这里（其余基本别动）====
ROOT_NAME      = "camera_demo_mini"   # 输出根目录名称
CAM_INDEX      = 0                     # 内建摄像头一般是 0；外接可试 1/2
BACKEND        = cv2.CAP_AVFOUNDATION  # macOS 推荐；Linux 可用 cv2.CAP_V4L2
DURATION_SEC   = 10                     # 录制时长（秒）
FRAME_W,FRAME_H= 1280,720              # 期望分辨率（按需调）
CAPTURE_FPS    = 30                    # 从摄像头读取的 FPS
EXTRACT_FPS    = 2                     
# ==================================

ROOT = Path.cwd()/ROOT_NAME
ROOT.mkdir(exist_ok=True)

# 录制原始视频（注意：如摄像头被占用先关闭其他应用）
cap = cv2.VideoCapture(CAM_INDEX, BACKEND)
assert cap.isOpened(), "打不开摄像头，请检查系统权限/是否被占用"

# 尝试设置分辨率和 FPS（有些摄像头不完全听话，尽力而为）
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS,         CAPTURE_FPS)

raw_mp4 = ROOT/"raw.mp4"
fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
# 实际写入 FPS 用 CAPTURE_FPS，保证视频时长 ≈ DURATION_SEC
writer  = cv2.VideoWriter(str(raw_mp4), fourcc, CAPTURE_FPS, (FRAME_W, FRAME_H))

t0 = time.time()
n_frames = 0
while time.time()-t0 < DURATION_SEC:
    ok, frame = cap.read()
    if not ok:
        print("read() 失败，提前结束"); break
    cv2.imshow("Recording Preview", frame)   # <-- 实时显示
    if cv2.waitKey(1) & 0xFF == ord('q'):    # <-- 可提前按 q 结束
        print("提前结束录制")
        break
    writer.write(frame)
    n_frames += 1

cv2.destroyAllWindows()


cap.release()
writer.release()
print(f"Saved video -> {raw_mp4}  ({n_frames} frames)")

# 抽帧（2 FPS）→ 保存到 ROOT/frames
frames_dir = ROOT/"frames"
frames_dir.mkdir(exist_ok=True)

# 重新打开视频，按步长取帧
cap = cv2.VideoCapture(str(raw_mp4))
assert cap.isOpened(), "打不开刚才录下的视频"

# 根据写入 FPS 计算步长
actual_fps = cap.get(cv2.CAP_PROP_FPS) or CAPTURE_FPS
step = max(1, int(round(actual_fps / EXTRACT_FPS)))

i = j = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    if i % step == 0:
        outp = frames_dir/f"frame_{j:05d}.jpg"
        cv2.imwrite(str(outp), frame)
        j += 1
    i += 1

cap.release()
print(f"Extracted {j} frames -> {frames_dir}")

# 供后续 cell 使用的“根目录指针”（Vid-B / Vid-C 会用到）
VIDEO_ROOT = ROOT
print("VIDEO_ROOT =", VIDEO_ROOT)

