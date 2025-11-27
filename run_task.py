import sys
import subprocess
from pathlib import Path
import yaml

py = sys.executable

task_name   = sys.argv[1]
timing_cond = sys.argv[2]
participant = sys.argv[3]

# ===== 1️⃣ 加载对应 YAML（不依赖 paths）=====
yaml_path = Path(f"tasks/experiment_{task_name}.yaml")
cfg = yaml.safe_load(yaml_path.read_text())

# ===== 2️⃣ 覆写 meta（实验控制入口）=====
cfg.setdefault("meta", {})
cfg["meta"]["explanation_timing"] = timing_cond
cfg["meta"]["participant_id"]     = participant
cfg["meta"]["task_name"]          = task_name

yaml_path_tmp = Path("tasks/_tmp_run.yaml")
yaml_path_tmp.write_text(yaml.dump(cfg, allow_unicode=True))

# ===== 3️⃣ 根据 task 自动选择 video_root =====
TASK_TO_VIDEO = {
    "cut_fruit_tool_error":       "camera_demo_fruit",
    "pour_cereal_inability":      "camera_demo_cereal",
    "grasp_apple_uncertainty":    "camera_demo_apple",
}

if task_name not in TASK_TO_VIDEO:
    raise ValueError(f"Unknown task_name: {task_name}")

video_root = Path(TASK_TO_VIDEO[task_name]).resolve()

print(f"✅ Task: {task_name} | Timing: {timing_cond} | PID: {participant}")
print(f"✅ Video Root: {video_root}")

# ===== 4️⃣ 执行完整流水线 =====
subprocess.run([py, "Vid-B.py", str(video_root)], check=True)
subprocess.run([py, "Vid-C.py", str(video_root)], check=True)
subprocess.run([py, "Vid-D.py", str(yaml_path_tmp), str(video_root)], check=True)
subprocess.run([py, "Vid-E.py", str(yaml_path_tmp), str(video_root)], check=True)
