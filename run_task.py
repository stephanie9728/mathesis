# run_task.py 伪代码框架

import sys, subprocess, yaml
from pathlib import Path

py = "/Users/stephaniezhang/miniconda3_x86/envs/thor-x86/bin/python"


task_name     = sys.argv[1]          # e.g. cut_fruit_tool_error
timing_cond   = sys.argv[2]          # "none" | "immediate" | "delayed"
participant   = sys.argv[3] if len(sys.argv) > 3 else "pilot"

yaml_path = Path(f"tasks/experiment_{task_name}_mini.yaml")
cfg = yaml.safe_load(yaml_path.read_text())

# 覆盖 YAML 里的 timing（保证一个入口控制）
cfg.setdefault("meta", {})
cfg["meta"]["explanation_timing"] = timing_cond
cfg["meta"]["participant_id"]     = participant
cfg["meta"]["task_name"]          = task_name
yaml_path_tmp = Path("tasks/_tmp_run.yaml")
yaml_path_tmp.write_text(yaml.dump(cfg, allow_unicode=True))

video_root = Path(cfg["paths"]["video_root"]).resolve()

print(f"✅ Task: {task_name} | Timing: {timing_cond} | PID: {participant}")
print(f"✅ Video Root: {video_root}")

subprocess.run([py, "Vid-B.py", str(video_root)], check=True)
subprocess.run([py, "Vid-C.py", str(video_root)], check=True)
subprocess.run([py, "Vid-D.py", str(video_root)], check=True)
subprocess.run([py, "Vid-E.py", str(yaml_path_tmp), str(video_root)], check=True)
