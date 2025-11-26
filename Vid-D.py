# Vid-D.py (NO-VIS VERSION)
# -----------------------------------------
# Usage:
#   python Vid-D.py tasks/experiment_xxx.yaml /path/to/VIDEO_ROOT
#
# Purpose:
#   - Only verifies that scene graph exists
#   - Prints basic stats
#   - NO visualization, NO pyvis, NO matplotlib

import sys
import json
from pathlib import Path

# ---------- 1) 读取命令行参数 ----------
if len(sys.argv) < 3:
    raise SystemExit(
        "Usage: python Vid-D.py <experiment_yaml> <video_root>\n"
        "e.g.   python Vid-D.py tasks/experiment_cut_fruit_tool_error_mini.yaml camera_demo_fruit"
    )

yaml_path  = Path(sys.argv[1]).resolve()
VIDEO_ROOT = Path(sys.argv[2]).resolve()

print(f"📄 Vid-D 使用配置: {yaml_path.name}")
print(f"📂 Vid-D 使用视频目录: {VIDEO_ROOT}")

# ---------- 2) 读取 scene graph ----------
GRAPH_PATH = VIDEO_ROOT / "yolo_scene_graph.json"
assert GRAPH_PATH.exists(), f"❌ 缺少 {GRAPH_PATH}，请先运行 Vid-B"

data = json.loads(GRAPH_PATH.read_text())

nodes = data.get("nodes", [])
edges = data.get("edges", [])

print(f"✅ Scene Graph OK")
print(f"   Nodes: {len(nodes)}")
print(f"   Edges: {len(edges)}")

# ---------- 3) 不再做任何可视化，直接结束 ----------
print("✅ Vid-D finished (visualization disabled by design)")
