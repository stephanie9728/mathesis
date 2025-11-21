# main/execute_replan.py  —— 绝对路径化 + 兜底版
import os
import json
import pickle
import shutil
from pathlib import Path
from contextlib import contextmanager

from ai2thor.controller import Controller
from ai2thor.platform import OSXIntel64

from utils import *
from task_utils import *
from constants import *
from action_primitives import *

# 放到 imports 之后（如果还没有）
def _safe_step(ctrl, action, **kwargs):
    """包一层，避免不支持的动作导致超时；失败就返回 False 并继续。"""
    try:
        ev = ctrl.step(action=action, **kwargs)
        return ev.metadata.get("lastActionSuccess", False)
    except Exception as e:
        print(f"[warn] step {action} failed:", e)
        return False

# ---------- 小工具 ----------
def _digits_in_name(name: str) -> int:
    """提取文件名中最后一个整数，用于排序帧号。"""
    import re, os
    stem = os.path.splitext(os.path.basename(name))[0]
    m = re.findall(r"(\d+)", stem)
    return int(m[-1]) if m else -1


@contextmanager
def _pushd(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# ---------- 执行 replanning 计划 ----------
def execute_correction_plan(task_idx, f_name, taskUtil, plan_path: Path):
    """从给定的 plan_path 读 replan.json 后逐条执行。"""
    with open(plan_path, "r") as f:
        plan = json.load(f)["plan"]

    for instr in plan:
        lis = instr.split(",")
        lis = [item.strip("() ") for item in lis]
        action = lis[0]
        params = lis[1:]

        taskUtil.chosen_failure = "blocking" if taskUtil.chosen_failure == "blocking" else None
        print("action, params:", action, params)

        # action_primitives 里函数是全局导入的
        func = globals()[action]
        func(taskUtil, *params, fail_execution=False, replan=True)

    is_success = check_task_success(task_idx, taskUtil.controller.last_event)
    print("Task success :-)" if is_success else "Task fail :-(")
    return is_success

# --- PATCH: ensure recovery frames exist, even if plan is empty ---
from pathlib import Path
import imageio

rec_root = Path(taskUtil.repo_path) / "recovery" / taskUtil.specific_folder_name
ego_dir = rec_root / "ego_img"
ego_dir.mkdir(parents=True, exist_ok=True)

# 若没有任何帧，至少存一帧当前画面，避免 generate_video() 报错
if not any(ego_dir.glob("*.png")):
    # controller.last_event.frame 是 (H,W,3) 的 numpy 数组
    imageio.imwrite(ego_dir / "frame_000000.png", taskUtil.controller.last_event.frame)
# --- /PATCH ---

# ---------- 运行校正 ----------
def run_correction(data_path, f_name):
    """
    data_path: 仓库根路径（字符串）。
    f_name   : 任务子路径，例如 'makeCoffee/makeCoffee-1'。
    所有相对路径都锚定到 data_path，避免工作目录带来的路径混乱。
    """
    base = Path(data_path).resolve()

    # 路径准备
    thor_dir   = base / "thor_tasks" / f_name
    events_dir = thor_dir / "events"
    video_mp4  = thor_dir / "original-video.mp4"

    plan_path  = base / "state_summary" / f_name / "replan.json"
    plan_alt   = base / "main" / "state_summary" / f_name / "replan.json"

    task_path  = thor_dir / "task.json"
    task_alt   = base / "main" / "thor_tasks" / f_name / "task.json"

    # replan.json：根下没有就从 main 下拷一份
    if not plan_path.exists():
        if plan_alt.exists():
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(plan_alt, plan_path)
            print("[info] copied replan.json ->", plan_path)
        else:
            raise FileNotFoundError(f"[run_correction] replan.json not found: {plan_path} nor {plan_alt}")

    # task.json：同理
    if not task_path.exists():
        if task_alt.exists():
            task_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(task_alt, task_path)
            print("[info] copied task.json ->", task_path)
        else:
            raise FileNotFoundError(f"[run_correction] task.json not found: {task_path} nor {task_alt}")

    with open(task_path, "r") as f:
        task = json.load(f)

    # 创建控制器（用 task 里的 scene，缺省 FloorPlan1）
    controller = Controller(
        agentMode="default",
        massThreshold=None,
        scene=task.get("scene", "FloorPlan1"),
        visibilityDistance=1.5,
        gridSize=0.25,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        width=960,
        height=960,
        fieldOfView=60,
        platform=OSXIntel64,
    )

    # 计算 last_frame，并尽量拿到最终帧事件（如果有 .pickle）
    last_frame = 0
    final_event = None

    if events_dir.exists():
        names = sorted(os.listdir(events_dir), key=_digits_in_name)
        if names:
            last_name = names[-1]
            last_frame = _digits_in_name(last_name)
            last_path = events_dir / last_name
            if last_path.suffix == ".pickle":
                try:
                    with open(last_path, "rb") as f:
                        final_event = pickle.load(f)
                except Exception as e:
                    print("[warn] failed to load final pickle:", last_path, "->", e)

    # 如果没有 events 或没有可用的 pickle，就用视频估一个帧号兜底
    if final_event is None and last_frame <= 0:
        est = 60  # 默认给 2 秒 @30fps
        if video_mp4.exists():
            try:
                from moviepy.editor import VideoFileClip
                with VideoFileClip(str(video_mp4)) as clip:
                    fps = clip.fps or 30
                    dur = clip.duration or 2.0
                    est = max(30, int(round(fps * dur)))
            except Exception as e:
                print("[warn] estimate last_frame from video failed:", e)
        last_frame = est - 1
        print(f"[info] no events pickle; use last_frame={last_frame}")

    

    # —— 运行 replanning 计划 —— 
    reachable_positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]

    chosen_failure = task.get("chosen_failure", None)
    failure_injection_params = task.get("failure_injection_params", None)

    taskUtil = TaskUtil(
        folder_name=f_name,
        controller=controller,
        reachable_positions=reachable_positions,
        failure_injection=False,
        index=0,
        repo_path=str(base),
        chosen_failure=chosen_failure,
        failure_injection_params=failure_injection_params,
        counter=last_frame,
        replan=True,
    )
    # 用我们已确定好的 plan_path；并把 CWD 推到仓库根（base）
    with _pushd(base):
        is_success = execute_correction_plan(task["task_idx"], f_name, taskUtil, plan_path)


    # 回写 success 到 replan.json
    with open(plan_path, "r") as f:
        replan_json = json.load(f)
    replan_json["success"] = is_success
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "w") as f:
        json.dump(replan_json, f, indent=2)
    
    from pathlib import Path
    import imageio

    # --- ensure recovery dirs & at least 1 frame ---
    rec_root = Path(taskUtil.repo_path) / "recovery" / taskUtil.specific_folder_name
    ego_dir = rec_root / "ego_img"
    ego_dir.mkdir(parents=True, exist_ok=True)

    # 如果没有任何帧，至少存一帧当前画面，避免 generate_video 报目录不存在
    if not any(ego_dir.glob("*.png")):
        # controller.last_event.frame 是 (H,W,3) 的图像数组
        imageio.imwrite(ego_dir / "frame_000000.png", taskUtil.controller.last_event.frame)
# -----------------------------------------------

    # 生成校正过程视频
    generate_video(taskUtil, recovery_video=True)

    controller.stop()
    return is_success
