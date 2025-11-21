# main/exp.py  —— audio-free, drop-in replacement
from __future__ import annotations

import os
import json
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# 本项目模块
from constants import *
from scene_graph import SceneGraph
from scene_graph import Node as SceneGraphNode
from get_local_sg import get_scene_graph, save_pcd
from data import *                 # 用到 convert_step_to_timestep/… 等
from utils import *                # 用到 get_robot_plan / get_initial_plan / get_replan_prefix / translate_plan 等
from clip_utils import *
from point_cloud_utils import *

# -----------------------------------------------------------------------------
# 全局常量/工具
# -----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]   # .../reflect
_LLM_DIR   = _REPO_ROOT / "LLM"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@contextmanager
def _pushd(path: Path | str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 文本化场景图
# -----------------------------------------------------------------------------
def get_scene_text(scene_graph: SceneGraph) -> str:
    out, visited = "", []
    for node in set(scene_graph.nodes):
        out += node.get_name() + ", "
    if out:
        out = out[:-2] + ". "
    for edge_key, edge in scene_graph.edges.items():
        s, t = edge_key
        if (edge_key not in visited) and ((t, s) not in visited):
            out += f"{edge.start.name} is {edge.edge_type} {edge.end.name}. "
            visited.append(edge_key)
    return out[:-1] if out else out


def get_held_object(folder_name: str, step_idx: int) -> str | None:
    """返回 step_idx 时刻机器人抓取的物体名（无则 None）"""
    while step_idx >= 0:
        p = Path(f"state_summary/{folder_name}/local_graphs/local_sg_{step_idx}.pkl")
        if p.exists():
            with p.open("rb") as f:
                local_sg = pickle.load(f)
            for key in local_sg.edges:
                if "robot gripper" in key and key[0] != "nothing":
                    return key[0]
            return None
        step_idx -= 1
    return None


# -----------------------------------------------------------------------------
# 关键：无音频版本的图生成与摘要生成
#   - 保留原有参数签名以兼容 notebook：WITH_AUDIO / detected_sounds 不再使用
# -----------------------------------------------------------------------------
def generate_scene_graphs(
    folder_name: str,
    events: List,
    object_list: List[str],
    nav_actions: Dict[Tuple[int, int], str],
    interact_actions: Dict[int, str],
    WITH_AUDIO: int | bool = False,         # 兼容参数（忽略）
    detected_sounds: List = None,           # 兼容参数（忽略）
):
    """生成 local graphs / key frames / global graph（无音频参与）"""
    with open(f"thor_tasks/{folder_name}/task.json") as f:
        task = json.load(f)

    global_pkl = Path(f"state_summary/{folder_name}/global_sg.pkl")
    if global_pkl.exists():
        return  # 已生成

    # 目录准备
    _ensure_dir(Path(f"state_summary/{folder_name}/local_graphs"))
    key_frames: List[int] = []

    prev_graph = SceneGraph(event=None, task=task)
    total_points_dict, bbox3d_dict = {}, {}
    obj_held_prev = None
    cnt, interval = 0, 2
    nav_end_indices = [idx[1] for idx in nav_actions.keys()]

    for step_idx, event in enumerate(events):
        # 丢弃纯导航的中间帧（无音频逻辑）
        if (step_idx + 1) not in interact_actions and (step_idx + 1) not in nav_end_indices:
            cnt += 1
            if cnt % interval == 0:
                continue

        print("[Frame]", step_idx + 1)
        local_sg, total_points_dict, obj_held_prev, bbox3d_dict = get_scene_graph(
            step_idx, event, object_list, total_points_dict, bbox3d_dict, obj_held_prev, task
        )
        print("========================[Current Graph]=====================")
        print(local_sg)

        # 1) 场景图变化触发关键帧
        if local_sg != prev_graph and (step_idx + 1) not in key_frames:
            key_frames.append(step_idx + 1)
            prev_graph = local_sg

        # 2) 交互/导航段落末端触发关键帧
        if ((step_idx + 1) in interact_actions) or ((step_idx + 1) in nav_end_indices):
            if (step_idx + 1) not in key_frames:
                key_frames.append(step_idx + 1)

        with open(f"state_summary/{folder_name}/local_graphs/local_sg_{step_idx}.pkl", "wb") as f:
            pickle.dump(local_sg, f)

    with open(f"state_summary/{folder_name}/L1_key_frames.txt", "w") as f:
        for frame in key_frames:
            f.write(f"{frame}\n")

    # ------------------ 构建 global graph ------------------
    global_sg = SceneGraph(events[-1], task)
    for label in total_points_dict.keys():
        name = get_label_from_object_id(label, events, task)
        if name is not None:
            new_node = SceneGraphNode(
                name=name,
                object_id=label,
                pos3d=bbox3d_dict[label].get_center(),
                corner_pts=np.array(bbox3d_dict[label].get_box_points()),
                pcd=total_points_dict[label],
                global_node=True,
            )
            global_sg.add_node_wo_edge(new_node)

    for label in total_points_dict.keys():
        object_name = label.split("|")[0]
        if object_name in object_list:
            name = get_label_from_object_id(label, events, task)
            if name is not None:
                for node in global_sg.total_nodes:
                    if node.name == name:
                        global_sg.add_node(node)

    global_sg.add_agent()
    with open(global_pkl, "wb") as f:
        pickle.dump(global_sg, f)


def generate_summary(
    folder_name: str,
    events: List,
    nav_actions: Dict[Tuple[int, int], str],
    interact_actions: Dict[int, str],
    WITH_AUDIO: int | bool = False,        # 兼容参数（忽略）
    detected_sounds: List = None,          # 兼容参数（忽略）
):
    """生成两级文字摘要（L1 事件级 / L2 子目标级），不包含任何音频描述。"""
    with open(f"thor_tasks/{folder_name}/task.json") as f:
        task = json.load(f)

    key_frames: List[int] = []
    with open(f"state_summary/{folder_name}/L1_key_frames.txt", "r") as f:
        key_frames = [int(x) for x in f.readlines()]

    # ---------- L1：事件级摘要 ----------
    L1_path = Path(f"state_summary/{folder_name}/state_summary_L1.txt")
    if not L1_path.exists():
        print("[INFO] Start generating event-based summary (audio-free)")
        L1_captions: List[str] = []
        for step_idx, event in enumerate(events):
            local_path = Path(f"state_summary/{folder_name}/local_graphs/local_sg_{step_idx}.pkl")
            if not local_path.exists():
                continue
            if (step_idx + 1) in key_frames:
                caption = ""
                # 动作
                if (step_idx + 1) in interact_actions:
                    caption += f"{convert_step_to_timestep(step=step_idx+1, video_fps=1)}. Action: {interact_actions[step_idx+1]}."
                else:
                    for (lo, hi), nav_name in nav_actions.items():
                        if lo <= (step_idx + 1) <= hi:
                            caption += f"{convert_step_to_timestep(step=step_idx+1, video_fps=1)}. Action: {nav_name}."
                            break
                if not caption:
                    continue

                # 视觉观察
                with local_path.open("rb") as f:
                    local_sg = pickle.load(f)
                scene_text = get_scene_text(local_sg)
                caption += f" Visual observation: {scene_text}\n"

                L1_captions.append(caption)

        _ensure_dir(L1_path.parent)
        L1_path.write_text("".join(L1_captions))
        print("[INFO] Write event-based summary")
    else:
        print("[INFO] Event-based summary already generated")
        L1_captions = L1_path.read_text().splitlines(keepends=True)

    # ---------- L2：子目标级摘要 ----------
    L2_path = Path(f"state_summary/{folder_name}/state_summary_L2.txt")
    if not L2_path.exists():
        print("[INFO] Start generating subgoal-based summary")
        L2_captions: List[str] = []
        for caption in L1_captions:
            step_num = convert_timestep_to_step(caption.split(".")[0], video_fps=1)
            if step_num in interact_actions:
                L2_captions.append(caption.replace("Action", "Goal"))
        L2_path.write_text("".join(L2_captions))
        print("[INFO] Write subgoal-based summary")
    else:
        print("[INFO] Subgoal-based summary already generated")


# -----------------------------------------------------------------------------
# LLM 推理 & 纠错计划生成（保持原逻辑；无音频改动）
# -----------------------------------------------------------------------------
def run_reasoning(folder_name: str, llm_prompter, global_sg: SceneGraph):
    with open(f"thor_tasks/{folder_name}/task.json") as f:
        task = json.load(f)

    reasoning_path = Path(f"state_summary/{folder_name}/reasoning.json")
    if reasoning_path.exists():
        print("[INFO] Reasoning already generated")
        return

    save_dir = _LLM_DIR / folder_name
    _ensure_dir(save_dir)

    prompt_info = json.loads((_LLM_DIR / "prompts.json").read_text())

    # L2 / L1 描述
    L2_captions = (Path(f"state_summary/{folder_name}/state_summary_L2.txt").read_text().splitlines(keepends=True))
    L1_captions = (Path(f"state_summary/{folder_name}/state_summary_L1.txt").read_text().splitlines(keepends=True))

    print(">>> Run step-by-step subgoal-level analysis...")
    selected_caption = ""
    prompt = {}
    reasoning_dict = {}

    for caption in L2_captions:
        print(">>> Verify subgoal...")
        subgoal = caption.split(". ")[1].split(": ")[1].lower()

        prompt["system"] = prompt_info["subgoal-verifier"]["template-system"]
        prompt["user"]   = prompt_info["subgoal-verifier"]["template-user"] \
                               .replace("[SUBGOAL]", subgoal) \
                               .replace("[OBSERVATION]", caption[caption.find("Visual observation"):])

        ans, _ = llm_prompter.query(
            prompt=prompt,
            sampling_params=prompt_info["subgoal-verifier"]["params"],
            save=prompt_info["subgoal-verifier"]["save"],
            save_dir=str(save_dir),
        )
        is_success = int(ans.split(", ")[0] == "Yes")
        if is_success == 0:
            selected_caption = caption
            print(f"[INFO] Failure identified in subgoal [{subgoal}] at {caption.split('.')[0]}")
            break
        else:
            print(f"[INFO] Subgoal [{subgoal}] succeeded!")

    if selected_caption:
        print(">>> Get detailed reasoning from L1...")
        step_name = selected_caption.split(".")[0]
        for cap in L1_captions:
            if step_name in cap:
                action = cap.split(". ")[1].split(": ")[1].lower()
                prev_obs = get_robot_plan(folder_name, step=step_name, with_obs=True)
                prompt_name = "reasoning-execution" if prev_obs else "reasoning-execution-no-history"

                prompt = {
                    "system": prompt_info[prompt_name]["template-system"],
                    "user":   prompt_info[prompt_name]["template-user"]
                                .replace("[ACTION]", action)
                                .replace("[TASK_NAME]", task["name"])
                                .replace("[STEP]", step_name)
                                .replace("[SUMMARY]", prev_obs)
                                .replace("[OBSERVATION]", cap[cap.find("Action"):]),
                }
                ans, _ = llm_prompter.query(
                    prompt=prompt,
                    sampling_params=prompt_info[prompt_name]["params"],
                    save=prompt_info[prompt_name]["save"],
                    save_dir=str(save_dir),
                )
                print("[INFO] Predicted failure reason:", ans)
                reasoning_dict["pred_failure_reason"] = ans

                prompt = {
                    "system": prompt_info["reasoning-execution-steps"]["template-system"],
                    "user":   prompt_info["reasoning-execution-steps"]["template-user"].replace("[FAILURE_REASON]", ans),
                }
                time_steps, _ = llm_prompter.query(
                    prompt=prompt,
                    sampling_params=prompt_info["reasoning-execution-steps"]["params"],
                    save=prompt_info["reasoning-execution-steps"]["save"],
                    save_dir=str(save_dir),
                )
                reasoning_dict["pred_failure_step"] = [t.replace(",", "") for t in time_steps.split(", ")]
                break
    else:
        print(">>> All actions are executed successfully, run plan-level analysis...")
        prompt = {
            "system": prompt_info["reasoning-plan"]["template-system"],
            "user":   prompt_info["reasoning-plan"]["template-user"]
                        .replace("[TASK_NAME]", task["name"])
                        .replace("[SUCCESS_CONDITION]", task["success_condition"])
                        .replace("[CURRENT_STATE]", get_scene_text(global_sg))
                        .replace("[OBSERVATION]", get_robot_plan(folder_name, step=None, with_obs=False)),
        }
        ans, _ = llm_prompter.query(
            prompt=prompt,
            sampling_params=prompt_info["reasoning-plan"]["params"],
            save=prompt_info["reasoning-plan"]["save"],
            save_dir=str(save_dir),
        )
        print("[INFO] Predicted failure reason:", ans)
        reasoning_dict["pred_failure_reason"] = ans

        prompt = {
            "system": prompt_info["reasoning-plan-steps"]["template-system"],
            "user":   prompt_info["reasoning-plan-steps"]["template-user"].replace("[PREV_PROMPT]", prompt["user"] + " " + ans),
        }
        step, _ = llm_prompter.query(
            prompt=prompt,
            sampling_params=prompt_info["reasoning-plan-steps"]["params"],
            save=prompt_info["reasoning-plan-steps"]["save"],
            save_dir=str(save_dir),
        )
        step_str = step.split(" ")[0].rstrip(".,")
        print("[INFO] Predicted failure time steps:", step_str)
        reasoning_dict["pred_failure_step"] = step_str

    reasoning_dict["gt_failure_reason"] = task.get("gt_failure_reason")
    reasoning_dict["gt_failure_step"]   = task.get("gt_failure_step")

    _ensure_dir(reasoning_path.parent)
    reasoning_path.write_text(json.dumps(reasoning_dict, indent=2))


def generate_replan(folder_name: str, llm_prompter, global_sg: SceneGraph, last_event, task_object_list: List[str]):
    with open(f"thor_tasks/{folder_name}/task.json") as f:
        task = json.load(f)

    curr_state = get_scene_text(global_sg)
    print("[INFO] Current state:", curr_state)
    global_object_list = list(set([o["objectType"] for o in last_event.metadata["objects"]]) | set(task_object_list))

    reason = json.loads(Path(f"state_summary/{folder_name}/reasoning.json").read_text())["pred_failure_reason"]

    replan_path = Path(f"state_summary/{folder_name}/replan.json")
    if replan_path.exists():
        print("[INFO] Skipping replan generation")
        plan = "\n".join(json.loads(replan_path.read_text())["original_plan"])
    else:
        prompt_info = json.loads((_LLM_DIR / "prompts.json").read_text())
        prompt = {
            "system": prompt_info["correction"]["template-system"].replace("[PREFIX]", get_replan_prefix()),
            "user":   prompt_info["correction"]["template-user"]
                        .replace("[TASK_NAME]", task["name"])
                        .replace("[PLAN]", get_initial_plan(task["actions"]))
                        .replace("[FAILURE_REASON]", reason)
                        .replace("[CURRENT_STATE]", curr_state)
                        .replace("[SUCCESS_CONDITION]", task["success_condition"]),
        }
        plan, _ = llm_prompter.query(
            prompt=prompt,
            sampling_params=prompt_info["correction"]["params"],
            save=prompt_info["correction"]["save"],
            save_dir=str(_LLM_DIR / folder_name),
        )

    translated = translate_plan(plan, global_object_list, last_event)
    print("========================Translated plan===========================")
    print(translated)

    replan_dict = {
        "original_plan": plan.split("\n"),
        "plan":          translated.split("\n")[:-1],
        "num_steps":     len(translated.split("\n")[:-1]),
    }
    _ensure_dir(replan_path.parent)
    replan_path.write_text(json.dumps(replan_dict, indent=4))
