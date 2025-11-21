# Vid-C.py  —  Scene Graph -> User-facing Explanation (optimized)

import os
import sys
import time
import json
from pathlib import Path


def load_scene_graph(video_root: Path):
    graph_path = video_root / "yolo_scene_graph.json"
    assert graph_path.exists(), f"缺少 {graph_path}，请先运行 Vid-B.py"
    graph = json.loads(graph_path.read_text())
    return graph_path, graph


def build_llm_inputs(graph: dict, top_k_nodes: int = 12, top_k_edges: int = 30):
    """
    把 scene graph 压缩成几行字符串，给 LLM 当输入。
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # 只取前 top_k_nodes 个节点，按出现次数排序
    nodes_sorted = sorted(nodes, key=lambda x: -x.get("count", 0))[:top_k_nodes]
    nodes_for_llm = [f'{n["label"]}×{n.get("count", 1)}' for n in nodes_sorted]

    # 只取前 top_k_edges 条边，按 count 排
    edges_sorted = sorted(edges, key=lambda x: -x.get("count", 0))[:top_k_edges]
    rel_for_llm = [
        f'{e["s"]} —{e["r"]}→ {e["o"]} (×{e.get("count", 1)})'
        for e in edges_sorted
    ]

    return nodes_for_llm, rel_for_llm


def try_init_llm():
    """
    优先使用你仓库里的 LLM 封装 LLM.prompt.LLMPrompter，
    模型默认设成 gpt-4o-mini，用一个很轻量的节流逻辑。
    """
    try:
        from LLM.prompt import LLMPrompter

        class ThrottledPrompter(LLMPrompter):
            def __init__(self, *args, rpm_cap=10, **kwargs):
                # rpm_cap：每分钟最大请求数，调大可以更快
                super().__init__(*args, **kwargs)
                self.min_interval = 60.0 / max(1, rpm_cap)
                self._last = 0.0

            def query(self, *a, **k):
                # 简单节流，避免触发速率限制
                wait = self.min_interval - (time.time() - self._last)
                if wait > 0:
                    time.sleep(wait)
                out = super().query(*a, **k)
                self._last = time.time()
                return out

        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key, "未检测到 OPENAI_API_KEY 环境变量"

        llm = ThrottledPrompter(
            gpt_version="gpt-4o-mini",  # 小模型：更快更便宜
            api_key=api_key,
        )
        print("✅ 使用 LLM.prompt (gpt-4o-mini) 生成解释")
        return llm

    except Exception as e:
        print("[WARN] LLM.prompt 不可用，改用本地兜底解释。原因:", e)
        return None


def build_prompts(nodes_for_llm, rel_for_llm):
    """
    优化过的 prompt，贴近你的实验场景：
      - 桌面、厨务场景
      - 强调人 / 工具 / 容器 / 食物
      - 2–3 条 bullet，第一人称、非技术表达
    """
    sys_prompt = (
        "You are the verbal module of a home-assistant robot. "
        "You are talking to a non-technical user while they watch the robot work at a table.\n"
        "\n"
        "You receive a coarse scene graph (objects and spatial relations) from the camera. "
        "Using ONLY this information, briefly explain what you currently see.\n"
        "\n"
        "Guidelines:\n"
        "  - Speak in FIRST PERSON as the robot (use 'I').\n"
        "  - Use simple, friendly language, no technical terms "
        "    (do NOT mention 'bounding boxes', 'scores', 'graph', or 'detections').\n"
        "  - Focus on: the person, tools (knife, fork, spoon), containers "
        "    (cup, bowl, mug, bottle), and food items (fruit, apple, banana, cereal, milk).\n"
        "  - If a person is NEAR or HOLDING a tool or container, make that the main point.\n"
        "  - If a tool is near a food item on a table (e.g., knife near apple), mention that "
        "    as what I seem to be preparing.\n"
        "  - Keep it SHORT: 2–3 bullet points maximum.\n"
        "  - Do NOT guess about success or failure of the task, and do NOT apologize. "
        "    Just describe what I see and what I seem to be doing.\n"
    )

    user_prompt = (
        "Here is the scene graph summary from my camera.\n\n"
        "Objects (top-k):\n"
        "  - " + "\n  - ".join(nodes_for_llm or ["(none)"]) + "\n\n"
        "Relationships (sampled edges):\n"
        "  - " + "\n  - ".join(rel_for_llm or ["(none)"]) + "\n\n"
        "Please respond with 2–3 bullet points in plain English."
    )

    return sys_prompt, user_prompt


def llm_explain_scene(llm, nodes_for_llm, rel_for_llm, save_dir: Path):
    sys_prompt, user_prompt = build_prompts(nodes_for_llm, rel_for_llm)
    prompt = {"system": sys_prompt, "user": user_prompt}

    # 为了速度，max_tokens 不要太大
    text, _ = llm.query(
        prompt=prompt,
        sampling_params={
            "temperature": 0.2,
            "max_tokens": 160,
        },
        save=False,
        save_dir=str(save_dir),
    )
    return text.strip()


def fallback_explanation(nodes_for_llm, rel_for_llm):
    """
    没有 LLM 时的兜底解释：不用网络，保证流程能跑完。
    """
    if not nodes_for_llm and not rel_for_llm:
        return (
            "• I can’t confidently recognize what is on the table from this view.\n"
            "• I will keep observing and adjusting as I work."
        )

    lines = []
    if nodes_for_llm:
        lines.append("• I can see: " + ", ".join(nodes_for_llm) + ".")
    if rel_for_llm:
        # 只展示少量关系，避免太啰嗦
        lines.append("• Some important spatial relations: " + "; ".join(rel_for_llm[:3]) + ".")
    else:
        lines.append("• I don't detect any strong spatial relations between objects yet.")
    return "\n".join(lines)


def main():
    # ====== 1) 解析视频目录 ======
    if len(sys.argv) > 1:
        video_root = Path(sys.argv[1]).resolve()
    else:
        # 默认：水果切割 demo
        video_root = Path("./camera_demo_fruit").resolve()

    print(f"📂 Vid-C 使用视频目录: {video_root}")
    graph_path, graph = load_scene_graph(video_root)

    # ====== 2) 准备 LLM 输入 ======
    nodes_for_llm, rel_for_llm = build_llm_inputs(graph)

    # ====== 3) 初始化 LLM（如果可用） ======
    llm = try_init_llm()

    # ====== 4) 生成解释 ======
    if llm is not None:
        try:
            text = llm_explain_scene(llm, nodes_for_llm, rel_for_llm, save_dir=video_root.parent)
        except Exception as e:
            print("[WARN] 调用 LLM 失败，改用兜底解释。原因:", e)
            text = fallback_explanation(nodes_for_llm, rel_for_llm)
    else:
        text = fallback_explanation(nodes_for_llm, rel_for_llm)

    # ====== 5) 保存到文本文件 ======
    out_txt = video_root / "yolo_scene_explanation.txt"
    out_txt.write_text(text, encoding="utf-8")

    print("\n=== LLM (user-facing) explanation ===")
    print(text)
    print("\nSaved:", out_txt)


if __name__ == "__main__":
    main()
