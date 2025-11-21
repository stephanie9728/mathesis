# Vid-D：Scene Graph 可视化（PNG + HTML）
import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# ========= 1. 加载 Scene Graph JSON =========
# （自动找前面 Vid-B 产出的文件）
import sys
from pathlib import Path

VIDEO_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./camera_demo_fruit")
VIDEO_ROOT = VIDEO_ROOT.resolve()

GRAPH_PATH = VIDEO_ROOT / "yolo_scene_graph.json"
assert GRAPH_PATH.exists(), f"缺少 {GRAPH_PATH}，请先运行 Vid-B"


graph_data = json.loads(GRAPH_PATH.read_text())
edges = graph_data.get("edges", [])
nodes = graph_data.get("nodes", [])
print(f"Loaded scene graph with {len(nodes)} nodes, {len(edges)} edges")

# ========= 2. 构建 MultiDiGraph =========
G = nx.MultiDiGraph()
for n in nodes:
    G.add_node(n["label"], count=n.get("count", 1))

for e in edges:
    G.add_edge(
        e["s"], e["o"],
        relation=e.get("r") or e.get("relation", ""),
        count=e.get("count", 1)
    )

# ========= 3. 辅助函数：聚合多重边 =========
def _aggregate_multiedges(G: nx.MultiDiGraph, topk=3, sep=' | '):
    """
    把 MultiDiGraph 聚合成 DiGraph：
      - 同一对 (u,v) 的多条边合到一条，weight = count 的和
      - label = 出现次数最高的 topk 个关系，例如 "near×12 | left_of×3"
    """
    H = nx.DiGraph()
    for n, dat in G.nodes(data=True):
        H.add_node(n, **dat)

    for u, v, dat in G.edges(data=True):
        rel = dat.get("relation") or dat.get("r") or "rel"
        cnt = int(dat.get("count", 1))
        if H.has_edge(u, v):
            H[u][v]["_rels"].append((rel, cnt))
            H[u][v]["weight"] += cnt
        else:
            H.add_edge(u, v, weight=cnt, _rels=[(rel, cnt)])

    edge_labels = {}
    max_w = 1
    for u, v in H.edges():
        rels = sorted(H[u][v]["_rels"], key=lambda x: -x[1])[:topk]
        edge_labels[(u, v)] = sep.join([f"{r}×{c}" for r, c in rels])
        max_w = max(max_w, H[u][v]["weight"])
        del H[u][v]["_rels"]

    return H, edge_labels, max_w


# ========= 4. 绘制 PNG =========
def draw_png(G: nx.MultiDiGraph, out_path):
    H, edge_labels, max_w = _aggregate_multiedges(G, topk=3)
    plt.figure(figsize=(8, 6), dpi=150)
    pos = nx.spring_layout(H, seed=42, k=0.7)

    nx.draw_networkx_nodes(
        H, pos,
        node_size=900,
        node_color="#EFF6FF",
        edgecolors="#2563EB",
        linewidths=1.5
    )
    nx.draw_networkx_labels(H, pos, font_size=10, font_weight="bold")

    widths = [1.0 + 4.0 * (H[u][v]["weight"] / max_w) for u, v in H.edges()]
    nx.draw_networkx_edges(
        H, pos,
        arrows=True,
        arrowstyle='-|>',
        width=widths,
        alpha=0.9,
        connectionstyle='arc3,rad=0.15'
    )

    nx.draw_networkx_edge_labels(
        H, pos,
        edge_labels=edge_labels,
        font_size=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
    )

    plt.axis('off')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("✅ PNG saved ->", out_path)


# ========= 5. 绘制 HTML（PyVis） =========
def draw_html(G: nx.MultiDiGraph, out_path):
    try:
        from pyvis.network import Network
    except ImportError:
        print("⚠️ 未安装 pyvis，可运行 `pip install pyvis` 启用 HTML 可视化")
        return

    H, edge_labels, max_w = _aggregate_multiedges(G, topk=3)
    net = Network(height="720px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-12000, central_gravity=0.2, spring_length=150, spring_strength=0.01)

    for n in H.nodes():
        count = H.nodes[n].get("count", 1)
        net.add_node(n, label=f"{n}×{count}",
                     color={"background": "#EFF6FF", "border": "#2563EB"},
                     borderWidth=2)

    for u, v, dat in H.edges(data=True):
        w = dat.get("weight", 1)
        lbl = edge_labels.get((u, v), "")
        net.add_edge(u, v, label=lbl, value=w, arrows="to", smooth=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    net.show(str(out_path))
    print("✅ HTML saved ->", out_path)


# ========= 6. 执行绘制 =========
viz_dir = VIDEO_ROOT / "viz"
png_path = viz_dir / "scene_graph.png"
html_path = viz_dir / "scene_graph.html"

draw_png(G, png_path)
draw_html(G, html_path)

print("\n🎉 Scene Graph Visualization Complete!")
print("   PNG:", png_path)
print("   HTML:", html_path)
