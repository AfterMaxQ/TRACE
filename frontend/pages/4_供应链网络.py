"""行业供应链传导网络"""
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from utils import inject_css, render_sidebar, load_io_adjacency, load_graph_features

st.set_page_config(page_title="供应链网络", layout="wide")
inject_css()
render_sidebar()

st.header("行业投入产出传导网络")

SECTOR_COLORS = {
    "制造": "#3B82F6", "资源": "#F59E0B", "服务": "#10B981",
    "科技": "#8B5CF6", "金融地产": "#EF4444",
}

def classify(industry):
    s = str(industry)
    if any(k in s for k in ["化学","钢铁","电子","汽车","机械","纺织","食品","制造","设备","医药"]):
        return "制造"
    if any(k in s for k in ["煤炭","石油","采矿","电力","农业","有色","矿","金属"]):
        return "资源"
    if any(k in s for k in ["计算机","通信","半导体","软件","互联网","IT"]):
        return "科技"
    if any(k in s for k in ["银行","地产","券商","保险","金融","房地产"]):
        return "金融地产"
    return "服务"

df_io = load_io_adjacency()
gf = load_graph_features()

if df_io.empty:
    st.warning("io_adjacency.csv 未找到")
    st.stop()

# Filter edges
min_w = st.slider("最小IO系数 (越大边越少)", 0.005, 0.10, 0.02, 0.005)
df_f = df_io[df_io["weight"] >= min_w]

with st.spinner("渲染网络图..."):
    net = Network(height="650px", width="100%", bgcolor="#F8FAFC",
                  font_color="#1E293B", directed=True)

    # Node sizes from PageRank
    if not gf.empty:
        pr_map = dict(zip(gf["shenwan_industry"], gf["pagerank"]))
    else:
        pr_map = {}

    nodes_added = set()
    for _, row in df_f.iterrows():
        src, tgt, w = row["source"], row["target"], row["weight"]
        for node in [src, tgt]:
            if node not in nodes_added:
                pr = pr_map.get(node, 0.005)
                size = max(8, pr * 800)
                color = SECTOR_COLORS.get(classify(node), "#CBD5E1")
                net.add_node(str(node), label=str(node)[:12],
                    title=f"行业: {node}<br>PageRank: {pr:.4f}",
                    size=size, color=color)
                nodes_added.add(node)
        net.add_edge(str(src), str(tgt), value=float(w),
            title=f"IO系数: {w:.4f}", color="#CBD5E1", arrows="to")

    net.set_options("""
    {
      "physics": {
        "barnesHut": {"gravitationalConstant": -3000, "centralGravity": 0.3,
                       "springLength": 120, "springConstant": 0.04},
        "minVelocity": 0.75
      }
    }
    """)

    net.save_graph(str(Path(__file__).parent.parent / "network_temp.html"))
    with open(str(Path(__file__).parent.parent / "network_temp.html"), "r", encoding="utf-8") as f:
        html = f.read()

st.markdown(
    '<span style="color:#3B82F6">■ 制造业</span> '
    '<span style="color:#F59E0B">■ 资源</span> '
    '<span style="color:#10B981">■ 服务业</span> '
    '<span style="color:#8B5CF6">■ 科技</span> '
    '<span style="color:#EF4444">■ 金融地产</span> &nbsp; | &nbsp; '
    '拖拽/滚轮缩放/点击高亮邻居',
    unsafe_allow_html=True)

st.caption(f"显示 {len(df_f)} 条边 (共 {len(df_io)} 条), {len(nodes_added)} 个行业节点")
components.html(html, height=680)
