"""
F-05 知识图谱模块

从非竞争型投入产出表提取 42×42 中间使用矩阵，展开到 111 个申万行业，
构建有向加权传导网络，计算图拓扑特征。

输入:
  data/42部门投入产出表2017-2023/2023年非竞争型投入产出表.xlsx
  data/industry_mapping.csv

输出:
  data/io_adjacency.csv   — 边列表 (source, target, weight), 供 GAT + 可视化
  data/io_nodes.csv        — 节点元数据 (111 申万行业)
  data/graph_features.csv  — 5 个图统计特征 (按 申万行业)

用法:
  python src/knowledge_graph.py
"""

import warnings
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
IO_TABLE_DIR = DATA_DIR / "42部门投入产出表2017-2023"
IO_TABLE_FILE = IO_TABLE_DIR / "2023年非竞争型投入产出表.xlsx"

# IO 部门编号 → 名称 (用于调试)
IO_NAMES = {
    1:"农林牧渔", 2:"煤炭采选", 3:"石油天然气", 4:"金属矿采选", 5:"非金属矿采选",
    6:"食品烟草", 7:"纺织品", 8:"纺织服装鞋帽", 9:"木材家具", 10:"造纸印刷文教",
    11:"石油炼焦核燃料", 12:"化学产品", 13:"非金属矿物", 14:"金属冶炼压延",
    15:"金属制品", 16:"通用设备", 17:"专用设备", 18:"交通运输设备",
    19:"电气机械器材", 20:"通信电子设备", 21:"仪器仪表", 22:"其他制造",
    23:"金属制品修理", 24:"电力热力", 25:"燃气", 26:"水",
    27:"建筑", 28:"批发零售", 29:"交运仓储邮政", 30:"住宿餐饮",
    31:"信息技术服务", 32:"金融", 33:"房地产", 34:"租赁商务", 35:"研发",
    36:"综合技术", 37:"水利环境", 38:"居民服务", 39:"教育", 40:"卫生社工",
    41:"文化体育娱乐", 42:"公共管理",
}


def _extract_matrix(excel_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """从非竞争型 IO 表提取 42×42 国产中间使用矩阵 + 总产出向量。

    返回:
      matrix: (42, 42) 国产中间使用矩阵 (万元)
      output_vec: (42,) 总产出向量 (万元)
    """
    if excel_path is None:
        excel_path = IO_TABLE_FILE

    df = pd.read_excel(excel_path, sheet_name=0, header=None)
    df.columns = [str(c) for c in range(df.shape[1])]

    # 国产中间使用矩阵: rows 5-46 (0-indexed), cols 4-45
    matrix = np.zeros((42, 42))
    for i in range(42):
        row_idx = 5 + i
        for j in range(42):
            col_idx = 4 + j
            try:
                v = float(df.iloc[row_idx, col_idx])
                matrix[i, j] = v if not np.isnan(v) else 0.0
            except (ValueError, TypeError):
                matrix[i, j] = 0.0

    # 总产出: col 53, rows 5-46
    output_vec = np.zeros(42)
    for i in range(42):
        row_idx = 5 + i
        try:
            v = float(df.iloc[row_idx, 53])
            output_vec[i] = v if not np.isnan(v) else 0.0
        except (ValueError, TypeError):
            output_vec[i] = 0.0

    return matrix, output_vec


def _load_io_to_shenwan() -> dict[int, list[str]]:
    """从 industry_mapping.csv 加载 IO 部门 → 申万行业列表。"""
    mp = pd.read_csv(DATA_DIR / "industry_mapping.csv")
    # 过滤空值和 NaN, 统一为 str
    mp = mp[mp["shenwan_industry"].notna() & (mp["shenwan_industry"] != "")]
    mp["shenwan_industry"] = mp["shenwan_industry"].astype(str)
    io_to_sw = mp.groupby("io_sector")["shenwan_industry"].apply(list).to_dict()
    return io_to_sw


def _build_shenwan_graph(
    matrix: np.ndarray,
    output_vec: np.ndarray,
    io_to_sw: dict[int, list[str]],
) -> pd.DataFrame:
    """42×42 直接消耗系数 → 111×111 申万行业邻接矩阵。

    步骤:
      1. 计算直接消耗系数 a_ij = x_ij / GO_j
      2. IO → 申万展开 (一个 IO 对应多个申万 → 复制边；多个 IO → 同一申万 → 取均值)

    返回:
      DataFrame (111, 111) index=columns=申万行业, values=直接消耗系数
    """
    # Step 1: 直接消耗系数
    a_matrix = np.zeros((42, 42))
    for j in range(42):
        if output_vec[j] > 0:
            a_matrix[:, j] = matrix[:, j] / output_vec[j]

    # Step 2: 构建 IO→申万 展开映射
    # 生成每个申万行业对应的 IO 部门列表 (1:1 or N:1)
    sw_to_io_list: dict[str, list[int]] = {}
    for io_id, sw_list in io_to_sw.items():
        for sw in sw_list:
            if sw:
                sw_to_io_list.setdefault(sw, []).append(io_id)

    sw_list_sorted = sorted(sw_to_io_list.keys())
    n_sw = len(sw_list_sorted)
    sw_to_idx = {sw: i for i, sw in enumerate(sw_list_sorted)}

    # 初始化 111×111 邻接矩阵 (按申万行业)
    adj_sw = np.zeros((n_sw, n_sw))

    # 每对 (IO i, IO j) 的 a_ij 分配到对应的 (申万 I, 申万 J)
    # 如果 IO_i 映射到多个申万, IO_j 也映射到多个申万 → 产生 n×m 条边
    for io_i in range(42):
        sw_i_list = io_to_sw.get(io_i + 1, [])
        sw_i_list = io_to_sw.get(io_i + 1, [])
        if not sw_i_list:
            continue
        for io_j in range(42):
            a_val = a_matrix[io_i, io_j]
            if a_val == 0:
                continue
            sw_j_list = io_to_sw.get(io_j + 1, [])
            if not sw_j_list:
                continue
            for sw_i in sw_i_list:
                for sw_j in sw_j_list:
                    if sw_i == sw_j:
                        continue  # 跳过多对一产生的申万自环
                    i_idx = sw_to_idx[sw_i]
                    j_idx = sw_to_idx[sw_j]
                    adj_sw[i_idx, j_idx] += a_val

    # 多个 IO→同一申万: 取均值
    for sw in sw_list_sorted:
        io_ids = sw_to_io_list.get(sw, [])
        if len(io_ids) > 1:
            idx = sw_to_idx[sw]
            # 对应列的传入系数取均值
            adj_sw[:, idx] /= len(io_ids)
            # 对应行的传出系数取均值
            adj_sw[idx, :] /= len(io_ids)

    result = pd.DataFrame(adj_sw, index=sw_list_sorted, columns=sw_list_sorted)
    return result


def _compute_graph_features(adj: pd.DataFrame) -> pd.DataFrame:
    """对 111×111 邻接矩阵计算 5 个图统计特征。

    返回:
      DataFrame: shenwan_industry + pagerank + weighted_out_degree
                 + weighted_in_degree + betweenness + clustering
    """
    G = nx.from_pandas_adjacency(adj, create_using=nx.DiGraph)

    # PageRank (有向图版本, 处理悬挂节点)
    pr = nx.pagerank(G, alpha=0.85, weight="weight")

    # 加权度
    out_deg = {n: G.out_degree(n, weight="weight") for n in G.nodes()}
    in_deg = {n: G.in_degree(n, weight="weight") for n in G.nodes()}

    # Betweenness (边权重取倒数 → 成本/距离, 权重越小路径越短)
    # 直接消耗系数大 = 传导路径短 = 成本低
    G_inv = G.copy()
    for u, v, d in G_inv.edges(data=True):
        w = d["weight"]
        d["weight"] = 1.0 / max(w, 1e-12)
    bc = nx.betweenness_centrality(G_inv, weight="weight", normalized=True)

    # Clustering (无向化取均值 → 只算有向比较慢且含义不明显)
    G_undirected = G.to_undirected()
    cc = nx.clustering(G_undirected, weight="weight")

    rows = []
    for node in adj.index:
        rows.append({
            "shenwan_industry": node,
            "pagerank": round(pr.get(node, 0), 6),
            "weighted_out_degree": round(out_deg.get(node, 0), 6),
            "weighted_in_degree": round(in_deg.get(node, 0), 6),
            "betweenness": round(bc.get(node, 0), 6),
            "clustering": round(cc.get(node, 0), 6),
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("  TRACE F-05 知识图谱构建")
    print("=" * 60)

    # 1. 提取 IO 矩阵
    print("[..] 提取 2023 非竞争型 IO 表...")
    matrix, output_vec = _extract_matrix()
    total_intermediate = matrix.sum(axis=1)
    valid_go = output_vec > 0

    print(f"      矩阵: {matrix.shape}, 非零元素: {(matrix > 0).sum()}")
    print(f"      国产中间投入合计: {total_intermediate.sum()/1e8:.1f} 亿元")
    print(f"      总产出 > 0 的部门: {valid_go.sum()}/42")

    # 2. 加载映射
    io_to_sw = _load_io_to_shenwan()
    n_io = len(io_to_sw)
    print(f"\n[..] IO→申万映射: {n_io} IO部门")

    # 3. 构建申万邻接矩阵
    print("[..] 构建申万行业邻接矩阵...")
    adj = _build_shenwan_graph(matrix, output_vec, io_to_sw)
    print(f"      申万图: {adj.shape[0]} × {adj.shape[1]}")
    n_edges = (adj.values > 0).sum()
    max_weight = adj.values.max()
    print(f"      有向边: {n_edges}, 最大直接消耗系数: {max_weight:.4f}")

    # 4. 输出边列表
    edges = []
    for source in adj.index:
        for target in adj.columns:
            w = adj.loc[source, target]
            if w > 0:
                edges.append({"source": source, "target": target, "weight": round(w, 6)})

    edge_df = pd.DataFrame(edges).sort_values("weight", ascending=False)
    edge_path = DATA_DIR / "io_adjacency.csv"
    edge_df.to_csv(edge_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] io_adjacency.csv: {len(edge_df)} 条边")
    print(f"      Top 5 边:")
    for _, row in edge_df.head(5).iterrows():
        print(f"        {row['source']} → {row['target']}: {row['weight']:.4f}")

    # 5. 输出节点表
    nodes = pd.DataFrame({
        "shenwan_industry": sorted(adj.index),
    })
    nodes.to_csv(DATA_DIR / "io_nodes.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] io_nodes.csv: {len(nodes)} 个节点")

    # 6. 计算图特征
    print("[..] 计算图拓扑特征...")
    features = _compute_graph_features(adj)
    feat_path = DATA_DIR / "graph_features.csv"
    features.to_csv(feat_path, index=False, encoding="utf-8-sig")
    print(f"[OK] graph_features.csv: {len(features)} × {len(features.columns)}")

    # 摘要
    print(f"\n      特征摘要:")
    for col in ["pagerank", "weighted_out_degree", "weighted_in_degree",
                "betweenness", "clustering"]:
        col_data = features[col]
        print(f"        {col}: [{col_data.min():.4f}, {col_data.max():.4f}], "
              f"mean={col_data.mean():.4f}, top={features.loc[col_data.idxmax(), 'shenwan_industry']}")

    print(f"\n[DONE] F-05 知识图谱构建完成")


if __name__ == "__main__":
    main()
