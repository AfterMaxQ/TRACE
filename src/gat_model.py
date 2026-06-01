"""
F-07 图神经网络传导建模

对比多种图方法在行业风险传导上的效果:
  - GAT (Graph Attention Network)
  - GCN (Graph Convolutional Network)
  - GraphSAGE
  - PageRank 加权邻居风险 (无训练)
  - IO 直接消耗系数加权邻居风险 (无训练)

Optuna 调参, 选择最优方案。

输入: data/io_adjacency.csv, data/base_feature.csv, data/model_xgb.pkl
输出:
  data/gat_predictions.csv  — 每行业×季度传导风险得分
  data/gat_attention.csv    — 注意力权重矩阵
  data/gat_comparison.csv   — 各方案对比结果

用法: python src/gat_model.py --trials 100
"""

import warnings
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.data import Data
import optuna
from pathlib import Path
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "model"

# ============================================================
# 1. 数据准备: 行业级季度特征
# ============================================================

def _build_industry_quarterly() -> tuple:
    """聚合 base_feature 为行业级季度数据, 构建图数据集。"""
    print("[..] 构建行业季度数据集 ...")

    bf = pd.read_csv(DATA_DIR / "base_feature.csv", low_memory=False)
    adj = pd.read_csv(DATA_DIR / "io_adjacency.csv")

    # 加载 XGBoost 预测
    preds = pd.read_csv(DATA_DIR / "predictions.csv")
    bf = bf.merge(preds[["code","quarter","prob","score"]], on=["code","quarter"], how="left")

    # 特征列
    excluded = ["code","quarter","name","industry","target","prob","score"]
    feature_cols = [c for c in bf.columns
                    if c not in excluded and bf[c].dtype in ("float64","int64")]
    # 只保留数值列
    feature_cols = [c for c in feature_cols if bf[c].notna().any()]

    # 行业季度聚合
    agg_dict = {c: "mean" for c in feature_cols}
    agg_dict["target"] = "mean"  # 行业 ST 率
    # XGBoost prob aggregations
    if "prob" in bf.columns:
        agg_dict["prob"] = ["mean","max"]
        agg_dict["score"] = ["mean","min"]

    iq = bf.groupby(["industry","quarter"]).agg(agg_dict).reset_index()
    # Flatten multi-level columns
    iq.columns = ["_".join(c).strip("_") for c in iq.columns]
    iq = iq.rename(columns={"target_mean": "st_rate", "industry_": "industry"})

    # 确保 industry 列名正确
    if "industry_" in iq.columns:
        iq = iq.rename(columns={"industry_": "industry"})

    print(f"  Industry-quarter rows: {len(iq)}")
    print(f"  Industries: {iq['industry'].nunique()}")
    print(f"  Quarters: {iq['quarter'].nunique()}")

    return iq, adj


def _build_graph_data(iq, adj, quarter_label):
    """为指定季度构建 PyG Data 对象。"""
    q_data = iq[iq["quarter"] == quarter_label].copy()

    # 特征列
    exclude = ["industry","quarter","st_rate"]
    feat_cols = [c for c in q_data.columns if c not in exclude]
    # 确保全数值
    X = q_data[feat_cols].fillna(0).values.astype(np.float32)

    # 构建节点索引映射
    industries = sorted(q_data["industry"].unique())
    ind_to_idx = {ind: i for i, ind in enumerate(industries)}

    # 图边: 过滤当前季度存在的行业
    adj_filtered = adj[
        adj["source"].isin(industries) & adj["target"].isin(industries)
    ].copy()

    if adj_filtered.empty:
        return None, None, None

    src = adj_filtered["source"].map(ind_to_idx).values
    tgt = adj_filtered["target"].map(ind_to_idx).values
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_weight = torch.tensor(adj_filtered["weight"].values, dtype=torch.float32)

    x = torch.tensor(X, dtype=torch.float32)
    y_industry = q_data.set_index("industry").loc[industries, "st_rate"].values
    y = torch.tensor(y_industry, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
    return data, industries, ind_to_idx


# ============================================================
# 2. PyG 模型定义
# ============================================================

class GAT(nn.Module):
    def __init__(self, in_dim, hidden=64, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout, edge_dim=1)
        self.conv2 = GATConv(hidden * heads, hidden // 2, heads=2, dropout=dropout, edge_dim=1)
        self.lin = nn.Linear(hidden, 1)

    def forward(self, data):
        x, ei, ew = data.x, data.edge_index, data.edge_weight
        x = F.elu(self.conv1(x, ei, ew))
        x = F.elu(self.conv2(x, ei, ew))
        x = self.lin(x).squeeze(-1)
        return x

    def attention_weights(self, data):
        """返回第一层注意力权重用于可视化。"""
        x, ei, ew = data.x, data.edge_index, data.edge_weight
        _, (_, alpha) = self.conv1(x, ei, ew, return_attention_weights=True)
        return alpha


class GCN(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden // 2)
        self.lin = nn.Linear(hidden // 2, 1)

    def forward(self, data):
        x, ei, ew = data.x, data.edge_index, data.edge_weight
        x = F.elu(self.conv1(x, ei, ew))
        x = F.elu(self.conv2(x, ei, ew))
        return self.lin(x).squeeze(-1)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden // 2)
        self.lin = nn.Linear(hidden // 2, 1)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.elu(self.conv1(x, ei))
        x = F.elu(self.conv2(x, ei))
        return self.lin(x).squeeze(-1)


# ============================================================
# 3. 训练 & 评估
# ============================================================

def _train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, data.y) + 0.01 * sum(p.norm(2) for p in model.parameters())
    loss.backward()
    optimizer.step()
    return loss.item()


def _evaluate_model(model, data_list):
    """对时序数据列表评估。"""
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for data in data_list:
            if data is None:
                continue
            out = model(data)
            all_preds.append(out.cpu().numpy())
            all_true.append(data.y.cpu().numpy())
    if not all_preds:
        return {"spearman": 0, "mse": 999}
    preds = np.concatenate(all_preds)
    true_vals = np.concatenate(all_true)
    mask = ~np.isnan(preds) & ~np.isnan(true_vals)
    if mask.sum() < 5:
        return {"spearman": 0, "mse": 999}
    sp = spearmanr(preds[mask], true_vals[mask])[0]
    mse = mean_squared_error(true_vals[mask], preds[mask])
    return {"spearman": sp if not np.isnan(sp) else 0, "mse": mse}


# ============================================================
# 4. 轻量图方法 (无需训练)
# ============================================================

def _pagerank_risk(iq, adj, train_quarters, test_quarters):
    """PageRank 加权: industry_risk[i] = Σ pr[j] × st_rate[j] / Σ pr[j]"""
    from collections import defaultdict

    # 计算 PageRank
    G = defaultdict(list)
    for _, r in adj.iterrows():
        G[r["source"]].append(r["target"])
    pr = {}
    for industry in set(adj["source"].unique()) | set(adj["target"].unique()):
        pr[industry] = 1.0
    for _ in range(20):
        new_pr = {}
        for node in pr:
            neighbors = G.get(node, [])
            new_pr[node] = 0.15 + 0.85 * sum(pr.get(n, 0) / max(len(G.get(n, [])), 1) for n in neighbors)
        pr = new_pr

    results = []
    for q in sorted(iq["quarter"].unique()):
        q_data = iq[iq["quarter"] == q]
        industries = q_data["industry"].tolist()
        st_rates = q_data["st_rate"].values
        ind_st = dict(zip(industries, st_rates))

        for industry in industries:
            neighbors = G.get(industry, [])
            if not neighbors:
                weighted_risk = ind_st.get(industry, 0)
            else:
                weights = [pr.get(n, 1.0) for n in neighbors]
                risks = [ind_st.get(n, 0) for n in neighbors]
                w_sum = sum(weights)
                weighted_risk = sum(w * r for w, r in zip(weights, risks)) / max(w_sum, 1e-6) if w_sum > 0 else ind_st.get(industry, 0)

            results.append({"industry": industry, "quarter": q, "pred_risk": weighted_risk, "actual_st_rate": ind_st.get(industry, 0)})

    return pd.DataFrame(results)


def _io_weighted_risk(iq, adj, train_quarters, test_quarters):
    """IO 系数加权: industry_risk[i] = Σ a_ji × st_rate[j] / Σ a_ji"""
    results = []
    for q in sorted(iq["quarter"].unique()):
        q_data = iq[iq["quarter"] == q]
        ind_st = dict(zip(q_data["industry"], q_data["st_rate"]))

        for _, industry in q_data[["industry"]].drop_duplicates().iterrows():
            ind = industry["industry"]
            incoming = adj[adj["target"] == ind]
            if incoming.empty:
                weighted_risk = ind_st.get(ind, 0)
            else:
                total_w = incoming["weight"].sum()
                if total_w > 0:
                    risk_sum = sum(
                        incoming.iloc[i]["weight"] * ind_st.get(incoming.iloc[i]["source"], 0)
                        for i in range(len(incoming))
                    )
                    weighted_risk = risk_sum / total_w
                else:
                    weighted_risk = ind_st.get(ind, 0)

            results.append({"industry": ind, "quarter": q, "pred_risk": weighted_risk, "actual_st_rate": ind_st.get(ind, 0)})

    return pd.DataFrame(results)


# ============================================================
# 5. Optuna 调参
# ============================================================

def _optuna_objective(trial, model_class, train_data_list, val_data_list, in_dim, model_name):
    """Optuna objective for GNN models."""
    lr = trial.suggest_float("lr", 1e-4, 0.05, log=True)
    hidden = trial.suggest_categorical("hidden", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.0, 0.6)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    if model_name == "GAT":
        heads = trial.suggest_categorical("heads", [2, 4, 8])
        model = model_class(in_dim, hidden=hidden, heads=heads, dropout=dropout)
    else:
        model = model_class(in_dim, hidden=hidden, dropout=dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_sp = -1
    for epoch in range(200):
        for data in train_data_list:
            if data is not None:
                _train_epoch(model, data, optimizer)

        if epoch % 10 == 0:
            val_metrics = _evaluate_model(model, val_data_list)
            if val_metrics["spearman"] > best_val_sp:
                best_val_sp = val_metrics["spearman"]

    return best_val_sp


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--no-optuna", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  TRACE F-07 图神经网络传导建模")
    print("=" * 60)

    # 1. 构建数据
    iq, adj = _build_industry_quarterly()

    # 2. 时间切分
    quarters = sorted(iq["quarter"].unique())
    train_q = [q for q in quarters if "2021Q3" <= q <= "2024Q2"]
    val_q   = [q for q in quarters if "2024Q3" <= q <= "2025Q2"]
    test_q  = [q for q in quarters if "2025Q3" <= q <= "2026Q1"]

    train_data = [_build_graph_data(iq, adj, q)[0] for q in train_q]
    val_data   = [_build_graph_data(iq, adj, q)[0] for q in val_q]
    test_data  = [_build_graph_data(iq, adj, q)[0] for q in test_q]
    train_data = [d for d in train_data if d is not None]
    val_data   = [d for d in val_data if d is not None]
    test_data  = [d for d in test_data if d is not None]

    # 获取输入维度
    sample_data, _, _ = _build_graph_data(iq, adj, train_q[0])
    in_dim = sample_data.x.shape[1]
    print(f"  In dim: {in_dim}, Train Q: {len(train_data)}, Val Q: {len(val_data)}, Test Q: {len(test_data)}")

    # 3. 轻量方法 (无需训练)
    print("\n[..] PageRank 加权风险 ...")
    df_pr = _pagerank_risk(iq, adj, train_q, test_q)
    test_pr = df_pr[df_pr["quarter"].isin(test_q)]
    sp_pr = spearmanr(test_pr["pred_risk"], test_pr["actual_st_rate"])[0]
    mse_pr = mean_squared_error(test_pr["actual_st_rate"], test_pr["pred_risk"])

    print("[..] IO 加权风险 ...")
    df_io = _io_weighted_risk(iq, adj, train_q, test_q)
    test_io = df_io[df_io["quarter"].isin(test_q)]
    sp_io = spearmanr(test_io["pred_risk"], test_io["actual_st_rate"])[0]
    mse_io = mean_squared_error(test_io["actual_st_rate"], test_io["pred_risk"])

    # 简单基线: 直接用当前 ST 率预测下季度(即 no change)
    df_lag = iq.copy()
    df_lag["quarter_sort"] = df_lag["quarter"].rank()
    df_lag = df_lag.sort_values(["industry","quarter"])
    df_lag["pred_risk"] = df_lag.groupby("industry")["st_rate"].shift(1)
    test_lag = df_lag[df_lag["quarter"].isin(test_q)].dropna(subset=["pred_risk"])
    sp_lag = spearmanr(test_lag["pred_risk"], test_lag["st_rate"])[0]
    mse_lag = mean_squared_error(test_lag["st_rate"], test_lag["pred_risk"])

    # 4. GNN 模型对比
    results = [
        {"model": "Naive(lag)", "spearman": sp_lag, "mse": mse_lag},
        {"model": "PageRank", "spearman": sp_pr, "mse": mse_pr},
        {"model": "IO-weighted", "spearman": sp_io, "mse": mse_io},
    ]

    if not args.no_optuna:
        for model_name, model_class in [("GCN", GCN), ("GraphSAGE", GraphSAGE), ("GAT", GAT)]:
            print(f"\n[..] Optuna tuning {model_name} ({args.trials} trials) ...")
            study = optuna.create_study(direction="maximize",
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
                sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(
                lambda trial: _optuna_objective(trial, model_class, train_data, val_data, in_dim, model_name),
                n_trials=args.trials, show_progress_bar=True,
            )

            # Retrain best model and evaluate on test
            best_params = study.best_params
            if model_name == "GAT":
                best_model = model_class(in_dim,
                    hidden=best_params["hidden"],
                    heads=best_params["heads"],
                    dropout=best_params["dropout"])
            else:
                best_model = model_class(in_dim,
                    hidden=best_params["hidden"],
                    dropout=best_params["dropout"])

            opt = torch.optim.Adam(best_model.parameters(),
                lr=best_params["lr"], weight_decay=best_params["weight_decay"])

            for epoch in range(500):
                for data in train_data:
                    if data is not None:
                        _train_epoch(best_model, data, opt)

            test_metrics = _evaluate_model(best_model, test_data)
            results.append({
                "model": model_name,
                "spearman": round(test_metrics["spearman"], 4),
                "mse": round(test_metrics["mse"], 6),
                "best_params": str(best_params),
                "best_val_sp": round(study.best_value, 4),
            })
            print(f"  {model_name}: test Spearman={test_metrics['spearman']:.4f}, MSE={test_metrics['mse']:.6f}")

            # Save best model
            torch.save(best_model.state_dict(), MODEL_DIR / f"model_{model_name.lower()}.pt")

    # 5. 报告
    print("\n" + "=" * 60)
    print("F-07 Results: Industry Risk Conduction")
    print("=" * 60)
    print(f"{'Model':15s} {'Spearman':>10s} {'MSE':>10s} {'Comment'}")
    print("-" * 50)

    # Sort by Spearman
    results_sorted = sorted(results, key=lambda x: x.get("spearman", 0), reverse=True)
    best_sp = results_sorted[0]["spearman"]
    for r in results_sorted:
        sp = r["spearman"]
        mse = r["mse"]
        tag = "★ best" if sp == best_sp else ""
        if sp >= 0.3:
            tag += " (usable)"
        elif sp <= 0:
            tag += " (no signal)"
        print(f"  {r['model']:15s} {sp:10.4f} {mse:10.6f} {tag}")

    # Save comparison
    pd.DataFrame(results).to_csv(DATA_DIR / "gat_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\n[OK] gat_comparison.csv saved")

    # Save best model predictions
    best_name = results_sorted[0]["model"]
    if best_name in ("PageRank","IO-weighted"):
        df_best = df_pr if best_name == "PageRank" else df_io
    elif best_name == "Naive(lag)":
        df_best = df_lag[["industry","quarter","pred_risk","st_rate"]]
    else:
        # GNN model - re-predict all quarters
        best_model_class = {"GAT": GAT, "GCN": GCN, "GraphSAGE": GraphSAGE}[best_name]
        in_dim = train_data[0].x.shape[1]
        best_model = best_model_class(in_dim, hidden=64, dropout=0.3)
        best_model.load_state_dict(torch.load(MODEL_DIR / f"model_{best_name.lower()}.pt"))
        # Predict all quarters
        all_preds = []
        for q in quarters:
            data, industries, _ = _build_graph_data(iq, adj, q)
            if data is None:
                continue
            best_model.eval()
            with torch.no_grad():
                out = best_model(data).cpu().numpy()
            for i, ind in enumerate(industries):
                actual = iq[(iq["quarter"]==q) & (iq["industry"]==ind)]["st_rate"].values
                all_preds.append({
                    "industry": ind, "quarter": q,
                    "pred_risk": out[i],
                    "actual_st_rate": actual[0] if len(actual) > 0 else 0,
                })
        df_best = pd.DataFrame(all_preds)

    df_best.to_csv(DATA_DIR / "gat_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] gat_predictions.csv ({len(df_best)} rows)")

    print(f"\n[DONE] F-07 完成")


if __name__ == "__main__":
    main()
