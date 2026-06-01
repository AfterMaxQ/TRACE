"""
F-08 模型融合模块

融合三路信号: XGBoost基础分 + 行业传导修正 + 舆情修正
网格搜索最优权重组合, 输出融合后风险评分。

输入: data/predictions.csv, data/base_feature.csv, data/io_adjacency.csv
输出:
  data/fusion_scores.csv  — 融合后评分
  data/fusion_grid.csv    — 网格搜索结果

用法: python src/fusion.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

DATA_DIR = Path(__file__).parent.parent / "data"


def _build_xgb_probs(bf):
    """为所有季度生成 XGBoost 预测概率 (使 val/test 都有 prob)。"""
    print("[..] 生成全量 XGBoost 预测 ...")

    KEY = ["code", "quarter", "name", "industry", "target"]
    feature_cols = [c for c in bf.columns if c not in KEY]

    X = bf[feature_cols]
    y = bf["target"]

    train_mask = (bf["quarter"] >= "2021Q3") & (bf["quarter"] <= "2024Q2")
    val_mask   = (bf["quarter"] >= "2024Q3") & (bf["quarter"] <= "2025Q2")

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[val_mask], y[val_mask]
    sw = (len(y_tr) - y_tr.sum()) / y_tr.sum()

    model = xgb.XGBClassifier(
        max_depth=7, learning_rate=0.1, n_estimators=300,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=10,
        scale_pos_weight=sw, objective="binary:logistic", eval_metric="auc",
        early_stopping_rounds=20, random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    bf["prob"] = model.predict_proba(X)[:, 1]
    return bf


def _compute_conduction(bf, adj):
    """行业传导修正 = NaiveLag(行业ST率) * IO暴露度。"""
    iq = bf.groupby(["industry", "quarter"])["target"].mean().reset_index()
    iq.columns = ["industry", "quarter", "st_rate"]
    iq = iq.sort_values(["industry", "quarter"])
    iq["st_rate_lag1"] = iq.groupby("industry")["st_rate"].shift(1)

    # IO暴露度
    exposure = {}
    for ind in bf["industry"].unique():
        d = bf[bf["industry"] == ind]
        out_d = d["weighted_out_degree"].mean()
        in_d = d["weighted_in_degree"].mean()
        total = out_d + in_d
        exposure[ind] = (in_d / total) if total > 0 else 0.5

    iq["io_exposure"] = iq["industry"].map(exposure).fillna(0.5)
    iq["conduction"] = (iq["st_rate_lag1"].fillna(0) * iq["io_exposure"]).clip(0, 1)
    return iq[["industry", "quarter", "conduction"]]


def _compute_sentiment(bf):
    """舆情修正 = neg_ratio 主导 + compliance_ratio 放大。"""
    s = bf[["code", "quarter", "neg_ratio", "compliance_ratio"]].copy()
    s["sentiment_adj"] = (
        s["neg_ratio"].fillna(0) * 2.0 +
        s["compliance_ratio"].fillna(0) * 3.0
    ).clip(0, 3)
    return s[["code", "quarter", "sentiment_adj"]]


def main():
    print("=" * 60)
    print("  TRACE F-08 模型融合 (Grid Search)")
    print("=" * 60)

    # 1. Load & build signals
    bf, adj = pd.read_csv(DATA_DIR / "base_feature.csv", low_memory=False), \
              pd.read_csv(DATA_DIR / "io_adjacency.csv")

    # XGBoost probs for all quarters
    bf = _build_xgb_probs(bf)

    # Conduction & sentiment
    cond = _compute_conduction(bf, adj)
    sent = _compute_sentiment(bf)

    # Merge
    df = bf[["code", "quarter", "industry", "target", "prob"]].merge(
        cond, on=["industry", "quarter"], how="left"
    ).merge(sent, on=["code", "quarter"], how="left")
    df["conduction"] = df["conduction"].fillna(0)
    df["sentiment_adj"] = df["sentiment_adj"].fillna(0)

    # Normalize auxiliary signals to [0, 1]
    cmax, cmin = df["conduction"].max(), df["conduction"].min()
    df["conduction"] = (df["conduction"] - cmin) / (cmax - cmin + 1e-8)
    smax, smin = df["sentiment_adj"].max(), df["sentiment_adj"].min()
    df["sentiment_adj"] = (df["sentiment_adj"] - smin) / (smax - smin + 1e-8)

    # 2. Time split
    tr_m = (df["quarter"] >= "2021Q3") & (df["quarter"] <= "2024Q2")
    va_m = (df["quarter"] >= "2024Q3") & (df["quarter"] <= "2025Q2")
    te_m = (df["quarter"] >= "2025Q3") & (df["quarter"] <= "2026Q1")
    df_tr, df_va, df_te = df[tr_m], df[va_m], df[te_m]
    print(f"Split: Train={len(df_tr):,}  Val={len(df_va):,}  Test={len(df_te):,}")

    # 3. Grid search
    print("\n[..] Grid search ...")

    # Signals correlation
    print(f"  Signal correlations (Val):")
    for a, b in [("prob","conduction"), ("prob","sentiment_adj"), ("conduction","sentiment_adj")]:
        sp = spearmanr(df_va[a], df_va[b])[0]
        print(f"    {a} vs {b}: r={sp:.4f}")

    w1_vals = [round(x,2) for x in np.arange(0.40, 1.01, 0.05)]
    w2_vals = [round(x,2) for x in np.arange(0.00, 0.51, 0.05)]

    results = []
    best_auc = 0
    best_w = None

    for w1 in w1_vals:
        for w2 in w2_vals:
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < -0.01 or w3 > 0.31:
                continue  # w3 must be in [0, 0.3]

            fused_va = (w1 * df_va["prob"] + w2 * df_va["conduction"] +
                       w3 * df_va["sentiment_adj"])
            fused_te = (w1 * df_te["prob"] + w2 * df_te["conduction"] +
                       w3 * df_te["sentiment_adj"])

            try:
                auc_va = roc_auc_score(df_va["target"], fused_va)
                auc_te = roc_auc_score(df_te["target"], fused_te)
                pr_te = average_precision_score(df_te["target"], fused_te)
            except ValueError:
                continue

            results.append({
                "w_xgb": w1, "w_cond": w2, "w_sent": max(w3, 0),
                "auc_val": round(auc_va, 4), "auc_test": round(auc_te, 4),
                "pr_test": round(pr_te, 4),
            })

            if auc_va > best_auc:
                best_auc = auc_va
                best_w = (w1, w2, max(w3, 0))

    if not results:
        print("[!!] No valid combinations")
        return

    grid = pd.DataFrame(results).sort_values("auc_val", ascending=False)

    # 4. Evaluate best
    w1, w2, w3 = best_w
    df_te = df_te.copy()
    df_te["fused"] = w1 * df_te["prob"] + w2 * df_te["conduction"] + w3 * df_te["sentiment_adj"]
    auc_fused = roc_auc_score(df_te["target"], df_te["fused"])
    auc_xgb = roc_auc_score(df_te["target"], df_te["prob"])
    pr_fused = average_precision_score(df_te["target"], df_te["fused"])
    pr_xgb = average_precision_score(df_te["target"], df_te["prob"])

    # 5. Report
    print("\n" + "=" * 60)
    print("F-08 融合结果")
    print("=" * 60)
    print(f"  最优权重:   w_xgb={w1:.2f}  w_cond={w2:.2f}  w_sent={w3:.2f}")
    print(f"  XGBoost:    AUC={auc_xgb:.4f}  PR={pr_xgb:.4f}")
    print(f"  Fused:      AUC={auc_fused:.4f}  PR={pr_fused:.4f}")
    print(f"  Delta:      AUC={auc_fused-auc_xgb:+.4f}  PR={pr_fused-pr_xgb:+.4f}")

    print(f"\n  Top 20 权重组合 (按 Val AUC):")
    print(f"  {'w_xgb':>7s} {'w_cond':>7s} {'w_sent':>7s} {'AUC_val':>9s} {'AUC_test':>9s} {'PR_test':>9s}")
    print(f"  {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*9} {'-'*9}")
    for _, r in grid.head(20).iterrows():
        flag = " <--" if r["w_xgb"]==w1 and r["w_cond"]==w2 else ""
        print(f"  {r['w_xgb']:7.2f} {r['w_cond']:7.2f} {r['w_sent']:7.2f} "
              f"{r['auc_val']:9.4f} {r['auc_test']:9.4f} {r['pr_test']:9.4f}{flag}")

    # 6. Save
    grid.to_csv(DATA_DIR / "fusion_grid.csv", index=False, encoding="utf-8-sig")

    df_all = df.copy()
    w1_a, w2_a, w3_a = best_w
    df_all["fused_score"] = w1_a * df_all["prob"] + w2_a * df_all["conduction"] + w3_a * df_all["sentiment_adj"]
    # 概率 → 评分卡分数 (0-1000), PDO=50, 基准600
    p_clip = np.clip(df_all["fused_score"].values, 1e-6, 1 - 1e-6)
    df_all["score"] = (600 - 50 * np.log(p_clip / (1 - p_clip)) / np.log(2)).astype(int)
    df_all[["code","quarter","prob","conduction","sentiment_adj","fused_score","score","target"]] \
        .to_csv(DATA_DIR / "fusion_scores.csv", index=False, encoding="utf-8-sig")

    print(f"\n[OK] fusion_grid.csv ({len(grid)} combinations)")
    print(f"[OK] fusion_scores.csv ({len(df_all)} rows)")
    print(f"\n[DONE] F-08 融合完成")


if __name__ == "__main__":
    main()
