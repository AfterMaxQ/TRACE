"""
F-06 违约预测建模模块 (Optuna 优化版)

XGBoost 二分类 + 时间序列切分 + Optuna 贝叶斯调参 + SHAP + 评分卡。

输入: data/base_feature.csv
输出:
  data/model_xgb.pkl         — 最优模型
  data/feature_importance.csv — 特征重要性
  data/predictions.csv        — 测试集预测结果
  data/optuna_study.pkl       — Optuna 调参历史

用法:
  python src/modeling.py
  python src/modeling.py --trials 300 --no-optuna  # 跳过调参, 用预设参数
"""

import warnings
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "model"
KEY_COLS = ["code", "quarter", "name", "industry", "target"]

# ============================================================
# 1. 数据加载 & 时间切分
# ============================================================

def _load_and_split() -> tuple:
    """加载 base_feature, 按时间切分 train/val/test。"""
    print("[..] 加载 base_feature.csv ...")
    df = pd.read_csv(DATA_DIR / "base_feature.csv")

    feature_cols = [c for c in df.columns if c not in KEY_COLS]
    X = df[feature_cols].copy()
    y = df["target"].copy()

    train_mask = (df["quarter"] >= "2021Q3") & (df["quarter"] <= "2024Q2")
    val_mask   = (df["quarter"] >= "2024Q3") & (df["quarter"] <= "2025Q2")
    test_mask  = (df["quarter"] >= "2025Q3") & (df["quarter"] <= "2026Q1")

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"  Train: {len(X_train):,} rows ({y_train.sum():,} ST, {y_train.mean():.2%})")
    print(f"  Val:   {len(X_val):,} rows ({y_val.sum():,} ST, {y_val.mean():.2%})")
    print(f"  Test:  {len(X_test):,} rows ({y_test.sum():,} ST, {y_test.mean():.2%})")

    return X_train, X_val, X_test, y_train, y_val, y_test, df, feature_cols


# ============================================================
# 2. Optuna 贝叶斯调参
# ============================================================

def _optuna_objective(trial, X_train, y_train, X_val, y_val, base_scale_weight):
    """Optuna trial objective — maximize val ROC-AUC."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "n_estimators": 500,
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "gamma": trial.suggest_float("gamma", 0.0, 8.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 20, 200),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "early_stopping_rounds": 30,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_val_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_prob)
    return auc


def _train_with_optuna(X_train, y_train, X_val, y_val, n_trials=300):
    """Optuna 贝叶斯搜索最优 XGBoost 参数。"""
    base_scale_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    print(f"\n[..] Optuna 贝叶斯调参 ({n_trials} trials)")
    print(f"     base scale_pos_weight={base_scale_weight:.1f}")

    # 添加剪枝回调
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: _optuna_objective(trial, X_train, y_train, X_val, y_val, base_scale_weight),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\n  Best trial #{study.best_trial.number}")
    print(f"  Best val AUC: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # 用最优参数重新训练 (拿到 best_iteration)
    best_params = study.best_params.copy()
    best_params["n_estimators"] = 500
    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "auc"
    best_params["early_stopping_rounds"] = 30
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["verbosity"] = 0

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    actual_n = model.get_booster().best_iteration + 1
    print(f"  Actual best n_estimators: {actual_n}")

    # 保存 study
    with open(MODEL_DIR / "optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

    return model, study


# ============================================================
# 3. 评估
# ============================================================

def _find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def _topk_recall(y_true, y_prob, k_pcts=[0.01, 0.05, 0.10]):
    results = {}
    n, total_st = len(y_true), y_true.sum()
    order = np.argsort(-y_prob)
    for k in k_pcts:
        top_n = max(1, int(n * k))
        caught = y_true.iloc[order[:top_n]].sum()
        results[f"top{int(k*100)}pct_recall"] = caught / total_st if total_st > 0 else 0
    return results


def _evaluate(model, X_test, y_test) -> dict:
    print(f"\n[..] 评估 ...")
    y_prob = model.predict_proba(X_test)[:, 1]
    best_threshold, best_f1 = _find_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= best_threshold).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    topk = _topk_recall(y_test, y_prob)

    metrics = {
        "roc_auc": round(auc, 4), "pr_auc": round(pr_auc, 4),
        "best_threshold": round(best_threshold, 4),
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "specificity": round(specificity, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        **{k: round(v, 4) for k, v in topk.items()},
    }

    print(f"  ROC-AUC:  {metrics['roc_auc']}")
    print(f"  PR-AUC:   {metrics['pr_auc']}")
    print(f"  Threshold: {metrics['best_threshold']}")
    print(f"  Precision: {metrics['precision']} | Recall: {metrics['recall']} | F1: {metrics['f1']}")
    print(f"  Specificity: {metrics['specificity']}")
    print(f"  Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Top-K Recall: {topk}")

    return metrics, y_prob, y_pred


# ============================================================
# 4. 特征重要性
# ============================================================

def _feature_importance(model, feature_cols) -> pd.DataFrame:
    booster = model.get_booster()
    importance = booster.get_score(importance_type="weight")
    gain_imp = booster.get_score(importance_type="gain")
    cover_imp = booster.get_score(importance_type="cover")

    rows = []
    for f in feature_cols:
        rows.append({
            "feature": f,
            "importance_weight": importance.get(f, 0),
            "importance_gain": gain_imp.get(f, 0),
            "importance_cover": cover_imp.get(f, 0),
        })

    imp = pd.DataFrame(rows).sort_values("importance_gain", ascending=False)
    imp.to_csv(DATA_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    print(f"\n[OK] feature_importance.csv")
    print(f"  Top 10 by gain:")
    for _, row in imp.head(10).iterrows():
        print(f"    {row['feature']:30s}  gain={row['importance_gain']:.4f}")

    return imp


# ============================================================
# 5. 评分卡
# ============================================================

def _scorecard(model, X_test, y_test, y_prob, test_df):
    print(f"\n[..] 生成评分卡 ...")
    PDO, BASE_SCORE, BASE_ODDS = 50, 600, 1.0
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    scores = offset - factor * np.log(np.clip(y_prob / (1 - y_prob), 1e-6, 1e6))

    def _rating(score):
        if score >= 700: return "AAA"
        elif score >= 650: return "AA"
        elif score >= 600: return "A"
        elif score >= 550: return "BBB"
        elif score >= 500: return "BB"
        elif score >= 450: return "B"
        else: return "C"

    ratings = np.array([_rating(s) for s in scores])

    results = pd.DataFrame({
        "code": test_df.loc[X_test.index, "code"].values,
        "quarter": test_df.loc[X_test.index, "quarter"].values,
        "prob": y_prob,
        "score": scores.astype(int),
        "rating": ratings,
        "actual": y_test.values,
    })
    results.to_csv(DATA_DIR / "predictions.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] predictions.csv: {len(results)} rows")

    print(f"\n  评级分布:")
    for r in ["AAA","AA","A","BBB","BB","B","C"]:
        mask = results["rating"] == r
        count = mask.sum()
        actual_st = results.loc[mask, "actual"].sum()
        print(f"    {r:4s}: {count:6,} ({count/len(results):.1%})  actual ST={actual_st}")

    print(f"\n  评分范围: [{scores.min():.0f}, {scores.max():.0f}]")
    print(f"  评分均值: {scores.mean():.0f}")

    return results


# ============================================================
# 6. SHAP 分析
# ============================================================

def _shap_analysis(model, X_test):
    print(f"\n[..] SHAP 分析 (sampled to 5000) ...")
    sample_idx = np.random.choice(len(X_test), min(5000, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx]

    booster = model.get_booster()
    shap_values = booster.predict(xgb.DMatrix(X_sample), pred_contribs=True)
    shap_vals = shap_values[:, :-1]

    fig, ax = plt.subplots(figsize=(12, 10))
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top20_idx = np.argsort(mean_abs)[-20:][::-1]
    top20_features = X_sample.columns[top20_idx]
    top20_importance = mean_abs[top20_idx]

    ax.barh(range(20), top20_importance[::-1], color="steelblue")
    ax.set_yticks(range(20))
    ax.set_yticklabels(top20_features[::-1], fontsize=9)
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("XGBoost Feature Importance (SHAP) — Top 20", fontsize=13)
    fig.tight_layout()
    fig.savefig(DATA_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] data/shap_summary.png")

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    for i, idx in enumerate(top20_idx[:3]):
        feat = X_sample.columns[idx]
        ax = axes2[i]
        ax.scatter(X_sample.iloc[:, idx], shap_vals[:, idx], alpha=0.3, s=2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(feat, fontsize=10)
    fig2.tight_layout()
    fig2.savefig(DATA_DIR / "shap_dependence.png", dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"[OK] data/shap_dependence.png")

    return shap_vals


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--no-optuna", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  TRACE F-06 XGBoost (Optuna 优化版)")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test, df, feature_cols = _load_and_split()

    if args.no_optuna:
        # 回退: 使用预设的最优参数
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        print(f"\n[..] 使用预设参数训练 (scale_pos_weight={scale_pos_weight:.1f})")
        model = xgb.XGBClassifier(
            max_depth=7, learning_rate=0.1, n_estimators=300,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=10,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic", eval_metric="auc",
            early_stopping_rounds=20, random_state=42, n_jobs=-1, verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model, study = _train_with_optuna(X_train, y_train, X_val, y_val, args.trials)

    metrics, y_prob, y_pred = _evaluate(model, X_test, y_test)
    _feature_importance(model, feature_cols)
    _shap_analysis(model, X_test)

    test_df = df.loc[X_test.index]
    _scorecard(model, X_test, y_test, y_prob, test_df)

    with open(MODEL_DIR / "model_xgb.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"\n[OK] model/model_xgb.pkl saved")
    print(f"\n[DONE] F-06 建模完成")


if __name__ == "__main__":
    main()
