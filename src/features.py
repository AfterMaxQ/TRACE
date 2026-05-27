"""
F-02 特征工程模块

计算30+财务比率、构建ST违约标签、生成特征宽表。

输入:
  data/balance_sheet.csv, data/income_statement.csv, data/cash_flow.csv
  data/market_quarterly.csv, data/macro_quarterly.csv
  data/company_info.csv (is_st 字段)

输出:
  data/base_feature.csv  — 特征宽表 (code + quarter + 58特征 + target)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import tushare as ts

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
TS_TOKEN = "4353d440506e8ba010599e3edc686356fa76d24ba66a00f693595de1"

# ============================================================
# 1. 数据加载
# ============================================================


def _load_financials():
    """加载三张财报，按 code + REPORT_DATE 合并"""
    print("[..] 加载三张财报...")

    bs = pd.read_csv(DATA_DIR / "balance_sheet.csv")
    ist = pd.read_csv(DATA_DIR / "income_statement.csv")
    cf = pd.read_csv(DATA_DIR / "cash_flow.csv")

    for df in [bs, ist, cf]:
        df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"])

    merged = bs.merge(ist, on=["code", "REPORT_DATE"], how="inner")
    merged = merged.merge(cf, on=["code", "REPORT_DATE"], how="inner")

    print(f"[OK] 财报合并完成  ({len(merged)} 行, {merged['code'].nunique()} 只)")
    return merged


def _add_quarter(df):
    """从 REPORT_DATE 提取季度标签 2020Q1 格式"""
    df["quarter"] = (
        df["REPORT_DATE"].dt.year.astype(str)
        + "Q"
        + ((df["REPORT_DATE"].dt.month - 1) // 3 + 1).astype(str)
    )
    return df


# ============================================================
# 2. 财务比率计算
# ============================================================


def _calc_financial_ratios(df):
    """计算 30+ 财务比率，全部使用向量化运算"""
    print("[..] 计算财务比率...")

    # -- 流动性 --
    df["current_ratio"] = df["TOTAL_CURRENT_ASSETS"] / df["TOTAL_CURRENT_LIAB"]
    df["quick_ratio"] = (df["TOTAL_CURRENT_ASSETS"] - df["INVENTORY"].fillna(0)) / df["TOTAL_CURRENT_LIAB"]
    df["cash_ratio"] = df["MONETARYFUNDS"] / df["TOTAL_CURRENT_LIAB"]

    # -- 杠杆 --
    df["debt_to_assets"] = df["TOTAL_LIABILITIES"] / df["TOTAL_ASSETS"]
    df["debt_to_equity"] = df["TOTAL_LIABILITIES"] / df["TOTAL_EQUITY"]
    df["equity_multiplier"] = df["TOTAL_ASSETS"] / df["TOTAL_EQUITY"]
    df["long_term_debt_ratio"] = df["TOTAL_NONCURRENT_LIAB"] / df["TOTAL_ASSETS"]
    df["equity_to_assets"] = df["TOTAL_PARENT_EQUITY"] / df["TOTAL_ASSETS"]

    # -- 偿付 --
    df["interest_coverage"] = (df["OPERATE_PROFIT"] + df["FE_INTEREST_EXPENSE"].fillna(0)) / df[
        "FE_INTEREST_EXPENSE"
    ].fillna(0).abs().replace(0, np.nan)

    # -- 盈利 --
    df["roe"] = df["PARENT_NETPROFIT"] / df["TOTAL_PARENT_EQUITY"]
    df["roa"] = df["NETPROFIT"] / df["TOTAL_ASSETS"]
    df["gross_margin"] = (df["OPERATE_INCOME"] - df["OPERATE_COST"]) / df["OPERATE_INCOME"]
    df["operating_margin"] = df["OPERATE_PROFIT"] / df["OPERATE_INCOME"]
    df["net_margin"] = df["NETPROFIT"] / df["OPERATE_INCOME"]
    df["ebitda_margin"] = (
        df["TOTAL_PROFIT"]
        + df["FE_INTEREST_EXPENSE"].fillna(0)
        + df["FA_IR_DEPR"].fillna(0)
        + df["IA_AMORTIZE"].fillna(0)
    ) / df["OPERATE_INCOME"]

    # -- 费用 --
    df["sale_expense_ratio"] = df["SALE_EXPENSE"] / df["OPERATE_INCOME"]
    df["admin_expense_ratio"] = df["MANAGE_EXPENSE"] / df["OPERATE_INCOME"]
    df["research_expense_ratio"] = df["RESEARCH_EXPENSE"] / df["OPERATE_INCOME"]
    df["finance_expense_ratio"] = df["FINANCE_EXPENSE"] / df["OPERATE_INCOME"]

    # -- 现金流 --
    df["cf_to_revenue"] = df["NETCASH_OPERATE"] / df["OPERATE_INCOME"]
    df["cf_to_debt"] = df["NETCASH_OPERATE"] / df["TOTAL_LIABILITIES"]
    df["cf_to_assets"] = df["NETCASH_OPERATE"] / df["TOTAL_ASSETS"]
    df["cf_to_netprofit"] = df["NETCASH_OPERATE"] / df["NETPROFIT"]
    df["free_cf"] = df["NETCASH_OPERATE"] + df["CONSTRUCT_LONG_ASSET"].fillna(0)

    # -- 资产结构 --
    df["fixed_asset_ratio"] = df["FIXED_ASSET"] / df["TOTAL_ASSETS"]
    df["intangible_ratio"] = df["INTANGIBLE_ASSET"] / df["TOTAL_ASSETS"]
    df["goodwill_ratio"] = df["GOODWILL"] / df["TOTAL_ASSETS"]
    df["inventory_ratio"] = df["INVENTORY"] / df["TOTAL_ASSETS"]
    df["receivables_ratio"] = df["ACCOUNTS_RECE"] / df["TOTAL_ASSETS"]

    # -- 营运资金 --
    df["working_capital"] = df["TOTAL_CURRENT_ASSETS"] - df["TOTAL_CURRENT_LIAB"]
    df["working_capital_to_assets"] = df["working_capital"] / df["TOTAL_ASSETS"]

    # -- 每股指标 --
    df["eps"] = df["PARENT_NETPROFIT"] / df["SHARE_CAPITAL"]
    df["bps"] = df["TOTAL_PARENT_EQUITY"] / df["SHARE_CAPITAL"]

    df = df.replace([np.inf, -np.inf], np.nan)

    ratio_count = 33
    print(f"[OK] 财务比率计算完成  ({ratio_count} 个比率)")
    return df


# ============================================================
# 3. 增长率 & 周转率
# ============================================================


def _calc_growth_rates(df):
    """计算同比增长率 (当前季度 vs 去年同期)"""
    print("[..] 计算同比增长率...")

    df = df.sort_values(["code", "REPORT_DATE"])

    growth_map = {
        "OPERATE_INCOME": "revenue_yoy",
        "NETPROFIT": "netprofit_yoy",
        "PARENT_NETPROFIT": "parent_netprofit_yoy",
        "TOTAL_ASSETS": "total_assets_yoy",
        "TOTAL_EQUITY": "total_equity_yoy",
        "OPERATE_PROFIT": "operate_profit_yoy",
        "NETCASH_OPERATE": "operating_cf_yoy",
        "TOTAL_LIABILITIES": "total_liabilities_yoy",
    }

    for col, name in growth_map.items():
        if col in df.columns:
            df[name] = df.groupby("code")[col].transform(lambda x: x / x.shift(4) - 1)

    print(f"[OK] 增长率计算完成  ({len(growth_map)} 个指标)")
    return df


def _calc_efficiency_ratios(df):
    """计算周转率 (使用当期与上期均值作为分母)"""
    print("[..] 计算周转率...")

    df = df.sort_values(["code", "REPORT_DATE"])

    avg_cols = {"TOTAL_ASSETS": "asset_turnover", "INVENTORY": "inventory_turnover",
                "ACCOUNTS_RECE": "receivable_turnover", "TOTAL_EQUITY": "equity_turnover"}

    for base_col, ratio_name in avg_cols.items():
        if base_col in df.columns:
            avg = df.groupby("code")[base_col].transform(lambda x: (x + x.shift(1)) / 2)
            numerator = df["OPERATE_COST"] if ratio_name == "inventory_turnover" else df["OPERATE_INCOME"]
            df[ratio_name] = numerator / avg

    df = df.replace([np.inf, -np.inf], np.nan)
    print(f"[OK] 周转率计算完成  ({len(avg_cols)} 个指标)")
    return df


# ============================================================
# 4. ST 标签构建 (时间序列)
# ============================================================


def _quarter_to_dates(quarter_str):
    """将 '2021Q3' 转为 (start_date, end_date) 的 pd.Timestamp"""
    year = int(quarter_str[:4])
    q = int(quarter_str[-1])
    start_month = (q - 1) * 3 + 1
    end_month = start_month + 2
    start = pd.Timestamp(year=year, month=start_month, day=1)
    end = start + pd.offsets.MonthEnd(1)
    return start, end


def _fetch_st_history():
    """从多数据源获取 ST 历史时间区间

    优先级: Tushare namechange (全市场) → akshare SZSE 简称变更 → None

    Returns:
        pd.DataFrame (ts_code, start_date, end_date, change_reason) 或 None
    """
    cache_path = DATA_DIR / "namechange_history.csv"

    if cache_path.exists():
        print("[..] 从缓存加载 ST 历史记录...")
        df = pd.read_csv(cache_path)
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        if "ann_date" in df.columns:
            df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
        print(f"[OK] 缓存加载成功  ({len(df)} 条变更记录)")
        return df

    # 1. 尝试 Tushare namechange
    print("[..] 尝试 Tushare Pro namechange...")
    try:
        pro = ts.pro_api(TS_TOKEN)
        df = pro.namechange(
            start_date="20000101", end_date="20261231",
            fields="ts_code,name,start_date,end_date,ann_date,change_reason",
        )
        if df is not None and len(df) > 0:
            df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
            df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
            df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
            df.to_csv(cache_path, index=False, encoding="utf-8-sig")
            print(f"[OK] Tushare namechange 获取成功  ({len(df)} 条, 已缓存)")
            return df
    except Exception as e:
        print(f"[!!] Tushare namechange 失败: {e}")

    # 2. 回退: akshare SZSE 简称变更 (仅深市, 免费无限制)
    print("[..] 回退: akshare SZSE 简称变更...")
    try:
        import akshare as ak
        sz = ak.stock_info_sz_change_name(symbol="简称变更")
        if sz is not None and len(sz) > 0:
            sz = sz.rename(columns={
                "变更日期": "start_date", "证券代码": "code_raw",
                "变更后简称": "new_name", "变更前简称": "old_name",
            })
            sz["start_date"] = pd.to_datetime(sz["start_date"], errors="coerce")
            # 补全代码后缀
            sz["ts_code"] = sz["code_raw"].astype(str).str.zfill(6) + ".SZ"
            # 识别 ST 变更: 新名含 ST/*ST → change_reason
            sz["change_reason"] = None
            sz.loc[sz["new_name"].str.contains(r"^\*?ST", na=False, regex=True), "change_reason"] = "*ST"
            # 计算 end_date (下一条变更的 start_date)
            sz = sz.sort_values(["ts_code", "start_date"])
            sz["end_date"] = sz.groupby("ts_code")["start_date"].shift(-1)
            # 只保留 ST 相关记录
            st_records = sz[sz["change_reason"].notna()].copy()
            # 对 ST 记录中的 start_date 实际取值 ST 更名日期
            st_records["change_reason"] = st_records["new_name"].apply(
                lambda n: "*ST" if n.startswith("*ST") else "ST"
            )
            st_records = st_records[["ts_code", "start_date", "end_date", "change_reason"]]
            st_records["ann_date"] = pd.NaT
            st_records.to_csv(cache_path, index=False, encoding="utf-8-sig")
            print(f"[OK] akshare SZSE ST记录获取成功  ({len(st_records)} 条, 已缓存, 仅深市)")
            return st_records
    except Exception as e:
        print(f"[!!] akshare SZSE 简称变更失败: {e}")

    return None


def _build_labels(df):
    """构建违约标签 target: 下一季度是否处于 ST 状态

    数据源优先级:
      1. baostock + akshare 季度 ST 标签 (st_labels.csv, 全市场时间序列)
      2. Tushare namechange / akshare SZSE (namechange_history.csv 缓存)
      3. is_st 快照 (回退)
    """
    print("[..] 构建ST违约标签...")

    # --- 数据源 1: bak_basic 季度末端快照 (全市场, 时间序列) ---
    quarterly_path = DATA_DIR / "st_labels.csv"
    if quarterly_path.exists():
        print("[..] 加载季度 ST 标签 (baostock + akshare)...")
        st_q = pd.read_csv(quarterly_path)
        df = df.merge(st_q[["code", "quarter", "is_st"]], on=["code", "quarter"], how="left")
        df["is_st_current"] = df["is_st"].fillna(0).astype(int)
        df = df.drop(columns=["is_st"])

        df = df.sort_values(["code", "quarter"])
        df["target"] = df.groupby("code")["is_st_current"].shift(-1)
        df["target"] = df["target"].fillna(0).astype(int)
        df = df.drop(columns=["is_st_current"])

        n_pos = df["target"].sum()
        print(f"[OK] ST标签构建完成 (时间序列)  (正样本: {n_pos:,}, 占比: {df['target'].mean():.3%})")
        return df

    # --- 数据源 2 & 3: Tushare namechange / akshare SZSE / is_st 快照 ---
    st_history = _fetch_st_history()

    if st_history is None:
        # 回退: is_st 快照
        info_path = DATA_DIR / "company_info.csv"
        if info_path.exists():
            info = pd.read_csv(info_path)
            st_codes = set(info[info["is_st"] == True]["ts_code"].astype(str))
            df["target"] = df["code"].astype(str).isin(st_codes).astype(int)
            print(f"[OK] 回退: 使用基本信息表 is_st  ({len(st_codes)} 只ST)")
        else:
            df["target"] = 0
            print("[!!] 无 ST 数据源，全部标记为 0")
        n_pos = df["target"].sum()
        print(f"[OK] ST标签构建完成  (正样本: {n_pos:,}, 占比: {df['target'].mean():.3%})")
        return df

    # 过滤 ST/*ST 记录
    st = st_history[st_history["change_reason"].isin(["ST", "*ST"])].copy()
    if len(st) == 0:
        df["target"] = 0
        print("[!!] 无 ST 历史记录，全部标记为 0")
        return df

    st["end_date"] = st["end_date"].fillna(pd.Timestamp.now())

    all_codes = df["code"].unique().tolist()
    all_quarters = sorted(df["quarter"].unique())
    quarter_ranges = {q: _quarter_to_dates(q) for q in all_quarters}

    st_map = {}
    for _, row in st.iterrows():
        code = row["ts_code"]
        st_start = row["start_date"]
        st_end = row["end_date"]
        if pd.isna(st_start):
            continue
        for q, (q_start, q_end) in quarter_ranges.items():
            if st_start <= q_end and st_end >= q_start:
                st_map[(code, q)] = True

    df["is_st_current"] = df.apply(
        lambda r: 1 if (r["code"], r["quarter"]) in st_map else 0, axis=1
    )

    df = df.sort_values(["code", "quarter"])
    df["target"] = df.groupby("code")["is_st_current"].shift(-1)
    df["target"] = df["target"].fillna(0).astype(int)
    df = df.drop(columns=["is_st_current"])

    n_pos = df["target"].sum()
    print(f"[OK] ST标签构建完成  (正样本: {n_pos:,}, 占比: {df['target'].mean():.3%})")
    return df


# ============================================================
# 5. 特征融合 & 缺失值处理
# ============================================================


def _merge_market_macro(df):
    """合并市场季度特征和宏观季度特征"""
    print("[..] 合并市场与宏观特征...")

    market = pd.read_csv(DATA_DIR / "market_quarterly.csv")
    mkt_cols = ["code", "quarter", "log_return", "volatility", "max_drawdown", "beta"]
    if "name" in market.columns and "name" not in df.columns:
        mkt_cols.append("name")
    if "industry" in market.columns and "industry" not in df.columns:
        mkt_cols.append("industry")
    df = df.merge(market[mkt_cols], on=["code", "quarter"], how="left")

    macro = pd.read_csv(DATA_DIR / "macro_quarterly.csv")
    macro = macro.rename(columns={"quarter_label": "quarter"})
    df = df.merge(macro, on="quarter", how="left")

    print(f"[OK] 特征合并完成  ({len(df)} 行, {len(df.columns)} 列)")
    return df


def _handle_missing(df):
    """处理缺失值：行业中位数填充 → 全局中位数 → 0"""
    print("[..] 处理缺失值...")

    ratio_cols = [
        "current_ratio", "quick_ratio", "cash_ratio",
        "debt_to_assets", "debt_to_equity", "equity_multiplier",
        "long_term_debt_ratio", "equity_to_assets", "interest_coverage",
        "roe", "roa", "gross_margin", "operating_margin", "net_margin", "ebitda_margin",
        "sale_expense_ratio", "admin_expense_ratio", "research_expense_ratio", "finance_expense_ratio",
        "cf_to_revenue", "cf_to_debt", "cf_to_assets", "cf_to_netprofit", "free_cf",
        "fixed_asset_ratio", "intangible_ratio", "goodwill_ratio",
        "inventory_ratio", "receivables_ratio", "working_capital_to_assets",
        "eps", "bps",
        "revenue_yoy", "netprofit_yoy", "parent_netprofit_yoy",
        "total_assets_yoy", "total_equity_yoy", "operate_profit_yoy", "operating_cf_yoy",
        "total_liabilities_yoy",
        "inventory_turnover", "receivable_turnover", "asset_turnover", "equity_turnover",
    ]
    available = [c for c in ratio_cols if c in df.columns]

    missing_before = df[available].isnull().sum().sum()

    industry_col = "industry" if "industry" in df.columns else None

    for col in available:
        if industry_col:
            medians = df.groupby(industry_col, dropna=False)[col].transform("median")
            df[col] = df[col].fillna(medians)
        df[col] = df[col].fillna(df[col].median())
        df[col] = df[col].fillna(0)

    macro_cols = [c for c in df.columns if c.startswith(("gdp_", "cpi", "pmi", "m2", "shero", "shibor"))]
    for col in macro_cols:
        df[col] = df[col].fillna(method="ffill")

    # Winsorize 极端值 (1%/99%)
    n_winsorized = 0
    for col in available:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        n_before = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_before > 0:
            df[col] = df[col].clip(lo, hi)
            n_winsorized += 1
    if n_winsorized > 0:
        print(f"[OK] Winsorize 完成  ({n_winsorized} 列, 1%/99% 截尾)")

    missing_after = df[available].isnull().sum().sum()
    print(f"[OK] 缺失值处理完成  (比率列缺失: {missing_before} → {missing_after})")
    return df


# ============================================================
# 6. 输出
# ============================================================

FEATURE_ORDER = [
    "code", "quarter", "name", "industry",
    # 流动性
    "current_ratio", "quick_ratio", "cash_ratio",
    # 杠杆
    "debt_to_assets", "debt_to_equity", "equity_multiplier",
    "long_term_debt_ratio", "equity_to_assets",
    # 偿付
    "interest_coverage",
    # 盈利
    "roe", "roa", "gross_margin", "operating_margin", "net_margin", "ebitda_margin",
    # 费用
    "sale_expense_ratio", "admin_expense_ratio", "research_expense_ratio", "finance_expense_ratio",
    # 现金流
    "cf_to_revenue", "cf_to_debt", "cf_to_assets", "cf_to_netprofit", "free_cf",
    # 资产结构
    "fixed_asset_ratio", "intangible_ratio", "goodwill_ratio",
    "inventory_ratio", "receivables_ratio",
    # 营运资金
    "working_capital_to_assets",
    # 每股
    "eps", "bps",
    # 增长
    "revenue_yoy", "netprofit_yoy", "parent_netprofit_yoy",
    "total_assets_yoy", "total_equity_yoy", "operate_profit_yoy", "operating_cf_yoy",
    "total_liabilities_yoy",
    # 周转
    "inventory_turnover", "receivable_turnover", "asset_turnover", "equity_turnover",
    # 市场
    "log_return", "volatility", "max_drawdown", "beta",
    # 宏观
    "gdp_yoy", "cpi_yoy", "pmi", "m2_yoy", "shero",
    "shibor_on", "shibor_1m", "shibor_1y",
    # 标签
    "target",
]


def main():
    print("=" * 60)
    print("  TRACE F-02 特征工程")
    print("=" * 60)
    t0 = time.time()

    # 1. 加载 & 合并
    df = _load_financials()
    df = _add_quarter(df)

    # 2. 计算财务比率
    df = _calc_financial_ratios(df)

    # 3. 增长率
    df = _calc_growth_rates(df)

    # 4. 周转率
    df = _calc_efficiency_ratios(df)

    # 5. ST 标签
    df = _build_labels(df)

    # 6. 合并市场 + 宏观特征
    df = _merge_market_macro(df)

    # 7. 缺失值处理
    df = _handle_missing(df)

    # 8. 筛选 & 排序输出列
    available_cols = [c for c in FEATURE_ORDER if c in df.columns]
    result = df[available_cols].copy()

    # 浮点精度
    float_cols = result.select_dtypes(include=[np.floating]).columns
    result[float_cols] = result[float_cols].round(6)

    # 按 code + quarter 排序
    result = result.sort_values(["code", "quarter"]).reset_index(drop=True)

    output_path = DATA_DIR / "base_feature.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    elapsed = time.time() - t0

    # 摘要
    print(f"\n[OK] 特征宽表 -> {output_path}")
    print(f"     耗时: {elapsed:.0f}s  |  {len(result):,} 行  |  {result['code'].nunique():,} 只股票")
    print(f"     特征数: {len(available_cols)} (含 target)  |  季度: {result['quarter'].nunique()}")
    if "target" in result.columns:
        print(f"     正样本占比: {result['target'].mean():.3%}")
    print(f"     时间范围: {result['quarter'].min()} – {result['quarter'].max()}")


if __name__ == "__main__":
    main()
