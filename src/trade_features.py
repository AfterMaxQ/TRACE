"""
F-04 贸易与投入产出特征工程

从 HS2 贸易、IO 表、关税、海外收入四源数据计算 18 个行业-季度级特征，
按 (shenwan_industry, quarter) 输出，供 features.py 合并。

输出: data/trade_features.csv
用法: python src/trade_features.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"

# ============================================================
# 1. 加载映射表 & 参考数据
# ============================================================

def _load_mapping():
    mp = pd.read_csv(DATA_DIR / "industry_mapping.csv")
    # io_sector → 申万行业列表
    io_to_sw = mp.groupby("io_sector")["shenwan_industry"].apply(list).to_dict()
    # 申万行业 → io_sector (取第一个)
    sw_to_io = mp.groupby("shenwan_industry")["io_sector"].first().to_dict()
    # io_sector → hs2 列表
    io_to_hs2 = {}
    for _, row in mp.iterrows():
        hs2_str = row.get("hs2_codes", "")
        if isinstance(hs2_str, str) and hs2_str:
            for h in hs2_str.split("|"):
                io_to_hs2.setdefault(row["io_sector"], set()).add(int(h))
    io_to_hs2 = {k: sorted(v) for k, v in io_to_hs2.items()}
    return io_to_sw, sw_to_io, io_to_hs2


# ============================================================
# 2. CEIC HS2 贸易数据解析
# ============================================================

# HS2 产品描述 → 代码 (针对 贸易代码 列缺失的行)
HS2_DESC_LOOKUP = {
    "鱼、甲壳动物": 3, "活树及其他活植物": 6, "可可及可可制品": 18,
    "谷物、粮食粉、淀粉": 19, "杂项食品": 21, "饮料、酒及醋": 22,
    "矿物燃料、矿物油": 27, "有机化学品": 28, "肥料": 31,
    "精油及香膏": 33, "肥皂、有机表面活性剂": 34,
    "蛋白类物质": 35, "塑料及其制品": 39,
    "皮革制品；鞍具": 42, "木及木制品": 44,
    "稻草、秸秆、针茅": 46, "纸及纸板": 48,
    "絮胎、毡呢及无纺织物": 56, "非针织或非钩编的服装": 62,
    "其他纺织制成品": 63, "鞋靴、护腿": 64, "帽类及其零件": 65,
    "陶瓷产品": 69, "玻璃及其制品": 70, "钢铁": 72,
    "钢铁制品": 73, "铝及其制品": 76,
    "其他贱金属、金属陶瓷": 81, "贱金属工具、器具": 82,
    "贱金属杂项制品": 83, "核反应堆、锅炉、机器": 84,
    "电机、电气设备": 85, "车辆及其零件": 87,
    "光学、照相": 90, "家具；寝具": 94, "玩具、游戏品": 95,
}

def _lookup_hs2(series_name: str) -> int | None:
    """从序列名称中匹配 HS2 代码。"""
    if not isinstance(series_name, str):
        return None
    desc_part = series_name.split(":")[-1] if ":" in series_name else series_name
    for keyword, hs2 in HS2_DESC_LOOKUP.items():
        if keyword in desc_part:
            return hs2
    return None


def _parse_ceic_trade() -> pd.DataFrame:
    """
    解析 ceic_china_hs2_trade_2018_2026.csv (横置 CEIC 格式)
    → 长表: hs2, direction, month, value_usd_mil
    """
    fp = DATA_DIR / "ceic_china_hs2_trade_2018_2026.csv"
    if not fp.exists():
        print("[!!] CEIC HS2 trade 文件不存在")
        return pd.DataFrame()

    raw = pd.read_csv(fp)
    month_cols = [c for c in raw.columns if "/" in str(c)]
    trade_code_col = raw.columns[9]  # 贸易代码

    records = []
    for _, row in raw.iterrows():
        # 从 贸易代码 提取 HS2
        tc = row.get(trade_code_col)
        hs2 = None
        if isinstance(tc, str) and "|" in tc:
            try:
                hs2 = int(tc.split("|")[-1].strip())
            except ValueError:
                pass
        if hs2 is None:
            hs2 = _lookup_hs2(str(row.iloc[0]))
        if hs2 is None:
            continue

        # 方向: 出口 or 进口
        series_name = str(row.iloc[0])
        if series_name.startswith("出口"):
            direction = "export"
        elif series_name.startswith("进口"):
            direction = "import"
        else:
            continue

        # 月度值
        for mc in month_cols:
            val = row.get(mc)
            if pd.isna(val):
                continue
            try:
                v = float(val)
            except (ValueError, TypeError):
                continue
            # 解析月份: "01/2018"
            parts = str(mc).split("/")
            if len(parts) == 2:
                month_str = f"{parts[1]}-{parts[0].zfill(2)}-01"
                records.append({
                    "hs2": hs2, "direction": direction,
                    "month": month_str, "value_usd_mil": v,
                })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["hs2", "direction", "month"]).reset_index(drop=True)
    print(f"  [CEIC trade] {len(df)} 条 ({df['hs2'].nunique()} HS2, "
          f"export={df[df['direction']=='export']['hs2'].nunique()} HS2, "
          f"import={df[df['direction']=='import']['hs2'].nunique()} HS2, "
          f"{df['month'].min().date()}–{df['month'].max().date()})")
    return df


# ============================================================
# 3. CEIC IO 表解析
# ============================================================

def _parse_ceic_io() -> pd.DataFrame:
    """
    解析 ceic_io_noncompetitive.csv → 提取 42 部门 × 时间 的:
      - total_intermediate_input (国产 + 进口)
      - imported_intermediate_input
      - total_output (国产)
      - value_added (推算)
    返回 DataFrame: io_sector, date, imported_input, total_input, total_output
    """
    fp = DATA_DIR / "ceic_io_noncompetitive.csv"
    if not fp.exists():
        print("[!!] CEIC IO 文件不存在")
        return pd.DataFrame()

    raw = pd.read_csv(fp)
    cols = list(raw.columns)

    # 时间行: 第一列是日期标签 (行29起)
    date_col = raw.columns[0]
    raw[date_col] = raw[date_col].astype(str)

    # 国产品中间投入列 (183-224), 进口品中间投入列 (225-266), 总产出国产品 (273-314)
    # 列名格式: "投入产出表:中间使用:中间投入:N 部门名:国产品/进口品"
    dom_input_start = 183
    dom_input_end = 224
    imp_input_start = 225
    imp_input_end = 266
    dom_output_start = 273
    dom_output_end = 314

    def _parse_io_sector(col_name):
        """从列名提取 IO 部门编号"""
        parts = str(col_name).split(":")
        for p in parts:
            p = p.strip()
            if p and p[0].isdigit():
                try:
                    num = int(p.split()[0])
                    return num
                except (ValueError, IndexError):
                    pass
        return None

    records = []
    for row_idx in range(28, len(raw)):
        date_label = str(raw.iloc[row_idx, 0]).strip()
        if not date_label or date_label == "nan":
            continue
        # 解析日期: "12/2015" → 2015-12-31
        parts = date_label.split("/")
        if len(parts) != 2:
            continue
        month, year = int(parts[0]), int(parts[1])
        dt = pd.Timestamp(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(0)

        for sector_id in range(1, 43):
            col_idx_dom = dom_input_start + sector_id - 1
            col_idx_imp = imp_input_start + sector_id - 1
            col_idx_out = dom_output_start + sector_id - 1

            if col_idx_dom >= len(cols) or col_idx_imp >= len(cols) or col_idx_out >= len(cols):
                continue

            try:
                dom_input = float(raw.iloc[row_idx, col_idx_dom])
            except (ValueError, TypeError):
                dom_input = np.nan
            try:
                imp_input = float(raw.iloc[row_idx, col_idx_imp])
            except (ValueError, TypeError):
                imp_input = np.nan
            try:
                total_out = float(raw.iloc[row_idx, col_idx_out])
            except (ValueError, TypeError):
                total_out = np.nan

            total_input = (dom_input if not np.isnan(dom_input) else 0) + \
                          (imp_input if not np.isnan(imp_input) else 0)

            records.append({
                "io_sector": sector_id,
                "date": dt,
                "domestic_input": dom_input,
                "imported_input": imp_input,
                "total_input": total_input,
                "total_output": total_out,
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["io_sector", "date"]).reset_index(drop=True)

    # 过滤 2025 年之后的数据 (IO 表年份频率，覆盖至 2023, 2024 可能是预测值)
    # 保留所有可用年份，会在季度填充时取最近可用

    print(f"  [CEIC IO] {len(df)} 条 ({df['io_sector'].nunique()} 部门, "
          f"{df['date'].min().date()}–{df['date'].max().date()})")
    return df


# ============================================================
# 4. 特征计算
# ============================================================

def _quarterly_from_monthly(trade_df: pd.DataFrame) -> pd.DataFrame:
    """月度 HS2 贸易 → 季度聚合。"""
    trade_df = trade_df.copy()
    trade_df["quarter"] = (
        trade_df["month"].dt.year.astype(str) + "Q" +
        ((trade_df["month"].dt.month - 1) // 3 + 1).astype(str)
    )
    quarterly = trade_df.groupby(["hs2", "direction", "quarter"])["value_usd_mil"].sum().reset_index()
    return quarterly


def _hs2_to_shenwan_trade(quarterly: pd.DataFrame, io_to_hs2: dict,
                          io_to_sw: dict) -> pd.DataFrame:
    """
    HS2 季度贸易 → 申万行业季度贸易。
    路径: HS2 → IO → 申万
    """
    # 构建 HS2 → IO 反向映射
    hs2_to_io = {}
    for io_id, hs2_list in io_to_hs2.items():
        for h in hs2_list:
            hs2_to_io.setdefault(h, []).append(io_id)

    # 展开: 每个 HS2 → 一个主 IO 部门 (取第一个)
    hs2_io_map = {h: io_list[0] for h, io_list in hs2_to_io.items()}
    quarterly["io_sector"] = quarterly["hs2"].map(hs2_io_map)
    quarterly = quarterly.dropna(subset=["io_sector"])
    quarterly["io_sector"] = quarterly["io_sector"].astype(int)

    # 聚到 IO 部门
    io_trade = quarterly.groupby(["io_sector", "direction", "quarter"])["value_usd_mil"].sum().reset_index()

    # 展开到申万行业 (一个 IO 部门 → 多个申万行业)
    rows = []
    for _, r in io_trade.iterrows():
        sw_list = io_to_sw.get(r["io_sector"], [])
        for sw in sw_list:
            rows.append({
                "shenwan_industry": sw,
                "quarter": r["quarter"],
                "direction": r["direction"],
                "value_usd_mil": r["value_usd_mil"],
            })
    df = pd.DataFrame(rows)
    # 去重: 同一(申万,季度,方向)从多个IO部门聚合时取和
    df = df.groupby(["shenwan_industry","quarter","direction"])["value_usd_mil"].sum().reset_index()
    return df


def _build_features(sw_trade: pd.DataFrame, io_df: pd.DataFrame,
                    tariff_df: pd.DataFrame, overseas_df: pd.DataFrame,
                    sw_to_io: dict, io_to_sw: dict, io_to_hs2: dict) -> pd.DataFrame:
    """组装最终的 trade_features.csv。"""

    # --- 贸易流特征 (export/import by industry-quarter) ---
    # pivot direction
    trade_pivot = sw_trade.pivot_table(
        index=["shenwan_industry", "quarter"],
        columns="direction", values="value_usd_mil", aggfunc="sum"
    ).fillna(0).reset_index()
    trade_pivot.columns.name = None

    for col in ["export", "import"]:
        if col not in trade_pivot.columns:
            trade_pivot[col] = 0.0
    trade_pivot["trade_balance"] = trade_pivot["export"] - trade_pivot["import"]

    # YoY growth
    trade_pivot = trade_pivot.sort_values(["shenwan_industry", "quarter"])
    for col in ["export", "import"]:
        trade_pivot[f"{col}_yoy"] = trade_pivot.groupby("shenwan_industry")[col].pct_change(4)

    # 行业份额
    trade_pivot["export_share"] = trade_pivot.groupby("quarter")["export"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0)
    trade_pivot["import_share"] = trade_pivot.groupby("quarter")["import"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0)

    trade_cols = ["export", "import", "trade_balance",
                  "export_yoy", "import_yoy", "export_share", "import_share"]

    # --- IO 特征 (年度 → 季度填充) ---
    # 每 IO 部门取最近可用年份的值
    io_df = io_df.copy()
    io_df["year"] = io_df["date"].dt.year

    # 计算比率
    io_df["import_dependency"] = np.where(
        io_df["total_input"] > 0,
        io_df["imported_input"] / io_df["total_input"], np.nan)
    io_df["backward_linkage"] = np.where(
        io_df["total_output"] > 0,
        io_df["total_input"] / io_df["total_output"], np.nan)
    io_df["domestic_value_added_ratio"] = np.where(
        io_df["total_output"] > 0,
        (1.0 - io_df["total_input"] / io_df["total_output"]).clip(lower=0), np.nan)

    # 按 IO 部门取每年最新值
    io_latest = io_df.sort_values("date").groupby(["io_sector", "year"]).last().reset_index()

    # 生成所有 (IO sector × quarter) 组合并前向填充
    quarters = sorted(trade_pivot["quarter"].unique())
    io_quarterly_rows = []
    for io_id in sorted(io_latest["io_sector"].unique()):
        io_data = io_latest[io_latest["io_sector"] == io_id].sort_values("year")
        for q in quarters:
            q_year = int(q[:4])
            # 找 <= q_year 的最近 IO 数据
            past = io_data[io_data["year"] <= q_year]
            if past.empty:
                past = io_data.head(1)  # 最早可用
            latest = past.iloc[-1]
            io_quarterly_rows.append({
                "io_sector": io_id,
                "quarter": q,
                "import_dependency": latest["import_dependency"],
                "backward_linkage": latest["backward_linkage"],
                "domestic_value_added_ratio": latest["domestic_value_added_ratio"],
            })
    io_quarterly = pd.DataFrame(io_quarterly_rows)

    # 展开 IO → 申万
    io_sw_rows = []
    for _, r in io_quarterly.iterrows():
        sw_list = io_to_sw.get(r["io_sector"], [])
        for sw in sw_list:
            io_sw_rows.append({
                "shenwan_industry": sw, "quarter": r["quarter"],
                "import_dependency": r["import_dependency"],
                "backward_linkage": r["backward_linkage"],
                "domestic_value_added_ratio": r["domestic_value_added_ratio"],
            })
    io_sw = pd.DataFrame(io_sw_rows)
    # 去重: 取均值
    io_sw = io_sw.groupby(["shenwan_industry","quarter"]).mean().reset_index()

    # --- 关税特征 (年度 → 季度填充, HS2 → IO → 申万) ---
    # 按 year → HS2 映射到 IO → 取平均
    tariff_df = tariff_df.copy()
    hs2_io_map = {}
    for io_id, hs2_list in io_to_hs2.items():
        for h in hs2_list:
            hs2_io_map[h] = io_id

    tariff_df["io_sector"] = tariff_df["hs2"].map(hs2_io_map)
    tariff_io = tariff_df.dropna(subset=["io_sector"])
    tariff_io["io_sector"] = tariff_io["io_sector"].astype(int)

    # IO × year 加权平均 (用 n_usable 做权重)
    tariff_io_agg = tariff_io.groupby(["io_sector", "year"]).apply(
        lambda g: np.average(g["avg_tariff"], weights=g["n_usable"])
        if g["n_usable"].sum() > 0 else g["avg_tariff"].mean()
    ).reset_index(name="us_tariff_rate")

    # 关税基线 (2017)
    bl = tariff_io_agg[tariff_io_agg["year"] == 2017][["io_sector", "us_tariff_rate"]].copy()
    bl.columns = ["io_sector", "us_tariff_baseline_2017"]

    # 季度填充
    tariff_q_rows = []
    for io_id in sorted(tariff_io_agg["io_sector"].unique()):
        tdata = tariff_io_agg[tariff_io_agg["io_sector"] == io_id].sort_values("year")
        bl_val = bl[bl["io_sector"] == io_id]["us_tariff_baseline_2017"].values
        baseline = bl_val[0] if len(bl_val) > 0 else 0.0
        for q in quarters:
            q_year = int(q[:4])
            past = tdata[tdata["year"] <= q_year]
            if past.empty:
                past = tdata.head(1)
            latest = past.iloc[-1]
            prev_year = tdata[tdata["year"] == q_year - 1]
            tariff_yoy = (latest["us_tariff_rate"] - prev_year["us_tariff_rate"].values[0]) \
                if len(prev_year) > 0 else 0.0
            tariff_q_rows.append({
                "io_sector": io_id, "quarter": q,
                "us_tariff_rate": latest["us_tariff_rate"],
                "us_tariff_yoy": tariff_yoy,
                "us_tariff_baseline_2017": baseline,
            })
    tariff_quarterly = pd.DataFrame(tariff_q_rows)

    # 展开到申万
    tariff_sw_rows = []
    for _, r in tariff_quarterly.iterrows():
        sw_list = io_to_sw.get(r["io_sector"], [])
        for sw in sw_list:
            tariff_sw_rows.append({
                "shenwan_industry": sw, "quarter": r["quarter"],
                "us_tariff_rate": r["us_tariff_rate"],
                "us_tariff_yoy": r["us_tariff_yoy"],
                "us_tariff_baseline_2017": r["us_tariff_baseline_2017"],
            })
    tariff_sw = pd.DataFrame(tariff_sw_rows)
    # 去重: 取均值
    tariff_sw = tariff_sw.groupby(["shenwan_industry","quarter"]).mean().reset_index()

    # --- 海外收入特征 ---
    overseas_df = overseas_df.copy()
    overseas_df["report_date"] = pd.to_datetime(overseas_df["report_date"])
    overseas_df["quarter"] = (
        overseas_df["report_date"].dt.year.astype(str) + "Q" +
        ((overseas_df["report_date"].dt.month - 1) // 3 + 1).astype(str)
    )
    # 合并公司信息获取行业
    info = pd.read_csv(DATA_DIR / "company_info.csv", dtype={"ts_code": str})
    overseas_df = overseas_df.merge(info[["ts_code", "industry"]], on="ts_code", how="left")
    overseas_df = overseas_df.dropna(subset=["industry"])

    # 前向填充 (每个公司只在报告期有数据)
    overseas_df = overseas_df.sort_values(["ts_code", "quarter"])
    overseas_df["overseas_ratio"] = overseas_df.groupby("ts_code")["overseas_ratio"].ffill()

    # 行业季度聚合
    ov_agg = overseas_df.groupby(["industry", "quarter"])["overseas_ratio"].agg([
        ("overseas_rev_median", "median"),
        ("overseas_rev_mean", "mean"),
    ]).reset_index()
    ov_agg.columns = ["shenwan_industry", "quarter", "overseas_rev_median", "overseas_rev_mean"]
    # p75
    p75 = overseas_df.groupby(["industry", "quarter"])["overseas_ratio"].quantile(0.75).reset_index()
    p75.columns = ["shenwan_industry", "quarter", "overseas_rev_p75"]
    ov_agg = ov_agg.merge(p75, on=["shenwan_industry", "quarter"], how="left")

    # --- 合并所有特征 ---
    trade_feat = trade_pivot[["shenwan_industry", "quarter"] + trade_cols].copy()
    trade_feat = trade_feat.merge(io_sw, on=["shenwan_industry", "quarter"], how="outer")
    trade_feat = trade_feat.merge(tariff_sw, on=["shenwan_industry", "quarter"], how="outer")
    trade_feat = trade_feat.merge(ov_agg, on=["shenwan_industry", "quarter"], how="outer")

    return trade_feat


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  F-04 贸易与投入产出特征计算")
    print("=" * 60)

    # 加载映射
    io_to_sw, sw_to_io, io_to_hs2 = _load_mapping()
    print(f"[..] 映射表: {len(io_to_sw)} IO部门, {len(sw_to_io)} 申万行业, "
          f"{len(io_to_hs2)} IO有HS2映射")

    # CEIC HS2 贸易
    trade_raw = _parse_ceic_trade()
    if trade_raw.empty:
        print("[!!] 无贸易数据，中止")
        return

    # CEIC IO 表
    io_raw = _parse_ceic_io()

    # 关税
    tariff_path = DATA_DIR / "hs2_tariff_annual.csv"
    if not tariff_path.exists():
        print("[!!] hs2_tariff_annual.csv 不存在, 跳过关税特征")
        tariff_df = pd.DataFrame(columns=["hs2","year","avg_tariff","n_usable"])
    else:
        tariff_df = pd.read_csv(tariff_path)

    # 海外收入
    ov_path = DATA_DIR / "overseas_revenue.csv"
    if not ov_path.exists():
        print("[!!] overseas_revenue.csv 不存在, 跳过海外收入特征")
        overseas_df = pd.DataFrame(columns=["ts_code","report_date","overseas_ratio"])
    else:
        overseas_df = pd.read_csv(ov_path)

    # 季度聚合 & HS2 → 申万
    quarterly = _quarterly_from_monthly(trade_raw)
    sw_trade = _hs2_to_shenwan_trade(quarterly, io_to_hs2, io_to_sw)
    print(f"  [trade→申万] {sw_trade['shenwan_industry'].nunique()} 个申万行业, "
          f"{sw_trade['quarter'].nunique()} 个季度")

    # 组装特征
    features = _build_features(sw_trade, io_raw, tariff_df, overseas_df,
                               sw_to_io, io_to_sw, io_to_hs2)

    # 清理: 替换 inf → NaN
    features = features.replace([np.inf, -np.inf], np.nan)

    # 圆整
    num_cols = features.select_dtypes(include=[np.number]).columns
    features[num_cols] = features[num_cols].round(6)

    # 排序 & 保存
    features = features.sort_values(["shenwan_industry", "quarter"]).reset_index(drop=True)
    out_path = DATA_DIR / "trade_features.csv"
    features.to_csv(out_path, index=False, encoding="utf-8-sig")

    n_cols = len(features.columns)
    n_rows = len(features)
    n_sw = features["shenwan_industry"].nunique()
    n_q = features["quarter"].nunique()
    print(f"\n[OK] trade_features.csv: {n_rows} 行 × {n_cols} 列")
    print(f"     申万行业: {n_sw}, 季度: {n_q}")
    feat_cols = [c for c in features.columns if c not in ["shenwan_industry", "quarter"]]
    print(f"     特征列 ({len(feat_cols)}): {feat_cols}")


if __name__ == "__main__":
    main()
