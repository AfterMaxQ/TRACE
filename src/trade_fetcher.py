"""
国际贸易特色数据采集模块
数据范围: 2020-07-01 — 2026-05-26
包含: BDI, 美元兑人民币汇率, 进出口总额, 美国对华关税, 集装箱运价指数
"""

import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

import akshare as ak

DATA_DIR = Path(__file__).parent.parent / "data"
START_DATE = "2020-07-01"
END_DATE = "2026-05-26"


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. BDI 波罗的海干散货指数
# ============================================================
def fetch_bdi() -> pd.DataFrame:
    """从 AKShare 获取 BDI 日线数据，计算季度均值"""
    print("[BDI] 获取波罗的海干散货指数...")
    try:
        df = ak.macro_shipping_bdi()
    except Exception as e:
        print(f"  [!!] AKShare BDI 失败: {e}")
        return pd.DataFrame()

    df.columns = ["date", "bdi", "change_pct", "chg_3m", "chg_6m", "chg_1y", "chg_2y", "chg_3y"]
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "bdi"]].copy()
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    df = df.sort_values("date").reset_index(drop=True)

    # 日线 → 季度均值
    df["quarter"] = df["date"].dt.to_period("Q")
    quarterly = df.groupby("quarter")["bdi"].mean().reset_index()
    quarterly.columns = ["quarter", "bdi_avg"]
    quarterly["quarter"] = quarterly["quarter"].astype(str)

    path = DATA_DIR / "bdi.csv"
    quarterly.to_csv(path, index=False)
    print(f"  [OK] BDI 季度均值已保存: {path} ({len(quarterly)} 行)")
    return quarterly


# ============================================================
# 2. 美元兑人民币汇率
# ============================================================
def fetch_usdcny() -> pd.DataFrame:
    """从中国银行获取美元兑人民币日线汇率"""
    print("[USDCNY] 获取美元兑人民币汇率...")
    try:
        df = ak.currency_boc_sina(
            symbol="美元",
            start_date=START_DATE.replace("-", ""),
            end_date=END_DATE.replace("-", ""),
        )
    except Exception as e:
        print(f"  [!!] AKShare 汇率失败: {e}")
        return pd.DataFrame()

    # 列名: 日期, 中行汇买价, 中行钞买价, 中行汇卖价/钞卖价, 中行折算价, 发布日期
    df.columns = ["date", "buy_rate", "cash_buy", "sell_rate", "mid_rate", "pub_date"]
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    df = df.sort_values("date").reset_index(drop=True)

    # 保留核心字段
    result = df[["date", "mid_rate", "buy_rate", "sell_rate"]].copy()
    result.columns = ["date", "central_parity", "buy_rate", "sell_rate"]

    path = DATA_DIR / "usdcny_daily.csv"
    result.to_csv(path, index=False)
    print(f"  [OK] USDCNY 日线已保存: {path} ({len(result)} 行)")
    return result


# ============================================================
# 3. 进出口总额
# ============================================================
def fetch_trade_balance() -> pd.DataFrame:
    """从海关总署获取月度进出口数据"""
    print("[TRADE] 获取进出口总额...")
    try:
        df = ak.macro_china_hgjck()
    except Exception as e:
        print(f"  [!!] AKShare 进出口失败: {e}")
        return pd.DataFrame()

    # 列名: 月份, 当月出口额-美元, 当月出口额-同比增长, 当月出口额-环比增长,
    #       当月进口额-美元, 当月进口额-同比增长, 当月进口额-环比增长,
    #       累计出口额-美元, 累计出口额-同比增长, 累计进口额-美元, 累计进口额-同比增长
    df.columns = [
        "month", "exports_monthly", "exports_yoy", "exports_mom",
        "imports_monthly", "imports_yoy", "imports_mom",
        "exports_cumulative", "exports_cumulative_yoy",
        "imports_cumulative", "imports_cumulative_yoy",
    ]

    # 解析月份 (格式: "2026年04月份")
    def parse_month(val):
        if isinstance(val, str):
            return val[:4] + "-" + val[5:7]
        return val

    df["month"] = df["month"].apply(parse_month)
    df["month"] = pd.to_datetime(df["month"].str.strip() + "-01", format="%Y-%m-%d")
    df = df[(df["month"] >= "2020-07-01") & (df["month"] <= END_DATE)]
    df = df.sort_values("month").reset_index(drop=True)

    # 计算季度汇总
    df["quarter"] = df["month"].dt.to_period("Q")
    quarterly = df.groupby("quarter").agg(
        exports_quarterly=("exports_monthly", "sum"),
        imports_quarterly=("imports_monthly", "sum"),
        trade_balance=("exports_monthly", lambda x: x.sum() - df.loc[x.index, "imports_monthly"].sum()),
    ).reset_index()
    quarterly["quarter"] = quarterly["quarter"].astype(str)
    # 单位换算: 美元 -> 亿美元
    quarterly["exports_quarterly"] = quarterly["exports_quarterly"] / 1e8
    quarterly["imports_quarterly"] = quarterly["imports_quarterly"] / 1e8
    quarterly["trade_balance"] = quarterly["trade_balance"] / 1e8

    # 同时保存月度明细
    monthly = df[["month", "exports_monthly", "imports_monthly",
                   "exports_yoy", "imports_yoy"]].copy()
    monthly["exports_monthly"] = monthly["exports_monthly"] / 1e8
    monthly["imports_monthly"] = monthly["imports_monthly"] / 1e8

    path_m = DATA_DIR / "trade_monthly.csv"
    monthly.to_csv(path_m, index=False)
    print(f"  [OK] 进出口月度数据已保存: {path_m} ({len(monthly)} 行)")

    path_q = DATA_DIR / "trade_quarterly.csv"
    quarterly.to_csv(path_q, index=False)
    print(f"  [OK] 进出口季度汇总已保存: {path_q} ({len(quarterly)} 行)")
    return quarterly


# ============================================================
# 4. 美国对华关税 (事件驱动)
# ============================================================
def fetch_us_tariffs() -> pd.DataFrame:
    """编译美国对华关税关键事件节点及税率数据"""
    print("[TARIFF] 编译美国对华关税事件数据...")

    records = [
        # 301 条款历史
        {"date": "2018-07-06", "event": "301条款 List 1 生效",
         "tariff_rate": 25.0, "coverage_usd_bn": 34, "status": "active"},
        {"date": "2018-08-23", "event": "301条款 List 2 生效",
         "tariff_rate": 25.0, "coverage_usd_bn": 16, "status": "active"},
        {"date": "2018-09-24", "event": "301条款 List 3 生效 (第一阶段)",
         "tariff_rate": 10.0, "coverage_usd_bn": 200, "status": "escalated"},
        {"date": "2019-05-10", "event": "301条款 List 3 税率上调至 25%",
         "tariff_rate": 25.0, "coverage_usd_bn": 200, "status": "active"},
        {"date": "2019-09-01", "event": "301条款 List 4A 生效",
         "tariff_rate": 15.0, "coverage_usd_bn": 112, "status": "reduced"},
        # 第一阶段贸易协议
        {"date": "2020-01-15", "event": "中美第一阶段贸易协议签署",
         "tariff_rate": None, "coverage_usd_bn": None, "status": "agreement"},
        {"date": "2020-02-14", "event": "List 4A 关税从 15% 下调至 7.5%",
         "tariff_rate": 7.5, "coverage_usd_bn": 112, "status": "reduced"},
        # 拜登政府时期
        {"date": "2022-03-23", "event": "USTR 重启 301 关税排除程序",
         "tariff_rate": None, "coverage_usd_bn": None, "status": "review"},
        {"date": "2022-05-03", "event": "USTR 启动对华301关税四年度审查",
         "tariff_rate": None, "coverage_usd_bn": None, "status": "review"},
        {"date": "2024-05-14", "event": "拜登政府宣布对华新关税 (电动车/电池/半导体/钢铝)",
         "tariff_rate": 100.0, "coverage_usd_bn": 18, "status": "announced"},
        {"date": "2024-09-27", "event": "301关税四年度审查结果: 部分关税上调生效",
         "tariff_rate": 50.0, "coverage_usd_bn": None, "status": "active"},
        # 特朗普第二任期
        {"date": "2025-01-20", "event": "特朗普就职, 威胁对华加征关税",
         "tariff_rate": None, "coverage_usd_bn": None, "status": "threatened"},
        {"date": "2025-02-01", "event": "特朗普签署行政令: 对华加征10%芬太尼关税",
         "tariff_rate": 10.0, "coverage_usd_bn": None, "status": "active"},
        {"date": "2025-02-04", "event": "10% 芬太尼相关关税正式生效",
         "tariff_rate": 10.0, "coverage_usd_bn": None, "status": "active"},
        {"date": "2025-03-04", "event": "对华关税再次加征10% (累计芬太尼关税20%)",
         "tariff_rate": 20.0, "coverage_usd_bn": None, "status": "active"},
        {"date": "2025-04-02", "event": "美国宣布对等关税: 对华34%对等关税",
         "tariff_rate": 34.0, "coverage_usd_bn": None, "status": "active"},
        {"date": "2025-04-09", "event": "对华对等关税上调至84%",
         "tariff_rate": 84.0, "coverage_usd_bn": None, "status": "active"},
        {"date": "2025-04-10", "event": "对华对等关税上调至125%",
         "tariff_rate": 125.0, "coverage_usd_bn": None, "status": "active"},
        {"date": "2025-04-11", "event": "美国宣布对等关税暂缓90天(除中国外); 对华维持125%",
         "tariff_rate": 125.0, "coverage_usd_bn": None, "status": "active"},
        {"date": "2025-05-12", "event": "中美日内瓦首次贸易谈判, 对华对等关税降至10%",
         "tariff_rate": 10.0, "coverage_usd_bn": None, "status": "active"},
    ]

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    df = df.sort_values("date").reset_index(drop=True)

    path = DATA_DIR / "us_tariffs.csv"
    df.to_csv(path, index=False)
    print(f"  [OK] 美国对华关税事件已保存: {path} ({len(df)} 条)")
    return df


# ============================================================
# 5. 集装箱运价指数 (SCFI / CCFI / HRCI)
# ============================================================
def _scrape_sse_scfi() -> pd.DataFrame:
    """
    尝试从上海航运交易所单期查询页面抓取最新 SCFI 综合指数及分航线数据。
    SSE 历史查询需要登录，此方法仅获取当前展示的最新一期数据作为补充。
    """
    print("  [..] 尝试从 SSE 抓取最新 SCFI 数据...")
    url = "https://www.sse.net.cn/index/singleIndex?indexType=scfi"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if not table:
            print("  [!!] 未找到 SCFI 数据表格")
            return pd.DataFrame()

        rows = table.find_all("tr")
        records = []
        for row in rows:
            cells = row.find_all(["th", "td"])
            if not cells:
                continue
            texts = [c.get_text(strip=True) for c in cells]
            records.append(texts)

        if not records:
            return pd.DataFrame()

        # 提取表头和数据
        # 表头: [航线, 单位, 权重, 上期 YYYY-MM-DD, 本期 YYYY-MM-DD, 与上期比涨跌]
        header = records[0] if records else []
        data_rows = records[1:] if len(records) > 1 else []

        # 从表头提取日期 (格式: "上期2026-05-15", "本期2026-05-22")
        import re
        prev_date_str = ""
        curr_date_str = ""
        if len(header) >= 5:
            dates_in_header = []
            for h in header:
                match = re.search(r"(\d{4}-\d{2}-\d{2})", h)
                if match:
                    dates_in_header.append(match.group(1))
            if len(dates_in_header) >= 2:
                prev_date_str = dates_in_header[0]
                curr_date_str = dates_in_header[1]

        result = []
        for r in data_rows:
            if len(r) >= 2 and r[0]:
                row_data = {
                    "route": r[0],
                    "unit": r[1] if len(r) > 1 else "",
                    "weight": r[2] if len(r) > 2 else "",
                    "prev_value": r[3] if len(r) > 3 and r[3] else None,
                    "curr_value": r[4] if len(r) > 4 and r[4] else None,
                    "change": r[5] if len(r) > 5 and r[5] else None,
                    "prev_date": prev_date_str,
                    "curr_date": curr_date_str,
                }
                result.append(row_data)

        df = pd.DataFrame(result)
        if df.empty:
            print("  [!!] 未提取到有效 SCFI 数据行")
            return df

        path = DATA_DIR / "scfi.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  [OK] SCFI 最新一期已抓取 ({curr_date_str}): {path}")
        return df
    except Exception as e:
        print(f"  [!!] SCFI 抓取失败: {e}")
        return pd.DataFrame()


def fetch_freight_indices() -> pd.DataFrame:
    """
    获取航运运价指数:
    - 从 AKShare macro_china_freight_index 获取 BDI/BCI/BSI/HRCI/BCTI/BDTI 周度数据
    - 从 SSE 抓取最新 SCFI 明细 (补充)
    """
    print("[FREIGHT] 获取集装箱/散货运价指数...")
    try:
        df = ak.macro_china_freight_index()
    except Exception as e:
        print(f"  [!!] AKShare 运价指数失败: {e}")
        return pd.DataFrame()

    # 列名: 截止日期, BCI, BHMI, BSI, BDI, HRCI, BCTI, BDTI
    df.columns = ["date", "bci", "bhmi", "bsi", "bdi", "hrci", "bcti", "bdti"]
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    df = df.sort_values("date").reset_index(drop=True)

    # 周度 → 季度均值
    df["quarter"] = df["date"].dt.to_period("Q")
    quarterly = df.groupby("quarter").agg(
        bdi_avg=("bdi", "mean"),
        bci_avg=("bci", "mean"),
        bsi_avg=("bsi", "mean"),
        hrci_avg=("hrci", "mean"),
        bcti_avg=("bcti", "mean"),
        bdti_avg=("bdti", "mean"),
    ).reset_index()
    quarterly["quarter"] = quarterly["quarter"].astype(str)

    # 保留周度明细
    path_w = DATA_DIR / "freight_weekly.csv"
    weekly_cols = ["date", "bdi", "bci", "bsi", "hrci", "bcti", "bdti"]
    df[weekly_cols].to_csv(path_w, index=False)
    print(f"  [OK] 运价指数周度数据已保存: {path_w} ({len(df)} 行)")

    path_q = DATA_DIR / "freight_quarterly.csv"
    quarterly.to_csv(path_q, index=False)
    print(f"  [OK] 运价指数季度均值已保存: {path_q} ({len(quarterly)} 行)")

    # 尝试抓取 SCFI 最新数据
    _scrape_sse_scfi()

    return quarterly


# ============================================================
# 主流程
# ============================================================
def main():
    ensure_data_dir()
    t0 = time.time()

    print("=" * 60)
    print("国际贸易特色数据采集")
    print(f"时间范围: {START_DATE} — {END_DATE}")
    print("=" * 60)

    fetch_bdi()
    print()

    fetch_usdcny()
    print()

    fetch_trade_balance()
    print()

    fetch_us_tariffs()
    print()

    fetch_freight_indices()
    print()

    elapsed = time.time() - t0
    print(f"\n全部完成, 耗时 {elapsed:.1f}s")


if __name__ == "__main__":
    main()
