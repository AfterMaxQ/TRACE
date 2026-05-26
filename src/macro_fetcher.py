import warnings
from datetime import datetime

import akshare as ak
import pandas as pd
import requests

warnings.filterwarnings("ignore")

START_DATE = "2020-01-01"
END_DATE = "2026-05-25"
DATA_DIR = "data"


def parse_china_month(m_str: str) -> str:
    """将 '2026年04月份' 转为 '2026-04'"""
    s = m_str.replace("年", "-").replace("月份", "")
    return s


def parse_yyyymm(m_str: str) -> str:
    """将 '201501' 转为 '2015-01'"""
    return f"{m_str[:4]}-{m_str[4:6]}"


def month_to_quarter(month_str: str) -> str:
    """将 '2026-04' 转为 '2026Q2'"""
    dt = pd.to_datetime(month_str)
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}Q{q}"


def _quarterly_agg(df: pd.DataFrame, col: str, agg: str, min_months: int = 2) -> pd.DataFrame:
    """月度数据转季度，过滤残缺季度（不足 min_months 个月）。

    df  需含 'quarter_label' 和数值列 col
    agg  'mean'（CPI/PMI/M2）或 'sum'（社融）
    """
    counts = df.groupby("quarter_label")[col].count()
    if agg == "sum":
        qdf = df.groupby("quarter_label", as_index=False)[col].sum()
    else:
        qdf = df.groupby("quarter_label", as_index=False)[col].mean()
    valid = counts[counts >= min_months].index
    qdf = qdf[qdf["quarter_label"].isin(valid)]
    qdf[col] = qdf[col].round(2)
    return qdf


def quarter_label(period: str) -> str:
    """统一季度标签"""
    return period


# ---------------------------------------------------------------------------
# 1. GDP
# ---------------------------------------------------------------------------
def parse_quarter_raw(q_str: str):
    """解析原始季度字符串，返回 (year, quarter_or_end, is_single)"""
    import re
    m = re.match(r"(\d{4})年第(\d+)-(\d+)季度", q_str)
    if m:
        return int(m.group(1)), int(m.group(3)), False
    m = re.match(r"(\d{4})年第(\d+)季度$", q_str)
    if m:
        return int(m.group(1)), int(m.group(2)), True
    return None, None, None


def fetch_gdp() -> pd.DataFrame:
    print(">>> 采集 GDP 数据 ...")
    raw = ak.macro_china_gdp()
    raw.columns = [str(c).strip() for c in raw.columns]
    df = raw.rename(columns={
        raw.columns[0]: "quarter_raw",
        raw.columns[1]: "gdp_abs",
        raw.columns[2]: "gdp_yoy",
    })
    df["gdp_yoy"] = pd.to_numeric(df["gdp_yoy"], errors="coerce")

    parsed = df["quarter_raw"].apply(parse_quarter_raw)
    df["year"] = parsed.apply(lambda x: x[0])
    df["q_end"] = parsed.apply(lambda x: x[1])
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["q_end"] = df["q_end"].astype(int)

    # 使用原始累计同比（实际增速），Q1=单季同比，Q2-Q4=累计同比
    df["quarter_label"] = df.apply(lambda r: f"{r['year']}Q{r['q_end']}", axis=1)
    df = df[df["year"] >= 2020]
    df = df.sort_values("quarter_label").reset_index(drop=True)
    df["gdp_yoy"] = df["gdp_yoy"].round(2)
    print(f"    GDP 数据: {df.shape[0]} 条（Q1为单季同比，Q2-Q4为累计同比）")
    return df[["quarter_label", "gdp_yoy"]]


# ---------------------------------------------------------------------------
# 2. CPI  — 东方财富源
# ---------------------------------------------------------------------------
def fetch_cpi() -> pd.DataFrame:
    print(">>> 采集 CPI 数据 ...")
    raw = ak.macro_china_cpi()
    raw.columns = [str(c).strip() for c in raw.columns]
    df = raw.rename(columns={raw.columns[0]: "month", "全国-同比增长": "cpi_yoy"})
    df["month"] = df["month"].apply(parse_china_month)
    df["month_dt"] = pd.to_datetime(df["month"])
    df = df[(df["month_dt"] >= START_DATE) & (df["month_dt"] <= END_DATE)]
    df["quarter_label"] = df["month"].apply(month_to_quarter)
    qdf = _quarterly_agg(df, "cpi_yoy", "mean")
    print(f"    CPI 数据: {qdf.shape[0]} 条（季度均值，≥2月/季）")
    return qdf


# ---------------------------------------------------------------------------
# 3. PMI
# ---------------------------------------------------------------------------
def fetch_pmi() -> pd.DataFrame:
    print(">>> 采集 PMI 数据 ...")
    raw = ak.macro_china_pmi()
    raw.columns = [str(c).strip() for c in raw.columns]
    df = raw.rename(columns={raw.columns[0]: "month", "制造业-指数": "pmi"})
    df["month"] = df["month"].apply(parse_china_month)
    df["month_dt"] = pd.to_datetime(df["month"])
    df = df[(df["month_dt"] >= START_DATE) & (df["month_dt"] <= END_DATE)]
    df["quarter_label"] = df["month"].apply(month_to_quarter)
    qdf = _quarterly_agg(df, "pmi", "mean")
    print(f"    PMI 数据: {qdf.shape[0]} 条（季度均值，≥2月/季）")
    return qdf


# ---------------------------------------------------------------------------
# 4. M2 增速
# ---------------------------------------------------------------------------
def fetch_m2() -> pd.DataFrame:
    print(">>> 采集 M2 数据 ...")
    raw = ak.macro_china_money_supply()
    raw.columns = [str(c).strip() for c in raw.columns]
    col_m2 = [c for c in raw.columns if "M2" in c and "同比增长" in c][0]
    df = raw.rename(columns={raw.columns[0]: "month", col_m2: "m2_yoy"})
    df["month"] = df["month"].apply(parse_china_month)
    df["month_dt"] = pd.to_datetime(df["month"])
    df = df[(df["month_dt"] >= START_DATE) & (df["month_dt"] <= END_DATE)]
    df["quarter_label"] = df["month"].apply(month_to_quarter)
    qdf = _quarterly_agg(df, "m2_yoy", "mean")
    print(f"    M2 数据: {qdf.shape[0]} 条（季度均值，≥2月/季）")
    return qdf


# ---------------------------------------------------------------------------
# 5. 社融规模
# ---------------------------------------------------------------------------
def fetch_sheRong() -> pd.DataFrame:
    print(">>> 采集 社融规模 数据 ...")
    raw = ak.macro_china_shrzgm()
    raw.columns = [str(c).strip() for c in raw.columns]
    df = raw.rename(columns={raw.columns[0]: "month", "社会融资规模增量": "shero"})
    df["month_str"] = df["month"].astype(str).str.strip()
    df["month"] = df["month_str"].apply(parse_yyyymm)
    df["month_dt"] = pd.to_datetime(df["month"])
    df = df[(df["month_dt"] >= START_DATE) & (df["month_dt"] <= END_DATE)]
    df["quarter_label"] = df["month"].apply(month_to_quarter)
    qdf = _quarterly_agg(df, "shero", "sum")
    print(f"    社融 数据: {qdf.shape[0]} 条（季度汇总，≥2月/季）")
    return qdf


# ---------------------------------------------------------------------------
# 6. Shibor 利率 — 直接调用东方财富 API
# ---------------------------------------------------------------------------
def _fetch_shibor_indicator(indicator: str, indicator_id: str) -> pd.DataFrame:
    """从东方财富 API 获取单个 Shibor 指标"""
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName": "RPT_IMP_INTRESTRATEN",
        "columns": "REPORT_DATE,IR_RATE",
        "filter": f'(MARKET_CODE="001")(CURRENCY_CODE="CNY")(INDICATOR_ID="{indicator_id}")',
        "pageNumber": 1,
        "pageSize": 2000,
        "sortTypes": -1,
        "sortColumns": "REPORT_DATE",
        "source": "WEB",
        "client": "WEB",
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    total_pages = data["result"]["pages"]
    rows = data["result"]["data"]
    for pg in range(2, total_pages + 1):
        params["pageNumber"] = pg
        try:
            r2 = requests.get(url, params=params, timeout=30)
            rows.extend(r2.json()["result"]["data"])
        except Exception:
            pass
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["REPORT_DATE"]).dt.date
    df = df[(df["date"] >= pd.to_datetime(START_DATE).date()) &
            (df["date"] <= pd.to_datetime(END_DATE).date())]
    df["rate"] = pd.to_numeric(df["IR_RATE"], errors="coerce")
    df = df.rename(columns={"date": "报告日", "rate": indicator})
    return df[["报告日", indicator]]


def fetch_shibor() -> pd.DataFrame:
    print(">>> 采集 Shibor 利率 ...")
    on = _fetch_shibor_indicator("shibor_on", "001")
    m1 = _fetch_shibor_indicator("shibor_1m", "201")
    y1 = _fetch_shibor_indicator("shibor_1y", "301")
    merged = on.merge(m1, on="报告日", how="outer").merge(y1, on="报告日", how="outer")
    merged["报告日"] = pd.to_datetime(merged["报告日"])
    merged["quarter_label"] = merged["报告日"].apply(
        lambda d: f"{d.year}Q{(d.month-1)//3+1}"
    )
    qdf = merged.groupby("quarter_label", as_index=False)[
        ["shibor_on", "shibor_1m", "shibor_1y"]
    ].mean()
    for col in ["shibor_on", "shibor_1m", "shibor_1y"]:
        qdf[col] = qdf[col].round(2)
    print(f"    Shibor 数据: {qdf.shape[0]} 条（季度均值）")
    return qdf


# ---------------------------------------------------------------------------
# 合并输出
# ---------------------------------------------------------------------------
def merge_quarterly(gdp, cpi, pmi, m2, shero, shibor) -> pd.DataFrame:
    print(">>> 合并季度数据 ...")
    from functools import reduce
    frames = [gdp, cpi, pmi, m2, shero, shibor]
    merged = reduce(
        lambda left, right: pd.merge(left, right, on="quarter_label", how="outer"),
        frames,
    )
    merged = merged.sort_values("quarter_label").reset_index(drop=True)
    merged = merged.dropna(subset=["gdp_yoy"])
    return merged


def main():
    print("=" * 60)
    print(f"宏观经济数据采集: {START_DATE} ~ {END_DATE}")
    print("=" * 60)

    gdp = fetch_gdp()
    cpi = fetch_cpi()
    pmi = fetch_pmi()
    m2 = fetch_m2()
    shero = fetch_sheRong()
    shibor = fetch_shibor()

    merged = merge_quarterly(gdp, cpi, pmi, m2, shero, shibor)

    import os
    os.makedirs(DATA_DIR, exist_ok=True)

    # 输出各指标单独 CSV
    gdp.to_csv(f"{DATA_DIR}/gdp.csv", index=False, encoding="utf-8-sig")
    cpi.to_csv(f"{DATA_DIR}/cpi.csv", index=False, encoding="utf-8-sig")
    pmi.to_csv(f"{DATA_DIR}/pmi.csv", index=False, encoding="utf-8-sig")
    m2.to_csv(f"{DATA_DIR}/m2.csv", index=False, encoding="utf-8-sig")
    shero.to_csv(f"{DATA_DIR}/shero.csv", index=False, encoding="utf-8-sig")
    shibor.to_csv(f"{DATA_DIR}/shibor.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(f"{DATA_DIR}/macro_quarterly.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("输出文件:")
    for f in ["gdp.csv", "cpi.csv", "pmi.csv", "m2.csv", "shero.csv", "shibor.csv", "macro_quarterly.csv"]:
        fp = f"{DATA_DIR}/{f}"
        print(f"  {fp}  ({os.path.getsize(fp)} bytes)")
    print("=" * 60)

    print("\n合并数据预览:")
    print(merged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()