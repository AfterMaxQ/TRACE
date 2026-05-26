"""
新闻数据采集模块

数据源：
  - 财联社电报 (CLS):          ak.stock_info_global_cls（市场级快讯）
  - 东方财富个股新闻:            ak.stock_news_em（按股票代码，最近100条）
  - 巨潮资讯公司公告:            ak.stock_notice_report（按日期回溯，2020年起月度采样）
  - 新浪财经7x24全球资讯:        ak.stock_info_global_sina（市场级）

处理流程：抓取 → HTML清洗 → 去重 → 存入 data/news_raw.csv

输出字段：date, code, source, title
"""
import re
import time
import random
import pandas as pd
from pathlib import Path

import requests
import akshare as ak

DATA_DIR = Path(__file__).parent.parent / "data"
STOCK_INFO_CSV = DATA_DIR / "TRACE_上市公司基本信息.csv"


# ============================================================
# 工具函数
# ============================================================

def _clean_html(text: str) -> str:
    """去除 HTML 标签，合并空白。"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_code(code: str) -> str:
    """统一股票代码为 000001.SZ 格式。"""
    code = str(code).strip().upper().replace(".SS", ".SH")
    if len(code) == 6:
        if code.startswith(("6", "9")):
            return f"{code}.SH"
        elif code.startswith(("0", "3")):
            return f"{code}.SZ"
        elif code.startswith(("8", "4")):
            return f"{code}.BJ"
    return code


def _short_code(code: str) -> str:
    """返回纯6位数字代码。"""
    return code.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")


def _detect_codes_from_title(title: str) -> list:
    """从新闻标题中检测提及的6位股票代码。"""
    found = set()
    for m in re.finditer(r"\b(\d{6})\b", str(title)):
        c = _normalize_code(m.group(1))
        if c.endswith((".SH", ".SZ", ".BJ")):
            found.add(c)
    return list(found)


def _load_codes() -> list:
    """加载全A股代码列表。"""
    if STOCK_INFO_CSV.exists():
        info = pd.read_csv(STOCK_INFO_CSV, usecols=["ts_code"])
        codes = sorted(set(
            _normalize_code(c) for c in info["ts_code"].dropna().unique()
        ))
    else:
        stock_df = pd.read_csv(DATA_DIR / "stock_daily.csv", usecols=["code"])
        codes = sorted(stock_df["code"].dropna().unique())
    return codes


# ============================================================
# 1. 财联社电报
# ============================================================

def fetch_cls_telegraph() -> pd.DataFrame:
    """财联社电报 — 市场级快讯，最近20条/分类。"""
    rows = []
    for cat in ("全球", "重点"):
        try:
            df = ak.stock_info_global_cls(symbol=cat)
            # 实际列名: 标题, 内容, 发布日期, 发布时间
            for _, row in df.iterrows():
                title = _clean_html(row["标题"])
                if not title:
                    continue
                date_obj = row["发布日期"]
                date_str = str(date_obj) if date_obj else ""
                if not date_str:
                    continue
                detected = _detect_codes_from_title(title) or [""]
                for c in detected:
                    rows.append({
                        "date": date_str,
                        "code": c,
                        "source": "cls",
                        "title": title[:500],
                    })
        except Exception as e:
            print(f"  cls({cat}): {e}")
    print(f"  CLS: {len(rows)} rows")
    return pd.DataFrame(rows)


# ============================================================
# 2. 东方财富个股新闻
# ============================================================

def fetch_em_stock_news(codes: list[str]) -> pd.DataFrame:
    """东方财富个股新闻 — 每只股票最近10条，遍历全A股。"""
    rows = []
    total = len(codes)
    for i, code in enumerate(codes):
        short = _short_code(code)
        try:
            df = ak.stock_news_em(symbol=short)
            if df.empty:
                continue
            for _, row in df.iterrows():
                title = _clean_html(row.iloc[1])
                if not title:
                    continue
                try:
                    ts = pd.Timestamp(row.iloc[3])
                    date_str = ts.strftime("%Y-%m-%d")
                except Exception:
                    continue
                rows.append({
                    "date": date_str,
                    "code": code,
                    "source": "eastmoney",
                    "title": title[:500],
                })
        except Exception:
            continue
        time.sleep(random.uniform(0.15, 0.35))
        if (i + 1) % 500 == 0:
            print(f"  eastmoney: {i + 1}/{total} stocks, {len(rows)} rows")
    print(f"  EastMoney: {len(rows)} rows")
    return pd.DataFrame(rows)


# ============================================================
# 3. 新浪财经7x24全球资讯
# ============================================================

def fetch_sina_global_news(max_pages: int = 10) -> pd.DataFrame:
    """新浪财经7x24全球资讯 — 市场级快讯，支持翻页回溯。"""
    rows = []
    url = "https://zhibo.sina.com.cn/api/zhibo/feed"
    for page in range(1, max_pages + 1):
        params = {
            "page": str(page),
            "page_size": "50",
            "zhibo_id": "152",
            "tag_id": "0",
            "dire": "f",
            "dpc": "1",
            "pagesize": "50",
            "type": "1",
        }
        headers = {
            "Referer": "https://finance.sina.com.cn/7x24",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            data_json = r.json()
            items = data_json["result"]["data"]["feed"]["list"]
            if not items:
                break
            for item in items:
                title = _clean_html(item.get("rich_text", ""))
                ctime = item.get("create_time", "")
                if not title or not ctime:
                    continue
                try:
                    ts = pd.Timestamp(ctime)
                    date_str = ts.strftime("%Y-%m-%d")
                except Exception:
                    continue
                detected = _detect_codes_from_title(title) or [""]
                for c in detected:
                    rows.append({
                        "date": date_str,
                        "code": c,
                        "source": "sina",
                        "title": title[:500],
                    })
        except Exception:
            break
        time.sleep(random.uniform(0.3, 0.6))
    print(f"  Sina global: {len(rows)} rows ({max_pages} pages)")
    return pd.DataFrame(rows)


# ============================================================
# 4. 巨潮资讯公司公告（历史回溯）
# ============================================================

def fetch_notice_report_backfill(dates: list[str]) -> pd.DataFrame:
    """巨潮资讯公司公告 — 按指定日期列表抓取，用于历史回溯。"""
    rows = []
    for date_str in dates:
        try:
            df = ak.stock_notice_report(symbol="全部", date=date_str)
            if df.empty:
                continue
            for _, row in df.iterrows():
                title = _clean_html(row["公告标题"])
                if not title:
                    continue
                code = _normalize_code(row["代码"])
                # 统一日期格式为 YYYY-MM-DD
                date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                rows.append({
                    "date": date_fmt,
                    "code": code,
                    "source": "cninfo",
                    "title": title[:500],
                })
        except Exception as e:
            print(f"  notice {date_str}: {e}")
        time.sleep(1.0)
        if len(rows) % 5000 == 0:
            print(f"  notice: {len(rows)} rows so far")
    print(f"  Notice backfill: {len(rows)} rows ({len(dates)} dates)")
    return pd.DataFrame(rows)


def _month_end_dates(start: str, end: str) -> list:
    """生成每月最后一天的日期列表。"""
    dr = pd.date_range(start, end, freq="ME")
    return [d.strftime("%Y%m%d") for d in dr]


# ============================================================
# 主流程
# ============================================================

def main():
    codes = _load_codes()
    print(f"Stock codes: {len(codes)}")

    all_frames = []

    # 1. 财联社电报（市场快讯）
    print("\n[1/4] CLS telegraph ...")
    all_frames.append(fetch_cls_telegraph())

    # 2. 东方财富个股新闻（全A股遍历）
    print("\n[2/4] East Money stock news (all stocks) ...")
    all_frames.append(fetch_em_stock_news(codes))

    # 3. 新浪财经7x24（市场快讯）
    print("\n[3/4] Sina 7x24 global news ...")
    all_frames.append(fetch_sina_global_news(max_pages=10))

    # 4. 巨潮资讯公告回溯（2020年起每月末）
    print("\n[4/4] CNINFO notice backfill (2020-01 ~ 2026-05 monthly) ...")
    notice_dates = _month_end_dates("2020-01-01", "2026-05-31")
    print(f"  Sampling {len(notice_dates)} month-end dates")
    all_frames.append(fetch_notice_report_backfill(notice_dates))

    # 合并 & 去重
    print("\nMerging & deduplicating ...")
    result = pd.concat(all_frames, ignore_index=True)

    # 过滤空 code（市场级新闻不关联具体股票）
    # 保留 code 为空的行，表示市场级别新闻

    # 再次清洗 title
    result["title"] = result["title"].apply(_clean_html)
    result = result[result["title"].str.len() > 0]

    # 去重
    before = len(result)
    result = result.drop_duplicates(subset=["date", "code", "source", "title"])
    print(f"  Dedup: {before} → {len(result)} rows ({before - len(result)} removed)")

    # 排序
    result = result.sort_values(["date", "code", "source"]).reset_index(drop=True)

    # 保存
    path = DATA_DIR / "news_raw.csv"
    result.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {path}")
    print(f"  Rows: {len(result)}")
    print(f"  Codes: {result['code'].nunique()}")
    print(f"  Date range: {result['date'].min()} — {result['date'].max()}")
    print(f"  Sources: {result['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
