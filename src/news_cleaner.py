"""
新闻数据清洗模块

对 news_raw.csv 执行: 来源过滤、代码过滤、标题清洗、去重、质量标记

输出: data/news_clean.csv
用法: python src/news_cleaner.py
"""

import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"

# 丢弃的来源
DROP_SOURCES = {"cls", "sina"}

# 研报标题中需要去除的末尾括号模式 (券商名/作者标记)
import re

BRACKET_SUFFIX_PATTERNS = [
    re.compile(r"[（(][^）)]*证券[^）)]*[）)]$"),   # （中信证券）
    re.compile(r"[（(][^）)]*证券[^）)]*[）)]$"),   # (东方财富证券)
    re.compile(r"[（(][^）)]*研究[^）)]*[）)]$"),   # （深度研究）
    re.compile(r"[（(][^）)]*评级[^）)]*[）)]$"),   # （增持评级）
    re.compile(r"[（(][^）)]*新券[^）)]*[）)]$"),   # （打新新券）
    re.compile(r"[（(][^）)]*[、，,][^）)]*[）)]$"), # 通用: 末尾多内容括号
]


def _clean_title(title: str) -> tuple[str, bool]:
    """清洗单条标题: 去括号尾缀、去空白。返回 (cleaned, had_bracket)。"""
    if not isinstance(title, str):
        return "", False

    t = title.strip()
    had_bracket = False

    for pat in BRACKET_SUFFIX_PATTERNS:
        if pat.search(t):
            t = pat.sub("", t).strip()
            had_bracket = True
            break

    # 去除可能残留的空括号
    t = re.sub(r"[（(]\s*[）)]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip(), had_bracket


def main():
    print("=" * 60)
    print("  新闻数据清洗")
    print("=" * 60)

    # 1. 加载
    raw = pd.read_csv(DATA_DIR / "news_raw.csv")
    n0 = len(raw)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    print(f"[1] 原始数据: {n0:,} 行")

    # 2. 过滤来源
    before = len(raw)
    raw = raw[~raw["source"].isin(DROP_SOURCES)]
    print(f"[2] 过滤来源 (drop cls,sina): {before:,} → {len(raw):,} (-{before-len(raw):,})")

    # 3. 过滤无 code
    before = len(raw)
    raw = raw.dropna(subset=["code"])
    print(f"[3] 过滤无code: {before:,} → {len(raw):,} (-{before-len(raw):,})")

    # 4. 过滤 .BJ (北交所), 只保留 .SZ / .SH
    before = len(raw)
    raw = raw[raw["code"].str.endswith((".SZ", ".SH"))]
    print(f"[4] 过滤非.SZ/.SH (含.BJ): {before:,} → {len(raw):,} (-{before-len(raw):,})")

    # 5. 标题清洗
    cleaned = raw["title"].apply(_clean_title)
    raw["title"] = cleaned.apply(lambda x: x[0])
    raw["had_bracket_src"] = cleaned.apply(lambda x: x[1])
    n_bracket = raw["had_bracket_src"].sum()
    print(f"[5] 标题清洗: 去除 {n_bracket:,} 条括号尾缀")

    # 过滤清洗后为空的标题
    before = len(raw)
    raw = raw[raw["title"].str.len() > 0]
    print(f"    过滤空标题: {before:,} → {len(raw):,} (-{before-len(raw):,})")

    # 6. 去重: 同 date+code 保留最长标题
    before = len(raw)
    raw["title_len"] = raw["title"].str.len()
    raw["is_short"] = raw["title_len"] <= 10
    raw = raw.sort_values(["date", "code", "title_len"], ascending=[True, True, False])
    raw = raw.drop_duplicates(subset=["date", "code"], keep="first")
    print(f"[6] 去重 (per date+code): {before:,} → {len(raw):,} (-{before-len(raw):,})")

    # 7. 排序输出
    out_cols = ["date", "code", "source", "title", "title_len", "is_short"]
    result = raw[out_cols].sort_values(["date", "code"]).reset_index(drop=True)
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")

    out_path = DATA_DIR / "news_clean.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 摘要
    print(f"\n{'='*60}")
    print(f"[OK] news_clean.csv: {len(result):,} 行")
    print(f"     代码: {result['code'].nunique():,} 只")
    print(f"     日期: {result['date'].min()} – {result['date'].max()}")
    print(f"     来源: {dict(result['source'].value_counts())}")
    print(f"     均值长度: {result['title_len'].mean():.0f} 字")
    print(f"     短标题: {result['is_short'].sum():,} 条")
    print(f"     总减少: {n0:,} → {len(result):,} ({(1-len(result)/n0)*100:.1f}%)")


if __name__ == "__main__":
    main()
