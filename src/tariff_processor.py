"""
美国进口关税数据处理器

从 USITC 年度 HTS8 级关税数据库中提取 MFN 从价税率，聚合到 HS2 章。

输出: data/hs2_tariff_annual.csv
用法: python src/tariff_processor.py
"""

import os
import zipfile
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
TARIFF_DIR = DATA_DIR / "tariff_data"

RATE_COLS = [
    "hts8", "brief_description",
    "mfn_text_rate", "mfn_rate_type_code",
    "mfn_ad_val_rate", "mfn_specific_rate", "mfn_other_rate",
    "begin_effect_date", "end_effective_date",
]

# 哨兵值: tariff database 用此值表示 "不可计算"
SENTINEL = 9999.0

# 年份列表
YEARS = list(range(2016, 2027))


def _read_tariff_txt(zip_path: Path) -> pd.DataFrame:
    """从 zip 中读取 tariff txt 文件，处理三种分隔符格式。"""
    with zipfile.ZipFile(zip_path) as z:
        txt_names = [n for n in z.namelist() if n.endswith(".txt")]
        if not txt_names:
            return pd.DataFrame()
        raw = z.read(txt_names[0]).decode("latin-1")

    # 判断分隔符
    first_line = raw.split("\n")[0]
    if "|" in first_line:
        sep = "|"
    elif first_line.count(",") > 20:
        sep = ","
    else:
        sep = ","  # fallback

    from io import StringIO
    return pd.read_csv(StringIO(raw), sep=sep, dtype=str, low_memory=False)


def _clean_rate(df: pd.DataFrame) -> pd.DataFrame:
    """提取并清洗关税税率字段。"""
    available = [c for c in RATE_COLS if c in df.columns]
    df = df[available].copy()

    # 数值转换
    for col in ["mfn_ad_val_rate", "mfn_specific_rate", "mfn_other_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 日期列
    for col in ["begin_effect_date", "end_effective_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 过滤: 仅保留非哨兵值的 ad valorem (type 7) 和 free (type 0)
    df = df[df["hts8"].notna() & (df["hts8"].str.strip() != "")]
    df["hs2"] = df["hts8"].str.strip().str[:2]

    # 过滤非标准 HS 章 (98/99 为特殊条款)
    df = df[df["hs2"].str.match(r"^\d{2}$")]
    df = df[~df["hs2"].isin(["98", "99"])]

    # MFN 税率类型
    df["rate_type"] = df.get("mfn_rate_type_code", pd.Series(dtype=str))
    # 标记可用行: type 0 (free) 或 type 7 (pure ad valorem)
    df["usable"] = df["rate_type"].isin(["0", "7"])
    df.loc[df["usable"], "ad_val"] = df.loc[df["usable"], "mfn_ad_val_rate"].fillna(0.0)
    df.loc[df["rate_type"] == "0", "ad_val"] = 0.0

    # 标记哨兵值
    sentinel_mask = (
        (df["mfn_ad_val_rate"] > SENTINEL / 2) |
        (df["mfn_specific_rate"] > SENTINEL / 2)
    )
    df.loc[sentinel_mask, "usable"] = False

    return df


def _yearly_rate(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """计算每个 HS2 章的年度有效税率 (7月1日快照)。"""
    snapshot_date = pd.Timestamp(f"{year}-07-01")

    # 对每个 HTS8 行, 判断该年是否在有效期内
    active = df[
        (df["begin_effect_date"] <= snapshot_date) &
        (df["end_effective_date"] >= snapshot_date)
    ].copy()

    if active.empty:
        return pd.DataFrame()

    # 按 HS2 聚合
    usable = active[active["usable"]]
    total_by_hs2 = active.groupby("hs2").size()
    usable_by_hs2 = usable.groupby("hs2").size()

    stats = usable.groupby("hs2")["ad_val"].agg(["mean", "median", "std", "count"])
    stats.columns = ["avg_tariff", "median_tariff", "std_tariff", "n_usable"]
    stats["n_total"] = total_by_hs2
    stats["coverage_ratio"] = stats["n_usable"] / stats["n_total"]
    stats["year"] = year

    return stats.reset_index()


def main():
    zips = sorted(TARIFF_DIR.glob("tariff_data_*.zip"))
    if not zips:
        print("[!!] 未找到 tariff zip 文件")
        return

    print(f"找到 {len(zips)} 个 tariff zip ({zips[0].stem[-4:]}–{zips[-1].stem[-4:]})")

    all_rates = []
    for zp in zips:
        year_tag = zp.stem[-4:]  # e.g. "2024"
        print(f"  [{year_tag}] 读取...", end=" ", flush=True)
        try:
            raw = _read_tariff_txt(zp)
            clean = _clean_rate(raw)
            print(f"{len(clean)} 行 (usable={clean['usable'].sum():.0f})")
            all_rates.append(clean)
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_rates:
        print("[!!] 无有效数据")
        return

    df = pd.concat(all_rates, ignore_index=True)
    total_hts8 = df["hts8"].nunique()
    print(f"\n总计: {len(df)} 行, {total_hts8} 个唯一 HTS8 码")

    # 逐年聚合
    results = []
    for yr in YEARS:
        yr_df = _yearly_rate(df, yr)
        if not yr_df.empty:
            results.append(yr_df)
            cov = yr_df["coverage_ratio"].mean()
            print(f"  {yr}: {len(yr_df)} HS2 章, 平均覆盖率 {cov:.1%}")

    out_df = pd.concat(results, ignore_index=True)
    out_df["hs2"] = out_df["hs2"].astype(int)
    out_df = out_df.sort_values(["year", "hs2"]).reset_index(drop=True)

    # 圆整
    for c in ["avg_tariff", "median_tariff", "std_tariff", "coverage_ratio"]:
        if c in out_df.columns:
            out_df[c] = out_df[c].round(6)

    out_path = DATA_DIR / "hs2_tariff_annual.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] hs2_tariff_annual.csv: {len(out_df)} 行 "
          f"({out_df['hs2'].nunique()} HS2 章 × {out_df['year'].nunique()} 年)")


if __name__ == "__main__":
    main()
