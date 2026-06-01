"""
企业海外收入占比采集模块

数据源: 东方财富 主营构成（按地区），通过 akshare stock_zygc_em 获取
输出: data/overseas_revenue.csv — 每公司×报告期的海外收入占比

用法:
  python src/overseas_revenue.py                     # 全量
  python src/overseas_revenue.py --limit 200          # 测试
  python src/overseas_revenue.py --workers 12
"""

import os

os.environ["TQDM_DISABLE"] = "1"

import time
import json
import argparse
import warnings
import pandas as pd
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
START_DATE = pd.Timestamp("2018-01-01")
END_DATE = pd.Timestamp("2026-05-26")
PROGRESS_FILE = os.path.join(DATA_DIR, ".overseas_revenue_progress.json")

# 海外地区关键词
OVERSEAS_KEYWORDS = [
    "国外", "海外", "境外", "出口", "港澳台",
    "亚洲", "欧洲", "美洲", "非洲", "大洋洲",
    "北美", "南美", "东南亚", "中东", "欧盟",
    "日本", "韩国", "美国", "德国", "英国",
    "其他业务(地区)", "其他地区", "其他国家和地区",
]

# 明确不是海外的关键词（排除项）
DOMESTIC_KEYWORDS = [
    "中国大陆", "中国境内", "国内", "境内",
    "华东", "华南", "华北", "华中", "东北", "西北", "西南",
    "中国（", "中国(",  # "中国(含港澳台)" 等含中国的复合标签
]


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_stock_codes() -> list[tuple[str, str]]:
    """从 company_info 加载全 A 股代码，转换为东方财富格式。"""
    csv_path = os.path.join(DATA_DIR, "company_info.csv")
    info_df = pd.read_csv(csv_path, dtype={"ts_code": str})
    codes = info_df["ts_code"].dropna().tolist()
    result = []
    for c in codes:
        c = c.strip()
        if "." in c:
            num, ex = c.split(".")
            # Tushare 格式 → 东方财富格式: 000001.SZ → SZ000001
            result.append((c, f"{ex}{num}"))
    return result


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def is_overseas(name: str) -> bool:
    """判断经营项目名称是否为海外地区。"""
    if not isinstance(name, str):
        return False
    # 先排除明确国内关键词
    for kw in DOMESTIC_KEYWORDS:
        if kw in name:
            return False
    for kw in OVERSEAS_KEYWORDS:
        if kw in name:
            return True
    return False


def fetch_one_stock(ts_code: str, ak_symbol: str) -> dict | None:
    """获取单只股票的分地区收入，提取海外收入占比。"""
    try:
        raw = ak.stock_zygc_em(symbol=ak_symbol)
    except Exception:
        return None

    if raw is None or raw.empty:
        return None

    # 列名(字节确认): 股票代码, 报告日期, 分类类型, 主营构成, 主营收入, 收入比例,
    #                  主营成本, 成本比例, 主营利润, 利润比例, 毛利率
    cols = list(raw.columns)
    date_col = cols[1]   # 报告日期
    dir_col = cols[2]    # 分类类型
    item_col = cols[3]   # 主营构成
    rev_col = cols[4]    # 主营收入
    pct_col = cols[5]    # 收入比例

    raw[date_col] = pd.to_datetime(raw[date_col])
    raw = raw[(raw[date_col] >= START_DATE) & (raw[date_col] <= END_DATE)]

    region = raw[raw[dir_col] == "按地区分类"]
    if region.empty:
        return None

    records = []
    for report_date, grp in region.groupby(date_col):
        total_rev = grp[rev_col].sum()
        if total_rev <= 0:
            continue

        overseas_mask = grp[item_col].apply(is_overseas)
        overseas_rev = grp.loc[overseas_mask, rev_col].sum()
        overseas_pct = overseas_rev / total_rev
        # 截断异常值: 部分公司有分部抵消导致 ratio > 1
        overseas_pct = max(0.0, min(1.0, overseas_pct))

        # 列出海外地区明细
        overseas_items = grp.loc[overseas_mask, item_col].tolist()

        records.append({
            "ts_code": ts_code,
            "report_date": report_date.strftime("%Y-%m-%d"),
            "total_revenue": round(total_rev, 2),
            "overseas_revenue": round(min(overseas_rev, total_rev), 2),
            "overseas_ratio": round(overseas_pct, 6),
            "region_count": len(grp),
            "overseas_regions": "|".join(overseas_items),
        })

    return records if records else None


def download_all(
    pending: list[tuple[str, str]],
    max_workers: int,
) -> tuple[list[dict], list[str]]:
    """多线程采集，返回 (记录列表, 失败列表)。"""
    all_records: list[dict] = []
    failed: list[str] = []

    def _fetch(ts_code: str, ak_sym: str):
        return fetch_one_stock(ts_code, ak_sym)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, ts, ak): (ts, ak) for ts, ak in pending}

        total = len(futures)
        pbar = tqdm(
            total=total, unit="stock", desc="采集进度",
            disable=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}] "
                       "hit={postfix[0]} fail={postfix[1]}",
            postfix=[0, 0],
        )
        hit_count = 0

        for future in as_completed(futures):
            ts_code, ak_sym = futures[future]
            try:
                result = future.result()
                if result:
                    all_records.extend(result)
                    hit_count += 1
                # 无地区数据的股票不算失败，只是跳过
            except Exception:
                failed.append(ts_code)

            pbar.update(1)
            pbar.postfix[0] = hit_count
            pbar.postfix[1] = len(failed)

        pbar.close()

    return all_records, failed


def run_collect(limit: int | None = None, workers: int = 12):
    ensure_data_dir()
    all_codes = load_stock_codes()
    if limit:
        all_codes = all_codes[:limit]

    # 断点续采
    output_path = os.path.join(DATA_DIR, "overseas_revenue.csv")
    done = set()
    if os.path.exists(output_path):
        try:
            existing = pd.read_csv(output_path, usecols=["ts_code"])
            done.update(existing["ts_code"].unique().tolist())
        except Exception:
            pass

    progress = load_progress()
    prev_failed = set(progress.get("failed", []))

    pending = [(ts, ak) for ts, ak in all_codes
               if ts not in done and ts not in prev_failed]
    total = len(all_codes)

    if not pending:
        print("[OK] 所有股票已完成")
        return

    print("=" * 60)
    print("  全 A 股海外收入占比采集")
    print("=" * 60)
    print(f"  总数: {total} 只, 已完成: {len(done)}, 待采集: {len(pending)}")
    print(f"  线程: {workers}")
    print(f"  时间: {START_DATE.date()} — {END_DATE.date()}")
    print("=" * 60, flush=True)

    t0 = time.time()
    batch_size = 500
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    all_failed: list[str] = []
    total_hit = 0

    for bi, batch in enumerate(batches, 1):
        print(f"\n--- 批次 {bi}/{len(batches)} ({len(batch)} 只) ---", flush=True)

        records, failed = download_all(batch, workers)
        all_failed.extend(failed)
        total_hit += len({r["ts_code"] for r in records})

        if records:
            df = pd.DataFrame(records)
            df["report_date"] = pd.to_datetime(df["report_date"])
            df = df.sort_values(["ts_code", "report_date"]).reset_index(drop=True)
            file_exists = os.path.exists(output_path)
            df.to_csv(
                output_path,
                mode="a" if file_exists else "w",
                header=not file_exists,
                index=False,
            )
            print(f"  [OK] +{len(df)} 条 ({df['ts_code'].nunique()} 只有地区数据)")
        else:
            print(f"  [--] 批次无命中")

        if all_failed:
            save_progress({"failed": sorted(all_failed)})

    elapsed = time.time() - t0
    print(f"\n[OK] 有海外收入的股票: {total_hit} 只")
    print(f"[OK] 总耗时 {elapsed/60:.0f} 分钟")
    print(f"[OK] 输出: {output_path}")

    if not all_failed and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全A股海外收入占比采集")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    run_collect(limit=args.limit, workers=args.workers)
