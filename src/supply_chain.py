"""
供应链与股权关联数据采集模块

数据源：
  - Tushare Pro: top10_holders (前十大股东), pledge_stat (质押统计)
  - 手工整理: supply_chain_edges.csv (前5大客户/供应商)

输出：
  - data/share_holders.csv     前十大股东（最新报告期）
  - data/pledge_stat.csv       股权质押统计（最新）
"""
import time
import pandas as pd
import tushare as ts
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Tushare Pro token
TS_TOKEN = "4353d440506e8ba010599e3edc686356fa76d24ba66a00f693595de1"


def _load_codes() -> list[str]:
    """加载全A股代码列表。"""
    csv_path = DATA_DIR / "TRACE_上市公司基本信息.csv"
    info = pd.read_csv(csv_path, dtype={"ts_code": str})
    return sorted(info["ts_code"].dropna().unique())


def _short_code(ts_code: str) -> str:
    """000001.SZ -> 000001"""
    return ts_code.split(".")[0]


# ============================================================
# 1. 前十大股东
# ============================================================

def fetch_top10_holders(codes: list[str]) -> pd.DataFrame:
    """从 Tushare 获取指定股票最新一期前十大股东。

    免费版限频 ~1次/分钟，只采集供应链相关股票。
    """
    print(f"[1/3] 采集前十大股东 (限频, {len(codes)} 只) ...")
    pro = ts.pro_api(TS_TOKEN)
    all_rows = []

    for i, code in enumerate(codes):
        try:
            df = pro.top10_holders(
                ts_code=code,
                start_date="20240101",
                end_date="20261231",
            )
            if df.empty:
                print(f"  [{code}] 无数据")
                continue
            latest_date = df["end_date"].max()
            latest = df[df["end_date"] == latest_date].copy()
            all_rows.append(latest)
            print(f"  [{i+1}/{len(codes)}] {code} OK ({len(latest)} holders)")
        except Exception as e:
            msg = str(e)[:80]
            print(f"  [{i+1}/{len(codes)}] {code} SKIP: {msg}")
            continue

        if i < len(codes) - 1:
            time.sleep(62)  # 1次/分钟限频

    if not all_rows:
        print("[!!] 未获取到股东数据")
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    # 保留关键列
    cols = ["ts_code", "end_date", "holder_name", "hold_amount",
            "hold_ratio", "hold_float_ratio", "holder_type"]
    result = result[[c for c in cols if c in result.columns]]
    result = result.sort_values(["ts_code", "hold_ratio"], ascending=[True, False])
    result = result.reset_index(drop=True)

    path = DATA_DIR / "share_holders.csv"
    result.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [OK] share_holders.csv: {len(result)} 行, {result['ts_code'].nunique()} 只")
    return result


# ============================================================
# 2. 股权质押统计
# ============================================================

def fetch_pledge_stat(codes: list[str]) -> pd.DataFrame:
    """从 Tushare 获取全A股最新股权质押统计。"""
    print("[2/3] 采集股权质押数据 ...")
    pro = ts.pro_api(TS_TOKEN)
    all_rows = []

    for i, code in enumerate(codes):
        try:
            df = pro.pledge_stat(ts_code=code)
            if df.empty:
                continue
            latest = df[df["end_date"] == df["end_date"].max()].copy()
            all_rows.append(latest)
        except Exception:
            continue

        if (i + 1) % 500 == 0:
            print(f"  pledge_stat: {i + 1}/{len(codes)} stocks")
        time.sleep(0.12)

    if not all_rows:
        print("[!!] 未获取到质押数据")
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    result = result.sort_values("pledge_ratio", ascending=False)
    result = result.reset_index(drop=True)

    path = DATA_DIR / "pledge_stat.csv"
    result.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [OK] pledge_stat.csv: {len(result)} 行, {result['ts_code'].nunique()} 只")
    return result


# ============================================================
# 3. 供应商/客户关系（手工维护 + 校验）
# ============================================================

def load_supply_chain_edges() -> pd.DataFrame:
    """加载手工维护的供应链关系表并做基本校验。"""
    path = DATA_DIR / "supply_chain_edges.csv"
    if not path.exists():
        print(f"[!!] {path} 不存在，请先手工创建")
        return pd.DataFrame()

    df = pd.read_csv(path, dtype=str)
    print(f"[3/3] 供应链关系: {len(df)} 条边, {df['source_code'].nunique()} 只股票")

    # 校验
    codes = _load_codes()
    unknown = set(df["source_code"].unique()) - set(codes)
    if unknown:
        print(f"  [!] 未知代码: {unknown}")

    # 检查关系类型
    valid_types = {"customer", "supplier", "shareholder", "subsidiary", "parent"}
    invalid = set(df["relation_type"].unique()) - valid_types
    if invalid:
        print(f"  [!] 无效关系类型: {invalid}")

    return df


def main():
    codes = _load_codes()

    # 优先采集供应链CSV中的股票
    sc_path = DATA_DIR / "supply_chain_edges.csv"
    if sc_path.exists():
        sc = pd.read_csv(sc_path, dtype=str)
        target_codes = sorted(sc["source_code"].unique().tolist())
        print(f"供应链股票: {len(target_codes)} 只")
    else:
        # 回退到全量（会很慢）
        target_codes = sorted(codes)[:50]
        print(f"供应链CSV不存在，取前50只测试: {len(target_codes)}")

    print(f"Tushare Pro token: {'*' * 8}...")

    fetch_top10_holders(target_codes)
    print()

    # pledge_stat 1次/小时限频，单独采集重点5家
    focus_codes = ["300750.SZ", "002594.SZ", "002475.SZ", "601012.SH", "600519.SH"]
    print("[2/3] 采集股权质押 (重点5家) ...")
    pro = ts.pro_api(TS_TOKEN)
    pledge_rows = []
    for i, code in enumerate(focus_codes):
        try:
            df = pro.pledge_stat(ts_code=code)
            if not df.empty:
                latest = df[df["end_date"] == df["end_date"].max()]
                pledge_rows.append(latest)
                print(f"  [{code}] OK ({len(latest)} rows)")
        except Exception as e:
            print(f"  [{code}] SKIP: {str(e)[:80]}")
        if i < len(focus_codes) - 1:
            time.sleep(5)

    if pledge_rows:
        pledge = pd.concat(pledge_rows, ignore_index=True)
        pledge.to_csv(DATA_DIR / "pledge_stat.csv", index=False, encoding="utf-8-sig")
        print(f"  [OK] pledge_stat.csv: {len(pledge)} 行")
    else:
        print("  [!!] 未获取到质押数据")

    print()
    load_supply_chain_edges()

    print("\n完成。")


if __name__ == "__main__":
    main()
