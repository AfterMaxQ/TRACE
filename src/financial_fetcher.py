"""
上市公司财务报表采集模块 — AKShare 版（多线程）

资产负债表：ak.stock_balance_sheet_by_report_em
利润表：    ak.stock_profit_sheet_by_report_em
现金流量表：ak.stock_cash_flow_sheet_by_report_em

用法：
  python src/financial_fetcher.py                     # 全量
  python src/financial_fetcher.py --limit 100          # 测试
  python src/financial_fetcher.py --workers 8 --retries 5
"""
import os

os.environ["TQDM_DISABLE"] = "1"  # 必须在 akshare 导入之前，抑制其内部 tqdm

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
START_DATE = pd.Timestamp("2021-07-01")
END_DATE = pd.Timestamp("2026-05-26")
PROGRESS_FILE = os.path.join(DATA_DIR, ".financial_fetch_progress.json")

# ============================================================
BALANCE_SHEET_COLS = [
    "REPORT_DATE", "TOTAL_ASSETS", "TOTAL_LIABILITIES", "TOTAL_EQUITY",
    "TOTAL_PARENT_EQUITY", "TOTAL_CURRENT_ASSETS", "TOTAL_CURRENT_LIAB",
    "MONETARYFUNDS", "INVENTORY", "ACCOUNTS_RECE",
    "SHORT_LOAN", "LONG_LOAN", "TOTAL_NONCURRENT_ASSETS",
    "TOTAL_NONCURRENT_LIAB", "FIXED_ASSET", "INTANGIBLE_ASSET",
    "GOODWILL", "LONG_EQUITY_INVEST", "CIP",
    "PREPAYMENT", "CONTRACT_ASSET", "CONTRACT_LIAB",
    "ADVANCE_RECEIVABLES", "ACCOUNTS_PAYABLE", "MINORITY_EQUITY",
    "DEFER_TAX_ASSET", "DEFER_TAX_LIAB", "LEASE_LIAB",
    "USERIGHT_ASSET", "DEVELOP_EXPENSE", "TREASURY_SHARES",
    "SHARE_CAPITAL", "CAPITAL_RESERVE", "SURPLUS_RESERVE",
    "UNASSIGN_RPOFIT", "OTHER_RECE", "OTHER_PAYABLE",
    "INTEREST_RECE", "INTEREST_PAYABLE", "BOND_PAYABLE",
    "NOTE_ACCOUNTS_RECE",
]

INCOME_STATEMENT_COLS = [
    "REPORT_DATE", "TOTAL_OPERATE_INCOME", "OPERATE_INCOME", "TOTAL_OPERATE_COST",
    "OPERATE_COST", "OPERATE_PROFIT", "TOTAL_PROFIT",
    "NETPROFIT", "PARENT_NETPROFIT", "DEDUCT_PARENT_NETPROFIT",
    "BASIC_EPS", "SALE_EXPENSE", "MANAGE_EXPENSE",
    "RESEARCH_EXPENSE", "FINANCE_EXPENSE", "FE_INTEREST_EXPENSE",
    "FE_INTEREST_INCOME", "OPERATE_TAX_ADD", "ASSET_IMPAIRMENT_LOSS",
    "CREDIT_IMPAIRMENT_LOSS", "INVEST_INCOME", "INVEST_JOINT_INCOME",
    "FAIRVALUE_CHANGE_INCOME", "ASSET_DISPOSAL_INCOME",
    "NONBUSINESS_INCOME", "NONBUSINESS_EXPENSE", "EXCHANGE_INCOME",
    "OTHER_INCOME", "CONTINUED_NETPROFIT", "DISCONTINUED_NETPROFIT",
    "MINORITY_INTEREST", "INCOME_TAX", "TOTAL_COMPRE_INCOME",
]

CASH_FLOW_COLS = [
    "REPORT_DATE", "TOTAL_OPERATE_INFLOW", "TOTAL_OPERATE_OUTFLOW", "NETCASH_OPERATE",
    "SALES_SERVICES", "RECEIVE_TAX_REFUND", "BUY_SERVICES",
    "PAY_STAFF_CASH", "PAY_ALL_TAX",
    "NETCASH_INVEST", "CONSTRUCT_LONG_ASSET", "INVEST_PAY_CASH",
    "WITHDRAW_INVEST", "RECEIVE_INVEST_INCOME", "DISPOSAL_LONG_ASSET",
    "NETCASH_FINANCE", "RECEIVE_LOAN_CASH", "PAY_DEBT_CASH",
    "ISSUE_BOND", "ACCEPT_INVEST_CASH", "ASSIGN_DIVIDEND_PORFIT",
    "ASSET_IMPAIRMENT", "FA_IR_DEPR", "IA_AMORTIZE",
    "INVENTORY_REDUCE", "OPERATE_RECE_REDUCE", "OPERATE_PAYABLE_ADD",
    "CCE_ADD", "END_CCE",
]

STMT_CONFIG = [
    ("balance_sheet", ak.stock_balance_sheet_by_report_em, BALANCE_SHEET_COLS),
    ("income_statement", ak.stock_profit_sheet_by_report_em, INCOME_STATEMENT_COLS),
    ("cash_flow", ak.stock_cash_flow_sheet_by_report_em, CASH_FLOW_COLS),
]


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_stock_codes() -> list[tuple[str, str]]:
    csv_path = os.path.join(DATA_DIR, "company_info.csv")
    info_df = pd.read_csv(csv_path, dtype={"ts_code": str})
    codes = info_df["ts_code"].dropna().tolist()
    result = []
    for c in codes:
        c = c.strip()
        if "." in c:
            code_num, exchange = c.split(".")
            result.append((c, f"{exchange}{code_num}"))
        else:
            result.append((c, c))
    return result


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            print("[WARN] 进度文件损坏, 已忽略")
    return {}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def _filter_and_tag(df: pd.DataFrame, cols: list[str], ts_code: str) -> pd.DataFrame:
    available = [c for c in cols if c in df.columns]
    df = df[available].copy()
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"])
    df = df[(df["REPORT_DATE"] >= START_DATE) & (df["REPORT_DATE"] <= END_DATE)]
    if not df.empty:
        df["code"] = ts_code
    return df


def fetch_one_stock(ts_code: str, ak_symbol: str, retries: int, retry_wait: float) -> dict:
    """获取单只股票的三张报表，带重试。"""
    result = {"ts_code": ts_code, "data": {}}

    for name, ak_fn, cols in STMT_CONFIG:
        for attempt in range(1, retries + 1):
            try:
                raw = ak_fn(symbol=ak_symbol)
                if raw is not None and not raw.empty:
                    clean = _filter_and_tag(raw, cols, ts_code)
                    if not clean.empty:
                        result["data"][name] = clean
                        break  # 成功，跳出重试循环
                    else:
                        print(f"      {ts_code} {name}: empty after filtering")
                        break  # 有数据但过滤后为空，不重试
            except Exception as exc:
                msg = str(exc).lower()
                is_timeout = "timeout" in msg or "timed out" in msg
                is_rate = "rate limit" in msg or "too many" in msg
                wait = retry_wait * attempt
                if attempt < retries and (is_timeout or is_rate):
                    print(f"      {ts_code} {name} attempt {attempt}/{retries}: "
                          f"{'timeout' if is_timeout else 'rate-limited'}, "
                          f"sleeping {wait:.1f}s")
                    time.sleep(wait)
                elif attempt < retries:
                    # 非网络错误也重试，但更长时间等待
                    time.sleep(wait * 2)
                else:
                    print(f"      {ts_code} {name} FAILED after {retries} attempts: {exc}")

    return result


def download_all(
    pending: list[tuple[str, str]],
    retries: int,
    retry_wait: float,
    max_workers: int,
) -> dict[str, list[pd.DataFrame]]:
    """
    多线程采集。返回 {stmt_name: [DataFrame, ...]}。
    """
    frames: dict[str, list] = {name: [] for name, _, _ in STMT_CONFIG}
    failed: list[str] = []

    def _fetch_with_retry(ts_code: str, ak_sym: str):
        return fetch_one_stock(ts_code, ak_sym, retries, retry_wait)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_with_retry, ts, ak): (ts, ak)
                       for ts, ak in pending}

            total = len(futures)
            pbar = tqdm(total=total, unit="stock", desc="采集进度",
                        disable=False,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                   "[{elapsed}<{remaining}, {rate_fmt}] "
                                   "failed={postfix[0]}",
                        postfix=[0])
            for future in as_completed(futures):
                ts_code, ak_sym = futures[future]
                try:
                    result = future.result()
                    has_data = False
                    for name, _, _ in STMT_CONFIG:
                        if name in result["data"]:
                            frames[name].append(result["data"][name])
                            has_data = True
                    if not has_data:
                        failed.append(ts_code)
                except Exception as exc:
                    failed.append(ts_code)
                    tqdm.write(f"      {ts_code} unexpected error: {exc}")

                pbar.update(1)
                pbar.postfix[0] = len(failed)

            pbar.close()
    except KeyboardInterrupt:
        print("\n[中断] 用户取消, 正在停止...", flush=True)

    return frames, failed


def append_csv(path: str, df: pd.DataFrame, is_first: bool):
    df.to_csv(path, mode="w" if is_first else "a", header=is_first, index=False)


def run_collect(limit: int | None = None, workers: int = 8, retries: int = 5,
                retry_wait: float = 0.5, batch_size: int = 200):
    ensure_data_dir()
    all_codes = load_stock_codes()
    if limit:
        all_codes = all_codes[:limit]

    # 断点续采：从已有 CSV 恢复已完成列表
    done = set()
    csv_paths = {name: os.path.join(DATA_DIR, f"{name}.csv") for name, _, _ in STMT_CONFIG}
    for name, path in csv_paths.items():
        if os.path.exists(path):
            try:
                existing = pd.read_csv(path, usecols=["code"])
                done.update(existing["code"].unique().tolist())
            except Exception:
                pass

    # 恢复失败列表
    progress = load_progress()
    failed = set(progress.get("failed", []))

    pending = [(ts, ak) for ts, ak in all_codes if ts not in done and ts not in failed]
    total = len(all_codes)

    if not pending:
        print("[OK] 所有股票已完成")
        return

    print("=" * 60)
    print("  全 A 股财务报表采集")
    print("=" * 60)
    print(f"  总数: {total} 只, 已完成: {len(done)}, 待采集: {len(pending)}")
    print(f"  线程: {workers}, 重试: {retries}次, 间隔: {retry_wait}s")
    print(f"  批次: {batch_size} 只/批, 共 {-(len(pending) // -batch_size)} 批")
    print(f"  时间: {START_DATE.date()} — {END_DATE.date()}")
    print("=" * 60, flush=True)

    t0 = time.time()
    total_ok = 0
    total_fail = 0
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]

    for bi, batch in enumerate(batches, 1):
        print(f"\n--- 批次 {bi}/{len(batches)} ({len(batch)} 只) ---", flush=True)

        frames, new_failed = download_all(batch, retries, retry_wait, workers)

        # 每批完成立即写入 CSV
        for name, _, _ in STMT_CONFIG:
            lst = frames.get(name, [])
            if lst:
                df = pd.concat(lst, ignore_index=True)
                df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"])
                df = df.sort_values(["code", "REPORT_DATE"]).reset_index(drop=True)
                file_exists = os.path.exists(csv_paths[name])
                append_csv(csv_paths[name], df, not file_exists)
                print(f"  [OK] {name}.csv +{len(df)} 条 ({df['code'].nunique()} 只)")

        failed.update(new_failed)
        total_ok += len(batch) - len(new_failed)
        total_fail += len(new_failed)

        # 每批保存失败进度（中断可续）
        if failed:
            save_progress({"failed": sorted(failed)})

    elapsed = time.time() - t0
    print(f"\n[OK] 本次新增: {total_ok} 成功, {total_fail} 失败")
    print(f"[OK] 总耗时 {elapsed/60:.0f} 分钟")

    if not failed and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全A股财务报表采集")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-wait", type=float, default=0.5)
    args = parser.parse_args()

    run_collect(limit=args.limit, workers=args.workers, retries=args.retries,
                retry_wait=args.retry_wait)
