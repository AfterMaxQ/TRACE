"""
多源数据采集模块 — yfinance 版

股票日线：yfinance 批量下载（全 A 股）
大盘指数：yfinance 沪深300 ETF (ASHR) + AKShare 备选
国债收益：AKShare bond_zh_us_rate（已稳定）

输出文件到 data/:
  - bond_yields.csv        中债国债收益率（日频: 2Y/10Y）
  - csi300_daily.csv 沪深300指数日线
  - stock_daily.csv        全A股前复权日线 OHLCV
"""
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak
import warnings
warnings.filterwarnings("ignore")

# 代理配置 — yfinance 底层 requests 会读取环境变量
PROXY_URL = "http://127.0.0.1:7897"
os.environ["HTTP_PROXY"] = PROXY_URL
os.environ["HTTPS_PROXY"] = PROXY_URL

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
START_DATE = "2021-07-01"
END_DATE = "2026-05-26"
START_DT = pd.Timestamp(START_DATE)
END_DT = pd.Timestamp(END_DATE)


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# 1. 股票日线 — yfinance
# ============================================================
def _make_yf_tickers(ts_codes: list[str]) -> list[str]:
    """将 Tushare ts_code (.SH/.SZ/.BJ) 转为 yfinance 格式 (.SS/.SZ/.BJ)"""
    result = []
    for c in ts_codes:
        c = str(c).strip()
        if c.endswith(".SH"):
            result.append(c.replace(".SH", ".SS"))
        elif c.endswith(".SZ") or c.endswith(".BJ"):
            result.append(c)
        elif len(c) == 6:
            if c.startswith(("6", "9")):
                result.append(f"{c}.SS")
            elif c.startswith(("0", "3")):
                result.append(f"{c}.SZ")
            elif c.startswith(("8", "4")):
                result.append(f"{c}.BJ")
        else:
            result.append(c)
    return result


def fetch_all_stocks_yf(csv_path: str | None = None,
                        batch_size: int = 500,
                        save_csv: bool = True) -> pd.DataFrame:
    """
    从上市公司基本信息 CSV 读取 ts_code 列表，
    用 yfinance 批量下载日线 OHLCV。

    yfinance 单次不宜超过 ~800 ticker，这里默认 500 一批。
    """
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "company_info.csv")

    info_df = pd.read_csv(csv_path)
    raw_codes = info_df["ts_code"].dropna().astype(str).tolist()
    tickers = _make_yf_tickers(raw_codes)
    tickers = sorted(set(tickers))

    total = len(tickers)
    print(f"[..] 全 A 股 ticker 共 {total} 只，分批下载 (batch={batch_size})...")

    all_frames = []
    failed_batches = []

    for i in range(0, total, batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        try:
            chunk = yf.download(
                tickers=batch,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,  # 自动前复权
                progress=False,
                threads=True,
            )
            if chunk.empty:
                failed_batches.append(f"batch_{batch_num}")
                print(f"  [{batch_num}/{total_batches}] 空数据, 跳过")
                continue

            # yf.download 多 ticker 返回 MultiIndex columns: (OHLCV, ticker)
            if isinstance(chunk.columns, pd.MultiIndex):
                melted = _melt_yf_multiticker(chunk)
            else:
                # 单 ticker
                chunk.columns = [c.lower() for c in chunk.columns]
                chunk["ticker"] = batch[0]
                melted = chunk.reset_index()
                if melted.columns[0] not in ("Date", "date"):
                    melted = melted.rename(columns={melted.columns[0]: "Date"})

            all_frames.append(melted)
            print(f"  [{batch_num}/{total_batches}] OK ({len(melted)} 条, {melted['ticker'].nunique()} 只)")

        except Exception as e:
            failed_batches.append(f"batch_{batch_num}")
            print(f"  [{batch_num}/{total_batches}] 异常: {e}")
            time.sleep(5)

        # 批次间暂停
        if i + batch_size < total:
            time.sleep(3)

    if not all_frames:
        raise RuntimeError("未获取到任何股票数据")

    result = pd.concat(all_frames, ignore_index=True)

    # 标准化列名
    result = result.rename(columns={
        "ticker": "code",
        "Date": "date",
        "Open": "open", "High": "high",
        "Low": "low", "Close": "close",
        "Volume": "volume",
    })
    # 小写兼容
    result = result.rename(columns={
        "date": "date", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "volume",
    })

    result["date"] = pd.to_datetime(result["date"])
    # 统一代码格式: yfinance .SS -> Tushare .SH
    result["code"] = result["code"].str.replace(".SS", ".SH", regex=False)
    # 确保有必需的列
    for col in ["open", "high", "low", "close", "volume", "code"]:
        if col not in result.columns:
            raise KeyError(f"缺失列: {col}")

    result = result.sort_values(["code", "date"]).reset_index(drop=True)

    if save_csv:
        path = os.path.join(DATA_DIR, "stock_daily.csv")
        result.to_csv(path, index=False)
        n = result["code"].nunique()
        print(f"[OK] 个股行情 -> {path}  ({len(result)} 条, {n} 只)")

    return result


def _melt_yf_multiticker(chunk: pd.DataFrame) -> pd.DataFrame:
    """将 yfinance MultiIndex columns 展开为长表。"""
    # columns: MultiIndex [(Open, AAPL), (High, AAPL), ...]
    frames = []
    tickers = chunk.columns.get_level_values(1).unique()
    for t in tickers:
        sub = chunk.xs(t, axis=1, level=1).copy()
        sub.columns = [c.lower() for c in sub.columns]
        sub["ticker"] = t
        sub = sub.reset_index()
        # yfinance 不同版本返回的 index name 可能为 None/Date/date
        if sub.columns[0] not in ("Date", "date"):
            sub = sub.rename(columns={sub.columns[0]: "Date"})
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


# ============================================================
# 2. 沪深300指数
# ============================================================
def fetch_csi300_index(save_csv: bool = True) -> pd.DataFrame:
    """
    尝试 yfinance ASHR (沪深300 ETF) → 失败则用 AKShare 指数日线。
    """
    df = None

    # 尝试 yfinance
    try:
        raw = yf.download("ASHR", start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)
        if not raw.empty:
            raw = raw.reset_index()
            raw = raw.rename(columns={
                "Date": "date", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume",
            })
            raw.columns = [c.lower() for c in raw.columns]
            raw["date"] = pd.to_datetime(raw["date"])
            df = raw[["date", "open", "high", "low", "close", "volume"]].copy()
            print(f"  [..] 使用 yfinance ASHR 作为沪深300代理")
    except Exception as e:
        print(f"  [!!] yfinance ASHR 失败: {e}")

    # 回退 AKShare
    if df is None or df.empty:
        try:
            raw = ak.stock_zh_index_daily(symbol="sh000300")
            raw["date"] = pd.to_datetime(raw["date"])
            df = raw[(raw["date"] >= START_DT) & (raw["date"] <= END_DT)].copy()
            df = df[["date", "open", "high", "low", "close", "volume"]].copy()
            print(f"  [..] 回退 AKShare 沪深300 指数")
        except Exception as e:
            print(f"  [!!] AKShare 指数也失败: {e}")
            return pd.DataFrame()

    df = df.sort_values("date").reset_index(drop=True)

    if save_csv and not df.empty:
        path = os.path.join(DATA_DIR, "csi300_daily.csv")
        df.to_csv(path, index=False)
        print(f"[OK] 沪深300指数 -> {path}  ({len(df)} 条)")

    return df


# ============================================================
# 3. 国债收益率
# ============================================================
def fetch_bond_yields(save_csv: bool = True) -> pd.DataFrame:
    """
    中债国债收益率 (1Y/10Y)，来源 AKShare bond_china_yield。
    API 限制单次查询跨度不能太长，按年分块请求后合并。
    """
    frames = []
    for yr in range(START_DT.year, END_DT.year + 1):
        s = f"{yr}0101"
        e = f"{yr}1231"
        try:
            raw = ak.bond_china_yield(start_date=s, end_date=e)
            gb = raw[raw["曲线名称"] == "中债国债收益率曲线"].copy()
            if gb.empty:
                continue
            gb["date"] = pd.to_datetime(gb["日期"])
            gb["yield_1y"] = pd.to_numeric(gb["1年"], errors="coerce")
            gb["yield_10y"] = pd.to_numeric(gb["10年"], errors="coerce")
            frames.append(gb[["date", "yield_1y", "yield_10y"]])
        except Exception as exc:
            print(f"  [!!] bond {yr}: {exc}")

    if not frames:
        print("[!!] bond_china_yield 全部失败，回退 bond_zh_us_rate")
        return _fetch_bond_yields_fallback(save_csv)

    gb = pd.concat(frames, ignore_index=True)
    gb = gb[(gb["date"] >= START_DT) & (gb["date"] <= END_DT)]
    gb = gb.dropna(subset=["yield_1y", "yield_10y"])
    gb = gb.sort_values("date").reset_index(drop=True)

    if save_csv:
        path = os.path.join(DATA_DIR, "bond_yields.csv")
        gb.to_csv(path, index=False)
        print(f"[OK] 国债收益率 -> {path}  ({len(gb)} 条, 1Y+10Y)")

    return gb


def _fetch_bond_yields_fallback(save_csv: bool = True) -> pd.DataFrame:
    """回退方案: bond_zh_us_rate (仅 2Y+10Y，无 1Y)。"""
    raw = ak.bond_zh_us_rate()
    date_col = raw.columns[0]
    col_map = {}
    for c in raw.columns:
        if "中国国债收益率" in c and "10年" in c and "-" not in c:
            col_map[c] = "yield_10y"
        elif "中国国债收益率" in c and "2年" in c and "-" not in c:
            col_map[c] = "yield_2y"
        elif "中国国债收益率" in c and "1年" in c and "-" not in c:
            col_map[c] = "yield_1y"
    df = raw[[date_col] + list(col_map.keys())].rename(columns=col_map)
    df["date"] = pd.to_datetime(df[date_col])
    cols = ["date"] + [v for v in col_map.values() if v in df.columns]
    df = df[cols]
    df = df[(df["date"] >= START_DT) & (df["date"] <= END_DT)]
    df = df.sort_values("date").reset_index(drop=True)
    if save_csv:
        path = os.path.join(DATA_DIR, "bond_yields.csv")
        df.to_csv(path, index=False)
        print(f"[OK] 国债收益率(回退) -> {path}  ({len(df)} 条)")
    return df


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    ensure_data_dir()
    print("=" * 55)
    print("  TRACE 数据采集 (yfinance)")
    print(f"  时间范围: {START_DATE} — {END_DATE}")
    print("=" * 55)

    # 1. 国债
    print("\n[1/3] 中债国债收益率")
    bond_df = fetch_bond_yields()
    if not bond_df.empty:
        d = str(bond_df["date"].iloc[-1])[:10]
        y2 = float(bond_df["yield_2y"].iloc[-1])
        y10 = float(bond_df["yield_10y"].iloc[-1])
        print(f"       {d}  2Y={y2:.4f}%  10Y={y10:.4f}%")

    # 2. 沪深300
    print("\n[2/3] 沪深300指数")
    csi300_df = fetch_csi300_index()
    if not csi300_df.empty:
        d = str(csi300_df["date"].iloc[-1])[:10]
        v = float(csi300_df["close"].iloc[-1])
        print(f"       {d}  收盘={v:.2f}")

    # 3. 全 A 股
    print("\n[3/3] 全 A 股日线")
    stock_df = fetch_all_stocks_yf()

    print("\n" + "=" * 55)
    print("  采集完成")
    print("=" * 55)
