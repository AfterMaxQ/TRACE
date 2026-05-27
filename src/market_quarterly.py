"""
季度市场指标计算模块

从 stock_daily.csv 读取全A股日线，计算每只股票每季度的：
  - 对数收益率：ln(季度末收盘价 / 季度初开盘价)
  - 日收益率波动率：季度内日对数收益率的标准差
  - 最大回撤：季度内收盘价相对区间最高点的最大跌幅
  - Beta：基于过去12个月日收益率对沪深300指数的滚动回归

输出 data/market_quarterly.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def _max_drawdown(close: pd.Series) -> float:
    """计算序列的最大回撤（从峰值到谷底的最大跌幅比例，负值）。"""
    if len(close) < 2:
        return np.nan
    running_max = close.expanding().max()
    drawdown = close / running_max - 1.0
    return drawdown.min()


def _log_return(close: pd.Series) -> float:
    """计算序列的对数收益率：ln(末值 / 初值)。"""
    if len(close) < 2:
        return np.nan
    return float(np.log(close.iloc[-1] / close.iloc[0]))


def _calc_quarterly_beta(
    daily_ret: pd.Series,
    index_ret: pd.Series,
    quarters: pd.DatetimeIndex,
    min_samples: int = 60,
) -> dict:
    """计算每个季度末的滚动Beta（过去12个月日收益率回归）。"""
    betas = {}
    for q_end in quarters:
        start = q_end - pd.DateOffset(months=12)
        mask_s = (daily_ret.index >= start) & (daily_ret.index <= q_end)
        mask_m = (index_ret.index >= start) & (index_ret.index <= q_end)
        s = daily_ret[mask_s]
        m = index_ret[mask_m]
        common = s.index.intersection(m.index)
        if len(common) < min_samples:
            betas[q_end] = np.nan
            continue
        cov = np.cov(s[common].values, m[common].values, ddof=1)[0, 1]
        betas[q_end] = cov / np.var(m[common].values, ddof=1)
    return betas


def main():
    print("Loading stock_daily.csv ...")
    stock_df = pd.read_csv(
        DATA_DIR / "stock_daily.csv",
        parse_dates=["date"],
        usecols=["date", "close", "code"],
    )
    # 归一化后缀: yfinance .SS → Tushare .SH (与 data_fetcher.py 一致)
    stock_df["code"] = stock_df["code"].str.replace(".SS", ".SH", regex=False)

    print("Loading csi300_index_daily.csv ...")
    csi300 = pd.read_csv(
        DATA_DIR / "csi300_index_daily.csv",
        parse_dates=["date"],
    )
    csi300 = csi300.set_index("date").sort_index()
    csi300["index_return"] = np.log(csi300["close"] / csi300["close"].shift(1))
    csi300_ret = csi300["index_return"].dropna()

    total_codes = stock_df["code"].nunique()
    print(f"Processing {total_codes} stocks ...")

    results = []
    for i, (code, group) in enumerate(stock_df.groupby("code")):
        if i % 500 == 0:
            print(f"  {i}/{total_codes}")

        group = group.set_index("date").sort_index()
        close = group["close"]

        # 日对数收益率
        daily_ret = np.log(close / close.shift(1)).dropna()
        if len(daily_ret) < 2:
            continue

        # ---- 季度聚合指标 ----
        q_log_ret = close.resample("QE").apply(_log_return)
        q_vol = daily_ret.resample("QE").std()
        q_mdd = close.resample("QE").apply(_max_drawdown)

        valid_quarters = q_log_ret.dropna().index

        # ---- 滚动Beta ----
        betas = _calc_quarterly_beta(daily_ret, csi300_ret, valid_quarters)

        # ---- 组装结果 ----
        for q_end in valid_quarters:
            results.append({
                "code": code,
                "quarter": f"{q_end.year}Q{q_end.quarter}",
                "log_return": q_log_ret[q_end],
                "volatility": q_vol[q_end],
                "max_drawdown": q_mdd[q_end],
                "beta": betas.get(q_end, np.nan),
            })

    print(f"Saving {len(results)} rows to market_quarterly.csv ...")
    result_df = pd.DataFrame(results)

    print("Loading TRACE_上市公司基本信息.csv ...")
    info = pd.read_csv(
        DATA_DIR / "TRACE_上市公司基本信息.csv", usecols=["ts_code", "name", "industry", "industry_csrc"]
    )
    info = info.rename(columns={"ts_code": "code"})
    result_df = result_df.merge(info, on="code", how="left")

    result_df.to_csv(DATA_DIR / "market_quarterly.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
