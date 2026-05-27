# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRACE (**T**rade-linked **R**isk **A**ssessment and **C**ontagion **E**ngine) — a multi-source data fusion platform for enterprise credit risk assessment with supply chain contagion modeling, NLP sentiment analysis, and AI Agent interaction. Python 3.10+, SQLite backend, Streamlit frontend.

The project is in the **feature engineering** phase (F-02). Data acquisition (F-01) is complete. Modeling, dashboard, and agent modules are planned but not yet implemented.

## Common Commands

```bash
# Data acquisition (run individually; some take 1-3 hours)
python -u src/data_fetcher.py        # Stock daily OHLCV + CSI 300 index + bond yields
python -u src/financial_fetcher.py    # Balance sheet, income statement, cash flow (all A-shares)
python -u src/macro_fetcher.py       # GDP, CPI, PMI, M2, SheRong, Shibor
python -u src/market_quarterly.py    # Quarterly returns, volatility, max drawdown, beta
python -u src/news_fetcher.py        # CLS telegraph, EastMoney/Sina stock news, CNINFO notices, research reports
python -u src/trade_fetcher.py       # BDI, USD/CNY, import/export, US tariffs, SCFI
python -u src/supply_chain.py        # Top10 shareholders, equity pledge stats, supply chain edge validation
python -u src/features.py           # F-02: Financial ratios, ST labels, feature wide table (~10s)

# Financial fetcher supports optional flags:
python src/financial_fetcher.py --limit 100 --workers 8
```

All scripts output to `data/`. Always use `python -u` for real-time progress output; Python buffering silences it otherwise.

## Architecture

### Data Acquisition Modules (src/)

| Module | Source API | Output (data/) | Time Coverage |
|--------|-----------|----------------|---------------|
| `data_fetcher.py` | yfinance + akshare | `stock_daily.csv`, `csi300_index_daily.csv`, `bond_yields.csv` (1Y/10Y) | 2021-07+ |
| `financial_fetcher.py` | akshare (EastMoney) | `balance_sheet.csv`, `income_statement.csv`, `cash_flow_statement.csv` | 2018+ |
| `macro_fetcher.py` | akshare + EastMoney API | `gdp.csv`, `cpi.csv`, `pmi.csv`, `m2.csv`, `shero.csv`, `shibor.csv`, `macro_quarterly.csv` | 2020+ |
| `market_quarterly.py` | Derived from `stock_daily.csv` + `csi300_index_daily.csv` | `market_quarterly.csv` | 2021Q3+ |
| `news_fetcher.py` | akshare + Sina/CLS APIs | `news_raw.csv` (includes research reports from EastMoney) | 2020+ |
| `trade_fetcher.py` | akshare + direct APIs | `bdi.csv`, `usdcny_daily.csv`, `trade_monthly.csv`, `trade_quarterly.csv`, `freight_weekly.csv`, `freight_quarterly.csv`, `us_tariffs.csv`, `scfi_latest.csv` | 2020+ |
| `supply_chain.py` | Tushare Pro | `share_holders.csv`, `pledge_stat.csv`, validates `supply_chain_edges.csv` | latest report period |
| `features.py` | Derived from financials + market + macro | `base_feature.csv` (42 features + target, 98K rows, 5,322 stocks) | 2021Q3–2026Q1 |

### Module Dependency Chain

```
data_fetcher.py ──→ stock_daily.csv ──→ market_quarterly.py ──→ market_quarterly.csv
                  └─ csi300_index_daily.csv ──┘

financial_fetcher.py ──→ balance_sheet.csv, income_statement.csv, cash_flow_statement.csv

macro_fetcher.py ──→ gdp.csv, cpi.csv, pmi.csv, m2.csv, shero.csv, shibor.csv
                  └─ macro_quarterly.csv (merged)

news_fetcher.py ──→ news_raw.csv

trade_fetcher.py ──→ bdi.csv, usdcny_daily.csv, trade_*.csv, freight_*.csv, us_tariffs.csv

supply_chain.py ──→ share_holders.csv, pledge_stat.csv (depends on TRACE_上市公司基本信息.csv + supply_chain_edges.csv)

features.py ──→ base_feature.csv (F-02, depends on the 3 financials + market_quarterly.csv + macro_quarterly.csv)
```

### Key Data Files (>10MB, gitignored)

| File | Size | Rows | Description |
|------|------|------|-------------|
| `stock_daily.csv` | 621MB | 6.5M | Full A-share daily OHLCV, 5500 stocks |
| `balance_sheet.csv` | 42MB | — | Quarterly balance sheets |
| `income_statement.csv` | 35MB | — | Quarterly income statements |
| `cash_flow_statement.csv` | 30MB | — | Quarterly cash flow statements |
| `news_raw.csv` | 38MB | 364K | News titles with date/code/source |
| `market_quarterly.csv` | 12MB | 97K | Quarterly returns, vol, drawdown, beta |

### Stock Code Convention

All modules use `000001.SZ` format (Tushare convention). yfinance requires `.SS` for Shanghai; the converter `_make_yf_tickers()` in `data_fetcher.py` handles this. Helper `_normalize_code()` exists in `news_fetcher.py` and `market_quarterly.py`.

## Important Conventions

- **Proxy**: `data_fetcher.py` sets `HTTP_PROXY=http://127.0.0.1:7897`. Other modules don't need it (akshare uses different transport).
- **Encoding**: All CSV output uses `utf-8-sig` (BOM for Excel compatibility on Windows).
- **Date format**: Always `YYYY-MM-DD` in output CSVs; raw APIs may differ (parse to unified format).
- **Floating precision**: All macro/quarterly numeric columns are rounded to 2 decimal places via `.round(2)`.
- **Quarter labels**: `2020Q1` format throughout — generated by `f"{year}Q{(month-1)//3+1}"`.
- **Quarterly completeness**: Monthly-sourced indicators (CPI, PMI, M2) require ≥2 months per quarter via `_quarterly_agg()`. Daily-sourced (Shibor) have no minimum.
- **GDP convention**: Q1 = single-quarter real YoY; Q2-Q4 = cumulative real YoY (China standard reporting, from akshare `国内生产总值-同比增长` column).
- **Stock code normalization**: `data_fetcher.py` converts yfinance's `.SS` suffix to Tushare-standard `.SH` after fetching (line ~148). All downstream modules expect the `000001.SZ`/`000001.SH` format.
- **Bond yields**: `data_fetcher.py` now uses `ak.bond_china_yield` (1Y/10Y) with yearly chunking, with `bond_zh_us_rate` (2Y/10Y) as fallback. Previously was 2Y/10Y exclusively.
- **Reference data**: `supply_chain.py` depends on `data/TRACE_上市公司基本信息.csv` (full A-share code list with `ts_code` column) and `data/supply_chain_edges.csv` (manually curated supplier/customer edges). Both must exist before running the module.

### Known Data Limitations

- `shero.csv` (社融) only has data through 2025Q4 — akshare source hasn't published 2026 data yet.
- `stock_news_em` returns only ~10 recent items per stock; historical depth comes from `stock_notice_report` (CNINFO corporate announcements, monthly sampled since 2020).
- `stock_info_global_cls` (CLS telegraph) returns only 20 recent items, market-level only (no per-stock code linkage).
- GDP Q2-Q4 values are cumulative growth rates, not single-quarter. This is the standard Chinese statistical reporting convention.
- Tushare Pro `top10_holders` is rate-limited to ~1 request/minute. `supply_chain.py` sleeps 62s between calls and only processes supply-chain-linked stocks (not full A-share universe).
- Tushare Pro `pledge_stat` has stricter rate limits (~1/hour). `supply_chain.py` only fetches 5 focus stocks for this endpoint.

## Behavioral Guidelines

These override default LLM behavior:

1. **Think before coding** — State assumptions, surface tradeoffs, ask when unclear.
2. **Simplicity first** — Minimum code, no speculative features, no premature abstractions.
3. **Surgical changes** — Touch only what's needed, match existing style, don't "improve" adjacent code.
4. **Goal-driven execution** — Define verifiable success criteria, loop until verified.
5. **Don't guess APIs** — Test small samples before full runs. Data acquisition scripts take hours; a broken full run wastes time.
6. **Development log** — After each task, append changed files, key decisions, and output summaries to `docs/开发日志.md` for traceability. Include date, module name, what changed, and why.
