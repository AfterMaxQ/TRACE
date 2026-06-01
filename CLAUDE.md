# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRACE (**T**rade-linked **R**isk **A**ssessment and **C**ontagion **E**ngine) — a multi-source data fusion platform for enterprise credit risk assessment with supply chain contagion modeling, NLP sentiment analysis, and AI Agent interaction. Python 3.10+, SQLite backend, Streamlit frontend.

**Current phase**: F-01–F-10 全部完成。

## Project Structure

```
TRACE/
├── src/           # 数据处理与特征工程模块 (14 modules)
├── backend/       # Agent 后端 (agent.py)
├── frontend/      # Streamlit 仪表盘 (app.py + 5 pages)
├── model/         # 训练好的模型文件 (pkl, pt)
├── data/          # 所有 CSV 数据文件
└── docs/          # 文档与日志
```

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
python -u src/overseas_revenue.py    # Enterprise-level overseas revenue ratio (akshare stock_zygc_em)

# Feature engineering pipeline (run in order; each ~10-30s except sentiment)
python src/news_cleaner.py           # F-03 prep: 448K → 200K cleaned news titles
python src/industry_mapping.py       # F-04 prep: IO sector ↔ Shenwan industry crosswalk
python src/tariff_processor.py       # F-04 prep: USITC HTS8 → HS2 annual tariff aggregation
python src/trade_features.py         # F-04: 16 trade/IO/tariff/overseas industry-quarter features
python src/sentiment.py              # F-03: DeepSeek LLM 5-dimension classification + quarterly aggregation
python src/knowledge_graph.py         # F-05: 42x42 IO matrix -> 111 Shenwan graph + 5 topology features
python src/modeling.py               # F-06: XGBoost training + SHAP + scorecard (ROC-AUC ~0.91)
python src/features.py               # F-02: Financial ratios + ST labels + merge all -> base_feature (91 cols, ~19s)

# Financial fetcher supports optional flags:
python src/financial_fetcher.py --limit 100 --workers 8
```

All scripts output to `data/`. Always use `python -u` for real-time progress output; Python buffering silences it otherwise.

## Architecture

### Data Acquisition & Processing Modules (src/)

| Module | Source API | Output (data/) | Time Coverage |
|--------|-----------|----------------|---------------|
| `data_fetcher.py` | yfinance + akshare | `stock_daily.csv`, `csi300_daily.csv`, `bond_yields.csv` (1Y/10Y) | 2021-07+ |
| `financial_fetcher.py` | akshare (EastMoney) | `balance_sheet.csv`, `income_statement.csv`, `cash_flow.csv` | 2018+ |
| `macro_fetcher.py` | akshare + EastMoney API | `gdp.csv`, `cpi.csv`, `pmi.csv`, `m2.csv`, `shero.csv`, `shibor.csv`, `macro_quarterly.csv` | 2020+ |
| `market_quarterly.py` | Derived from `stock_daily.csv` + `csi300_daily.csv` | `market_quarterly.csv` | 2021Q3+ |
| `news_fetcher.py` | akshare + Sina/CLS APIs | `news_raw.csv` (includes research reports from EastMoney) | 2020+ |
| `trade_fetcher.py` | akshare + direct APIs | `bdi.csv`, `usdcny_daily.csv`, `trade_monthly.csv`, `trade_quarterly.csv`, `freight_weekly.csv`, `freight_quarterly.csv`, `us_tariffs.csv`, `scfi.csv` | 2020+ |
| `supply_chain.py` | Tushare Pro | `shareholders.csv`, `pledge_stat.csv`, validates `supply_chain_edges.csv` | latest report period |
| `overseas_revenue.py` | akshare `stock_zygc_em` | `overseas_revenue.csv` (67K rows, 5,457 stocks) | 2018-2025 |
| `news_cleaner.py` | Derived from `news_raw.csv` | `news_clean.csv` (200K rows, 6-step cleaning) | 2017-2026 |
| `industry_mapping.py` | Hardcoded IO→CSRC + HS2→IO mappings + `company_info.csv` | `industry_mapping.csv` (437 rows, 42 IO × 110 Shenwan) | static |
| `tariff_processor.py` | USITC HTS8 zip files | `hs2_tariff_annual.csv` (96 HS2 × 11 years) | 2016-2026 |
| `trade_features.py` | CEIC HS2 trade + IO table + tariffs + overseas revenue | `trade_features.csv` (16 features, 3,756 rows) | 2018Q1-2026Q2 |
| `sentiment.py` | DeepSeek v4 Flash via OpenAI SDK | `sentiment_raw.csv` (200K labels) + `sentiment_features.csv` (9 quarterly aggregates) | 2017-2026 |
| `knowledge_graph.py` | NBS 2023 IO table (42×42 matrix) | `io_adjacency.csv` (11.9K edges), `io_nodes.csv` (110 nodes), `graph_features.csv` (5 topology features) | 2023 |
| `modeling.py` | `base_feature.csv` (91 cols) | `model_xgb.pkl`, `feature_importance.csv`, `predictions.csv`, `shap_summary.png` | — |
| `features.py` | Derived from all above | `base_feature.csv` (91 columns, 98K rows, 5,322 stocks) | 2021Q3-2026Q1 |

### Module Dependency Chain

```
data_fetcher.py ──→ stock_daily.csv ──→ market_quarterly.py ──→ market_quarterly.csv
                  └─ csi300_daily.csv ──┘                          │
                                                                   │
financial_fetcher.py ──→ balance_sheet.csv, income_statement.csv,  │
                         cash_flow.csv                              │
                                                                   │
macro_fetcher.py ──→ macro_quarterly.csv                           │
                                                                   │
news_fetcher.py ──→ news_raw.csv ──→ news_cleaner.py ──→ news_clean.csv ──→ sentiment.py ──→ sentiment_features.csv
                                                                                                │
trade_fetcher.py ──→ bdi.csv, usdcny_daily.csv, trade_*.csv                                      │
                                                                                                 │
overseas_revenue.py ──→ overseas_revenue.csv ──┐                                                 │
industry_mapping.py ──→ industry_mapping.csv ──┤                                                 │
tariff_processor.py ──→ hs2_tariff_annual.csv ──┤                                                 │
CEIC data (manual) ───→ ceic_china_hs2_trade ──┼──→ trade_features.py ──→ trade_features.csv ──┤
                       ceic_io_noncompetitive ──┘                                                 │
                                                                                                 │
supply_chain.py ──→ shareholders.csv, pledge_stat.csv                                            │
                                                                                                 │
IO table (NBS) ──────→ knowledge_graph.py ──→ graph_features.csv ──────────────────────────────┤
                                                io_adjacency.csv                                 │
                                                                                                 │
st_labels.csv ────────────────────────────────────────────────────────────────────────────────────┼──→ features.py ──→ base_feature.csv
company_info.csv ─────────────────────────────────────────────────────────────────────────────────┘        (91 cols)
```

### Key Data Files (>10MB, gitignored)

| File | Size | Rows | Description |
|------|------|------|-------------|
| `stock_daily.csv` | 621MB | 6.5M | Full A-share daily OHLCV, 5500 stocks |
| `balance_sheet.csv` | 42MB | — | Quarterly balance sheets |
| `income_statement.csv` | 35MB | — | Quarterly income statements |
| `cash_flow.csv` | 30MB | — | Quarterly cash flow statements |
| `news_raw.csv` | 38MB | 364K | Raw news titles with date/code/source |
| `news_clean.csv` | — | 200K | Cleaned news (after 6-step pipeline) |
| `sentiment_raw.csv` | — | 200K | LLM 5-dimension classification labels |
| `market_quarterly.csv` | 12MB | 97K | Quarterly returns, vol, drawdown, beta |
| `base_feature.csv` | — | 98K | Final feature wide table (91 columns) |

### base_feature.csv Column Groups (91 total)

| Group | Count | Source Module |
|-------|-------|---------------|
| Key columns | 4 | code, quarter, name, industry |
| Liquidity | 3 | financial ratios |
| Leverage | 5 | financial ratios |
| Solvency | 1 | financial ratios |
| Profitability | 6 | financial ratios |
| Expense ratios | 4 | financial ratios |
| Cash flow | 5 | financial ratios |
| Asset structure | 5 | financial ratios |
| Working capital | 1 | financial ratios |
| Per-share | 2 | financial ratios |
| Growth (YoY) | 8 | financial ratios |
| Turnover | 4 | financial ratios |
| Market | 4 | market_quarterly |
| Macro | 8 | macro_quarterly |
| Trade/IO (F-04) | 16 | trade_features |
| Sentiment (F-03) | 9 | sentiment_features |
| Graph topology (F-05) | 5 | knowledge_graph |
| Target label | 1 | ST labels |

### Missing Value Imputation (features.py `_handle_missing`)

Three-tier for financial ratios: industry median → global median → 0. Macro columns: forward fill. Trade/sentiment columns: industry forward-fill → industry median → global median → 0. All ratio columns Winsorized at 1%/99%.

## Important Conventions

- **Proxy**: `data_fetcher.py` sets `HTTP_PROXY=http://127.0.0.1:7897`. Other modules don't need it (akshare uses different transport).
- **Encoding**: All CSV output uses `utf-8-sig` (BOM for Excel compatibility on Windows).
- **Date format**: Always `YYYY-MM-DD` in output CSVs; raw APIs may differ (parse to unified format).
- **Floating precision**: Macro/quarterly numeric columns rounded to 2 decimal places via `.round(2)`. `base_feature.csv` uses `.round(6)`.
- **Quarter labels**: `2020Q1` format throughout — generated by `f"{year}Q{(month-1)//3+1}"`.
- **Quarterly completeness**: Monthly-sourced indicators (CPI, PMI, M2) require ≥2 months per quarter via `_quarterly_agg()`. Daily-sourced (Shibor) have no minimum.
- **GDP convention**: Q1 = single-quarter real YoY; Q2-Q4 = cumulative real YoY (China standard reporting, from akshare `国内生产总值-同比增长` column).
- **Stock code normalization**: `data_fetcher.py` converts yfinance's `.SS` suffix to Tushare-standard `.SH` after fetching (line ~148). All downstream modules expect the `000001.SZ`/`000001.SH` format.
- **Bond yields**: `data_fetcher.py` uses `ak.bond_china_yield` (1Y/10Y) with yearly chunking, with `bond_zh_us_rate` (2Y/10Y) as fallback.
- **Tushare Pro token**: Set via `TS_TOKEN` environment variable. `supply_chain.py` requires it; `features.py` uses it as primary ST-label source with akshare SZSE as fallback.
- **DeepSeek API key**: `sentiment.py` has the key hardcoded (line 32). Move to environment variable before sharing or committing.
- **Reference data**: `supply_chain.py` depends on `data/company_info.csv` (full A-share code list with `ts_code` column) and `data/supply_chain_edges.csv` (manually curated supplier/customer edges). Both must exist before running the module.
- **CEIC data**: `ceic_china_hs2_trade_2018_2026.csv` and `ceic_io_noncompetitive.csv` are manually exported from CEIC database — not fetched programmatically.

### Industry Mapping (F-04 Two-Hop Bridge)

IO sectors (GB/T 4754, 42 departments) → CSRC codes → Shenwan industries:
1. **IO → CSRC**: 42 hardcoded entries in `industry_mapping.py` (`IO_TO_CSRC` dict)
2. **CSRC → Shenwan**: Data-driven cross-tabulation from `company_info.csv`
3. **HS2 → IO**: 89 hardcoded product-category entries (`HS2_TO_IO` dict)

One Shenwan industry ("陶瓷") has no CSRC bridge — handled via `MANUAL_SHENWAN_IO` supplement. IO→Shenwan expansion produces duplicates (one Shenwan maps from multiple IO sectors); `trade_features.py` deduplicates via groupby aggregation.

### News Cleaning Pipeline (news_cleaner.py)

448K → 200K (-55.5%) via 6 steps: (1) source filter (drop cls + sina), (2) .BJ filter (Beijing Stock Exchange), (3) bracket suffix removal (research report source tags), (4) dedup per (date, code) keeping longest title, (5) quality tags (title_len, is_short).

### ST Label Construction (features.py)

Target = next quarter ST/*ST status (time-series, not snapshot). Data source priority:

1. `data/st_labels.csv` — baostock `isST` daily field (SH, no rate limit) + akshare SZSE name-change intervals
2. `data/namechange_history.csv` — Tushare `namechange` (full market, 1/hr rate limit) or akshare SZSE fallback
3. `company_info.csv` `is_st` snapshot (last resort)

ST labels must be regenerated (baostock query) if the quarter range changes.

## Known Data Limitations

- `shero.csv` (社融) only has data through 2025Q4 — akshare source hasn't published 2026 data yet.
- `sentiment_features.csv` ends at 2025Q4 despite `news_clean.csv` having 2026 data — needs re-aggregation.
- `overseas_revenue.csv` ends at 2025-12-31 — depends on annual report disclosure cycle.
- `scfi.csv` is a single weekly snapshot (18 lines), not a historical time series. `freight_weekly.csv` provides the HRCI proxy time series.
- `stock_news_em` returns only ~10 recent items per stock; historical depth comes from `stock_notice_report` (CNINFO corporate announcements, monthly sampled since 2020).
- `stock_info_global_cls` (CLS telegraph) returns only 20 recent items, market-level only (no per-stock code linkage).
- GDP Q2-Q4 values are cumulative growth rates, not single-quarter. This is the standard Chinese statistical reporting convention.
- Tushare Pro `top10_holders` is rate-limited to ~1 request/minute. `supply_chain.py` sleeps 62s between calls and only processes supply-chain-linked stocks.
- Tushare Pro `pledge_stat` has stricter rate limits (~1/hour). `supply_chain.py` only fetches 5 focus stocks.
- All modules start from 2021Q3 (not Q1) — `stock_daily.csv` starts 2021-07-01, financial statements start 2021-09-30 (Q3 reports).

## Behavioral Guidelines

1. **Think before coding** — State assumptions, surface tradeoffs, ask when unclear.
2. **Simplicity first** — Minimum code, no speculative features, no premature abstractions.
3. **Surgical changes** — Touch only what's needed, match existing style, don't "improve" adjacent code.
4. **Goal-driven execution** — Define verifiable success criteria, loop until verified.
5. **Don't guess APIs** — Test small samples before full runs. Data acquisition scripts take hours; a broken full run wastes time.
6. **Development log** — After each task, append changed files, key decisions, and output summaries to `docs/开发日志.md` for traceability. Include date, module name, what changed, and why.
