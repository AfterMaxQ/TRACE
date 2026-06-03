# TRACE Data Catalog

> 所有 CSV 使用 `utf-8-sig` 编码 (BOM, Excel 兼容)，`YYYY-MM-DD` 日期格式，股票代码 `000001.SZ` / `600000.SH` 格式。

## 生成顺序 & 依赖图

```
[参考数据]  company_info.csv, supply_chain_edges.csv  ← 手工维护
                │
    ┌───────────┼───────────┬──────────────────┐
    ▼           ▼           ▼                  ▼
[F-01 采集]  data_fetcher  financial_fetcher  macro_fetcher  trade_fetcher  news_fetcher
    │           │           │                  │              │
    ▼           ▼           ▼                  ▼              ▼
 stock_daily  balance_sheet  gdp,cpi,pmi,    bdi,usdcny,   news_raw
 csi300_daily income_statement m2,shero,shibor trade_*,freight_*,
 bond_yields cash_flow     macro_quarterly   us_tariffs,scfi
    │                       │
    ▼                       │
market_quarterly ──────────┘
    │           │           │
    ▼           ▼           ▼
[供应链]  supply_chain ──→ shareholders, pledge_stat
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
[F-02 特征]  features ──→ st_labels, namechange_history, base_feature
```

## 文件清单

### 参考数据 (手工维护)

| 文件 | 行数 | 描述 | 消费者 |
|------|------|------|--------|
| `company_info.csv` | ~5,500 | A股上市公司基本信息 (ts_code, name, industry, is_st, list_date ...) | data_fetcher, financial_fetcher, market_quarterly, news_fetcher, supply_chain, features |
| `supply_chain_edges.csv` | ~42 | 20家重点企业的前5大客户/供应商关系 (source_code, target_name, relation_type, amount_pct, source) | supply_chain |

### 财务数据 (F-01, financial_fetcher.py)

| 文件 | 大小 | 行数 | 描述 | 消费者 |
|------|------|------|------|--------|
| `balance_sheet.csv` | 42MB | ~103K | 资产负债表 (TOTAL_ASSETS, TOTAL_LIABILITIES, TOTAL_EQUITY 等 41 列) | features |
| `income_statement.csv` | 35MB | ~103K | 利润表 (OPERATE_INCOME, NETPROFIT, PARENT_NETPROFIT 等 33 列) | features |
| `cash_flow.csv` | 30MB | ~103K | 现金流量表 (NETCASH_OPERATE, NETCASH_INVEST, NETCASH_FINANCE 等 30 列) | features |

三张表均按 code + REPORT_DATE 组织，覆盖全部 A 股 2018+。

### 市场数据 (F-01, data_fetcher.py + market_quarterly.py)

| 文件 | 大小 | 行数 | 描述 | 消费者 |
|------|------|------|------|--------|
| `stock_daily.csv` | 621MB | 6.5M | 全 A 股前复权日线 OHLCV (date, open, high, low, close, volume, code) | market_quarterly |
| `csi300_daily.csv` | — | — | 沪深 300 指数日线 (date, close) | market_quarterly |
| `bond_yields.csv` | — | ~1.2K | 中债国债收益率日频 1Y/10Y (date, yield_1y, yield_10y) | — |
| `market_quarterly.csv` | 12MB | 97K | 季度市场指标 (log_return, volatility, max_drawdown, beta) + name/industry | features |

### 宏观数据 (F-01, macro_fetcher.py)

| 文件 | 行数 | 描述 | 频率 | 消费者 |
|------|------|------|------|--------|
| `gdp.csv` | ~25 | GDP 当季同比 (gdp_yoy) | 季度 | — |
| `cpi.csv` | ~72 | CPI 当月同比 (cpi_yoy) | 月度 | — |
| `pmi.csv` | ~72 | 制造业 PMI (pmi) | 月度 | — |
| `m2.csv` | ~72 | M2 同比增速 (m2_yoy) | 月度 | — |
| `shero.csv` | ~60 | 社会融资规模增量 (shero) | 月度 | — |
| `shibor.csv` | ~1.5K | 上海银行间拆放利率日频 (shibor_on, shibor_1m, shibor_1y) | 日频 | — |
| `macro_quarterly.csv` | 26 | 宏观季度合并表 (gdp_yoy, cpi_yoy, pmi, m2_yoy, shero, shibor_*) | 季度 | features |

月度指标聚合为季度时要求 ≥2 个月有数据；日频 Shibor 无最低要求。GDP Q2-Q4 为累计同比 (中国标准报告惯例)。

### 贸易数据 (F-01, trade_fetcher.py)

| 文件 | 行数 | 描述 | 频率 | 消费者 |
|------|------|------|------|--------|
| `bdi.csv` | ~72 | 波罗的海干散货指数 (bdi) | 季度均值 | — |
| `usdcny_daily.csv` | ~1.2K | 美元/人民币中间价 (usdcny) | 日频 | — |
| `trade_monthly.csv` | ~60 | 进出口月度数据 (export, import, balance) | 月度 | — |
| `trade_quarterly.csv` | ~20 | 进出口季度汇总 | 季度 | — |
| `us_tariffs.csv` | 13 | 美国对华关税事件记录 (date, tariff_rate, description) | 事件 | — |
| `freight_weekly.csv` | — | 中国出口集装箱运价指数 (CCFI) 周度 | 周度 | — |
| `freight_quarterly.csv` | — | CCFI 季度均值 | 季度 | — |
| `scfi.csv` | — | 上海出口集装箱运价指数 (SCFI) 最新 | 快照 | — |

### 舆情数据 (F-01, news_fetcher.py)

| 文件 | 大小 | 行数 | 描述 | 消费者 |
|------|------|------|------|--------|
| `news_raw.csv` | 38MB | 448K | 新闻原始数据 (date, code, title, source) — 含 cninfo/东方财富/新浪/CLS 四源 | F-03 NLP |

### 供应链数据 (F-01, supply_chain.py)

| 文件 | 行数 | 描述 | 消费者 |
|------|------|------|--------|
| `shareholders.csv` | 200 | 20 家重点企业前十大股东 (ts_code, holder_name, hold_ratio_pct, holder_type) | — |
| `pledge_stat.csv` | 20 | 20 家重点企业股权质押统计 (ts_code, pledge_count, pledge_ratio) | — |

### 特征工程数据 (F-02, features.py)

| 文件 | 行数 | 描述 | 来源 |
|------|------|------|------|
| `st_labels.csv` | 107K | 全市场股票 × 季度 ST 状态 (code, quarter, is_st) — 时间序列 | baostock (沪市) + akshare (深市) |
| `namechange_history.csv` | 1,437 | 深市 ST/*ST 历史变更记录 (ts_code, start_date, end_date, change_reason) | akshare stock_info_sz_change_name |
| `base_feature.csv` | 98K | **特征宽表** (61 列) — 财务比率 + 增长率 + 周转率 + 市场指标 + 宏观指标 + target | features.py |

### base_feature.csv 列说明

| 类别 | 列名 | 数量 |
|------|------|------|
| 标识 | code, quarter, name, industry | 4 |
| 流动性 | current_ratio, quick_ratio, cash_ratio | 3 |
| 杠杆 | debt_to_assets, debt_to_equity, equity_multiplier, long_term_debt_ratio, equity_to_assets | 5 |
| 偿付 | interest_coverage | 1 |
| 盈利 | roe, roa, gross_margin, operating_margin, net_margin, ebitda_margin | 6 |
| 费用 | sale_expense_ratio, admin_expense_ratio, research_expense_ratio, finance_expense_ratio | 4 |
| 现金流 | cf_to_revenue, cf_to_debt, cf_to_assets, cf_to_netprofit, free_cf | 5 |
| 资产结构 | fixed_asset_ratio, intangible_ratio, goodwill_ratio, inventory_ratio, receivables_ratio | 5 |
| 营运 | working_capital_to_assets | 1 |
| 每股 | eps, bps | 2 |
| 增长率 | revenue_yoy, netprofit_yoy, parent_netprofit_yoy, total_assets_yoy, total_equity_yoy, operate_profit_yoy, operating_cf_yoy, total_liabilities_yoy | 8 |
| 周转率 | inventory_turnover, receivable_turnover, asset_turnover, equity_turnover | 4 |
| 市场 | log_return, volatility, max_drawdown, beta | 4 |
| 宏观 | gdp_yoy, cpi_yoy, pmi, m2_yoy, shero, shibor_on, shibor_1m, shibor_1y | 8 |
| 标签 | target (下季度 ST=1, 否则 0) | 1 |
| **合计** | | **61** |

### 生成命令速查

```bash
# F-01: 数据采集 (按依赖顺序)
python -u src/data_fetcher.py          # → stock_daily, csi300_daily, bond_yields
python -u src/financial_fetcher.py     # → balance_sheet, income_statement, cash_flow
python -u src/macro_fetcher.py         # → gdp, cpi, pmi, m2, shero, shibor, macro_quarterly
python -u src/market_quarterly.py      # → market_quarterly  (依赖 stock_daily + csi300_daily)
python -u src/news_fetcher.py          # → news_raw
python -u src/trade_fetcher.py         # → bdi, usdcny, trade_*, freight_*, us_tariffs, scfi
python -u src/supply_chain.py          # → shareholders, pledge_stat  (依赖 company_info + supply_chain_edges)

# F-02: 特征工程
python -u src/features.py              # → base_feature  (约 10s)
```
