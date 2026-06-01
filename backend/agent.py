"""
F-09 AI Agent 模块

DeepSeek function calling — 5 个风险分析工具, 自然语言交互。

工具:
  query_risk_score     — 公司综合评分 + 趋势 + 关键因素
  trace_supply_chain   — 行业上下游 + 企业供应链关系
  analyze_macro        — 宏观指标解读
  simulate_tariff      — 关税冲击模拟 + 传导
  generate_report      — 自动生成 Markdown 风险简报

用法:
  python src/agent.py                    # 交互模式
  python src/agent.py -r 000488.SZ         # 一键生成报告
  python src/agent.py -q "000725 风险?"   # 单次问答
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "model"
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"

SYSTEM_PROMPT = """你是 TRACE 金融风险分析助手, 专精于 A 股上市公司信用风险评估、供应链传导分析和宏观经济解读。

你的能力:
- 查询公司的综合风险评分、评级和历史趋势
- 分析行业供应链上下游传导风险
- 解读宏观经济指标对信用风险的影响
- 模拟关税变化对行业和公司的影响
- 生成结构化的风险分析报告

回复原则:
- 使用中文, 简洁专业
- 风险结果引用具体数字(评分、百分比、排名)
- 区分"公司自身问题"和"行业/宏观问题"
- 当用户只给股票代码时, 默认返回该公司的综合风险概况
- 可以用表格呈现对比数据"""


# ============================================================
# Data Loaders (cached on first use)
# ============================================================

_fusion = None
_feat_imp = None
_io_adj = None
_macro = None
_tariff = None
_info = None
_sc_edges = None
_ind_map = None


def _load_all():
    global _fusion, _feat_imp, _io_adj, _macro, _tariff, _info, _sc_edges, _ind_map
    if _fusion is None:
        _fusion = pd.read_csv(DATA_DIR / "fusion_scores.csv")
        if "quarter" in _fusion.columns:
            _fusion["quarter"] = _fusion["quarter"].astype(str)
    if _feat_imp is None:
        _feat_imp = pd.read_csv(DATA_DIR / "feature_importance.csv")
    if _io_adj is None:
        _io_adj = pd.read_csv(DATA_DIR / "io_adjacency.csv")
    if _macro is None:
        _macro = pd.read_csv(DATA_DIR / "macro_quarterly.csv")
        if "quarter_label" in _macro.columns:
            _macro = _macro.rename(columns={"quarter_label": "quarter"})
    if _tariff is None:
        _tariff = pd.read_csv(DATA_DIR / "hs2_tariff_annual.csv")
    if _info is None:
        _info = pd.read_csv(DATA_DIR / "company_info.csv", dtype={"ts_code": str})
    if _sc_edges is None:
        sc_path = DATA_DIR / "supply_chain_edges.csv"
        if sc_path.exists():
            _sc_edges = pd.read_csv(sc_path)
        else:
            _sc_edges = pd.DataFrame()
    if _ind_map is None:
        _ind_map = pd.read_csv(DATA_DIR / "industry_mapping.csv")


def _normalize_code(code: str) -> str:
    code = str(code).strip().upper().replace(".SS", ".SH")
    if len(code) == 6:
        if code.startswith(("6", "9")): return f"{code}.SH"
        elif code.startswith(("0", "3")): return f"{code}.SZ"
    return code


# ============================================================
# Tool 1: query_risk_score
# ============================================================

def query_risk_score(code: str) -> str:
    """查询公司的综合风险评分、评级、趋势和关键因素。"""
    _load_all()
    code = _normalize_code(code)

    # Find company info
    company = _info[_info["ts_code"] == code]
    if company.empty:
        # Try short code match
        short = code.split(".")[0]
        company = _info[_info["ts_code"].str.startswith(short)]
    if company.empty:
        return json.dumps({"error": f"未找到股票代码 {code}"}, ensure_ascii=False)

    name = company.iloc[0].get("name", code)
    industry = company.iloc[0].get("industry", "未知")

    # Latest quarter data
    company_data = _fusion[_fusion["code"] == code]
    if company_data.empty:
        # Try short code
        company_data = _fusion[_fusion["code"].str.startswith(code.split(".")[0])]

    if company_data.empty:
        return json.dumps({"error": f"未找到 {code} 的评分数据", "code": code, "name": name}, ensure_ascii=False)

    latest = company_data.sort_values("quarter").iloc[-1]
    quarter = latest["quarter"]

    # Trend (last 3 quarters)
    trend_data = company_data.sort_values("quarter").tail(4)
    trend = []
    for _, r in trend_data.iterrows():
        trend.append({"quarter": str(r["quarter"]), "score": int(r.get("fused_score", r.get("prob", 0)*1000))})

    # Score from fused_score (0-1000) or derive from prob (0-1)
    fs = latest.get("fused_score", 0)
    if pd.notna(fs) and fs > 1:
        score = int(fs)
    else:
        score = int((1 - latest.get("prob", 0.5)) * 1000)
    prob = float(latest["prob"]) if "prob" in latest.index else 0.5

    # Rating
    if score >= 700: rating, label = "AAA", "极低风险"
    elif score >= 650: rating, label = "AA", "低风险"
    elif score >= 600: rating, label = "A", "中低风险"
    elif score >= 550: rating, label = "BBB", "中等风险"
    elif score >= 500: rating, label = "BB", "关注级"
    elif score >= 450: rating, label = "B", "预警级"
    else: rating, label = "C", "高风险"

    # Percentile
    latest_q = _fusion[_fusion["quarter"] == quarter]
    if len(latest_q) > 0:
        if "fused_score" in latest_q.columns:
            pct = (latest_q["fused_score"] < score).mean()
        else:
            pct = (latest_q["prob"] < prob).mean()
    else:
        pct = 0.5

    # Industry avg
    ind_scores = latest_q[latest_q["code"].isin(_info[_info["industry"]==industry]["ts_code"])]
    if len(ind_scores) > 0 and "fused_score" in ind_scores.columns:
        ind_avg = int(ind_scores["fused_score"].mean())
    else:
        ind_avg = None

    # Top risk factors from feature importance
    top_features = _feat_imp.head(5)["feature"].tolist() if len(_feat_imp) > 0 else ["bps","eps","net_margin","debt_to_assets","compliance_ratio"]

    return json.dumps({
        "code": code, "name": str(name), "industry": str(industry),
        "latest_quarter": str(quarter),
        "score": score, "rating": rating, "rating_label": label,
        "default_prob": round(prob, 4),
        "percentile": f"前 {pct*100:.0f}% ({'风险较高' if pct > 0.5 else '风险较低'})",
        "industry_avg_score": ind_avg,
        "top_risk_factors": [str(f) for f in top_features],
        "score_trend": trend,
    }, ensure_ascii=False, indent=2)


# ============================================================
# Tool 2: trace_supply_chain
# ============================================================

def trace_supply_chain(code_or_industry: str) -> str:
    """分析公司或行业的供应链上下游传导风险。"""
    _load_all()

    # Determine if input is a code or industry name
    industry_name = None
    code = None
    name = None

    normalized = _normalize_code(code_or_industry)
    company = _info[_info["ts_code"] == normalized]
    if company.empty:
        short = normalized.split(".")[0] if "." in normalized else normalized
        company = _info[_info["ts_code"].str.startswith(short)]

    if not company.empty:
        code = company.iloc[0]["ts_code"]
        name = company.iloc[0].get("name", code)
        industry_name = company.iloc[0].get("industry", None)
    else:
        # Treat as industry name
        industry_name = code_or_industry

    if not industry_name:
        return json.dumps({"error": f"无法识别: {code_or_industry}"}, ensure_ascii=False)

    # IO upstream: target = this industry -> source = upstream
    upstream = _io_adj[_io_adj["target"] == industry_name].nlargest(8, "weight")

    # IO downstream: source = this industry -> target = downstream
    downstream = _io_adj[_io_adj["source"] == industry_name].nlargest(8, "weight")

    # Get risk scores for upstream/downstream industries
    latest_q = str(_fusion["quarter"].max())

    def _get_ind_risk(ind):
        stocks = _info[_info["industry"] == ind]["ts_code"].tolist()
        if not stocks: return None
        ind_data = _fusion[(_fusion["code"].isin(stocks)) & (_fusion["quarter"] == latest_q)]
        if ind_data.empty: return None
        return int(ind_data["fused_score"].mean()) if "fused_score" in ind_data.columns else int(ind_data["prob"].mean()*1000)

    up_list = []
    for _, e in upstream.iterrows():
        risk = _get_ind_risk(e["source"])
        exposure = "高" if e["weight"] > 0.05 else ("中" if e["weight"] > 0.01 else "低")
        up_list.append({
            "industry": e["source"], "io_coefficient": round(e["weight"], 4),
            "risk_score": risk, "exposure": exposure
        })

    down_list = []
    for _, e in downstream.iterrows():
        risk = _get_ind_risk(e["target"])
        exposure = "高" if e["weight"] > 0.05 else ("中" if e["weight"] > 0.01 else "低")
        down_list.append({
            "industry": e["target"], "io_coefficient": round(e["weight"], 4),
            "risk_score": risk, "exposure": exposure
        })

    # Known relationships
    known = []
    if code and len(_sc_edges) > 0:
        sc_code = _sc_edges[_sc_edges["source_code"] == code]
        for _, r in sc_code.iterrows():
            known.append({"type": str(r.get("relation_type","?")),
                          "name": str(r.get("target_name","?")),
                          "amount_pct": str(r.get("amount_pct","?"))})

    # Conduction risk summary
    high_risk_upstream = [u for u in up_list if u.get("risk_score") and u["risk_score"] < 550]
    conduction_risk = "高" if len(high_risk_upstream) >= 2 else ("中" if len(high_risk_upstream) >= 1 else "低")

    return json.dumps({
        "entity": {"code": code, "name": str(name) if name else None, "industry": industry_name},
        "upstream_industries": up_list,
        "downstream_industries": down_list,
        "known_relationships": known,
        "conduction_risk": conduction_risk,
    }, ensure_ascii=False, indent=2)


# ============================================================
# Tool 3: analyze_macro
# ============================================================

def analyze_macro(quarter: str = None) -> str:
    """解读宏观经济指标对信用风险的影响。"""
    _load_all()

    if quarter is None:
        quarter = str(_macro["quarter"].max())

    q_data = _macro[_macro["quarter"] == quarter]
    if q_data.empty:
        quarters = sorted(_macro["quarter"].unique())
        q_data = _macro[_macro["quarter"] == quarters[-1]]
        quarter = quarters[-1]

    row = q_data.iloc[0]

    # Previous quarter for trend
    prev_q_idx = sorted(_macro["quarter"].unique()).index(quarter)
    prev_data = None
    if prev_q_idx > 0:
        prev_q = sorted(_macro["quarter"].unique())[prev_q_idx - 1]
        prev_data = _macro[_macro["quarter"] == prev_q].iloc[0] if len(_macro[_macro["quarter"]==prev_q])>0 else None

    def _trend(curr, prev, name):
        if prev is None or pd.isna(curr) or pd.isna(prev): return "无数据"
        if curr > prev * 1.02: return "上升"
        if curr < prev * 0.98: return "下降"
        return "持平"

    def _signal(indicator, value):
        rules = {
            "gdp_yoy": lambda v: "正面" if v > 5 else ("中性" if v > 3 else "负面"),
            "pmi": lambda v: "偏正面" if v > 50.5 else ("中性" if v > 49.5 else "偏负面"),
            "cpi_yoy": lambda v: "偏正面" if 1 <= v <= 3 else ("需关注(通缩)" if v < 0.5 else "需关注(通胀)"),
            "m2_yoy": lambda v: "偏正面" if v > 8 else ("中性" if v > 6 else "偏负面"),
            "shibor_on": lambda v: "偏正面" if v < 2 else ("中性" if v < 4 else "偏负面"),
            "shero": lambda v: "偏正面" if v > 3 else ("中性" if v > 1 else "偏负面"),
        }
        fn = rules.get(indicator, lambda v: "中性")
        return fn(value) if not pd.isna(value) else "无数据"

    indicators = {}
    for col in ["gdp_yoy", "pmi", "cpi_yoy", "m2_yoy", "shibor_on", "shero"]:
        if col not in row.index:
            continue
        val = row[col]
        prev_val = prev_data[col] if prev_data is not None and col in prev_data.index else None
        indicators[col] = {
            "value": round(float(val), 2) if not pd.isna(val) else None,
            "trend": _trend(val, prev_val, col),
            "signal": _signal(col, val),
        }

    # Summary
    signals = [v["signal"] for v in indicators.values()]
    positive = sum(1 for s in signals if "正面" in str(s))
    negative = sum(1 for s in signals if "负面" in str(s) or "通缩" in str(s) or "通胀" in str(s))
    if positive >= 4:
        summary = f"{quarter} 经济形势良好, 信用环境偏正面, 有利于企业偿债。"
    elif positive >= 2:
        summary = f"{quarter} 经济温和运行, 信用环境中性偏正面。"
    else:
        summary = f"{quarter} 经济面临压力, 信用环境需关注。"

    return json.dumps({
        "quarter": quarter,
        "indicators": indicators,
        "summary": summary,
    }, ensure_ascii=False, indent=2)


# ============================================================
# Tool 4: simulate_tariff
# ============================================================

def simulate_tariff(company_or_industry: str, rate_change_pct: float) -> str:
    """模拟关税变化对行业和下游的影响。"""
    _load_all()

    # Resolve to industry
    code = None
    industry_name = None
    normalized = _normalize_code(company_or_industry)
    company = _info[_info["ts_code"] == normalized]
    if company.empty:
        short = normalized.split(".")[0] if "." in normalized else normalized
        company = _info[_info["ts_code"].str.startswith(short)]
    if not company.empty:
        code = company.iloc[0]["ts_code"]
        industry_name = company.iloc[0].get("industry", company_or_industry)
    else:
        industry_name = company_or_industry

    # Find IO sector for this industry
    io_row = _ind_map[_ind_map["shenwan_industry"] == industry_name]
    if io_row.empty:
        # Fuzzy match
        io_row = _ind_map[_ind_map["shenwan_industry"].str.contains(industry_name[:3], na=False)]
    if io_row.empty:
        return json.dumps({"error": f"未找到行业映射: {industry_name}"}, ensure_ascii=False)

    io_sector = io_row.iloc[0]["io_sector"]
    hs2_str = io_row.iloc[0].get("hs2_codes", "")

    # Import dependency from fusion data
    ind_stocks = _info[_info["industry"] == industry_name]["ts_code"].tolist()
    latest_q = str(_fusion["quarter"].max())
    ind_data = _fusion[(_fusion["code"].isin(ind_stocks)) & (_fusion["quarter"] == latest_q)]

    # Use base_feature for import_dependency
    bf = pd.read_csv(DATA_DIR / "base_feature.csv", low_memory=False)
    ind_bf = bf[(bf["industry"] == industry_name) & (bf["quarter"] == latest_q)]
    import_dep = ind_bf["import_dependency"].mean() if len(ind_bf) > 0 else 0.1

    # Current tariff rate (latest year)
    tariff_latest = _tariff[_tariff["year"] == _tariff["year"].max()]
    if isinstance(hs2_str, str) and hs2_str:
        hs2_list = [int(h) for h in hs2_str.split("|") if h.strip()]
        tariff_hs2 = tariff_latest[tariff_latest["hs2"].isin(hs2_list)]
        current_tariff = tariff_hs2["avg_tariff"].mean() if len(tariff_hs2) > 0 else 0.035
    else:
        current_tariff = 0.035

    new_tariff = current_tariff + rate_change_pct
    cost_impact = import_dep * rate_change_pct * 100
    severity = "轻微" if cost_impact < 1 else ("中等" if cost_impact < 3 else ("显著" if cost_impact < 6 else "严重"))

    # Downstream impact
    downstream = _io_adj[_io_adj["source"] == industry_name].nlargest(5, "weight")
    down_affected = []
    for _, e in downstream.iterrows():
        pass_through = e["weight"] * cost_impact / 100
        impact = "显著" if pass_through > 1 else ("中等" if pass_through > 0.3 else "轻微")
        down_affected.append({
            "industry": e["target"],
            "io_coefficient": round(e["weight"], 4),
            "cost_pass_through_pct": round(pass_through, 2),
            "impact": impact,
        })

    return json.dumps({
        "scenario": f"美国对{industry_name}相关商品加征{rate_change_pct*100:.0f}%关税",
        "direct_impact": {
            "industry": industry_name,
            "us_tariff_current": round(current_tariff, 4),
            "us_tariff_new": round(new_tariff, 4),
            "import_dependency": round(import_dep, 4),
            "cost_impact_pct": round(cost_impact, 2),
            "severity": severity,
        },
        "downstream_affected": down_affected,
        "summary": f"{rate_change_pct*100:.0f}%关税对{industry_name}直接成本影响约{cost_impact:.1f}%, 严重程度: {severity}。"
    }, ensure_ascii=False, indent=2)


# ============================================================
# Tool 5: generate_report
# ============================================================

def generate_report(code: str) -> str:
    """生成公司的综合风险分析报告（Markdown格式）。"""
    _load_all()
    code = _normalize_code(code)

    # Gather data from other tools
    risk_json = query_risk_score(code)
    risk = json.loads(risk_json)
    if "error" in risk:
        return f"# 错误\n\n{risk['error']}"

    chain_json = trace_supply_chain(code)
    chain = json.loads(chain_json)

    macro_json = analyze_macro()
    macro = json.loads(macro_json)

    # Build report
    report = f"""# {risk.get('name', code)} ({code}) 风险分析报告

## 综合评分
- **风险评级: {risk['rating']} ({risk['rating_label']})**
- 评分: {risk['score']}/1000 | 行业排名: {risk['percentile']}
- 违约概率: {risk['default_prob']:.2%}
"""

    if risk.get("industry_avg_score"):
        report += f"- 行业平均分: {risk['industry_avg_score']}\n"

    report += f"\n### 评分趋势\n"
    for t in risk.get("score_trend", []):
        bar = "█" * min(int(t["score"]/20), 40)
        report += f"- {t['quarter']}: {t['score']} {bar}\n"

    report += f"\n## 关键风险因素\n"
    for i, f in enumerate(risk.get("top_risk_factors", [])[:5]):
        report += f"{i+1}. {f}\n"

    report += f"\n## 供应链分析\n"
    if chain.get("upstream_industries"):
        report += "### 上游依赖\n"
        for u in chain["upstream_industries"][:5]:
            risk_tag = f"(风险评分: {u['risk_score']})" if u.get("risk_score") else ""
            report += f"- {u['industry']}: IO系数 {u['io_coefficient']:.4f}, 暴露度: {u['exposure']} {risk_tag}\n"

    if chain.get("downstream_industries"):
        report += "\n### 下游客户行业\n"
        for d in chain["downstream_industries"][:5]:
            report += f"- {d['industry']}: IO系数 {d['io_coefficient']:.4f}, 暴露度: {d['exposure']}\n"

    if chain.get("conduction_risk"):
        report += f"\n**传导风险: {chain['conduction_risk']}**\n"

    if chain.get("known_relationships"):
        report += "\n### 已知企业关系\n"
        for kr in chain["known_relationships"]:
            report += f"- {kr['type']}: {kr['name']} ({kr['amount_pct']})\n"

    report += f"\n## 宏观环境 ({macro['quarter']})\n"
    for ind, detail in macro.get("indicators", {}).items():
        v = detail.get("value", "N/A")
        s = detail.get("signal", "")
        report += f"- {ind}: {v} ({s})\n"
    report += f"\n{macro.get('summary', '')}\n"

    report += f"\n## 风险提示\n"
    if risk["score"] < 500:
        report += "!! 高风险预警 - 公司面临严重的信用风险, 建议密切关注。\n"
    elif risk["score"] < 600:
        report += "!! 需关注 - 公司风险高于行业平均, 建议定期跟踪关键指标变化。\n"
    else:
        report += "OK 风险可控 - 综合评分在安全区间。\n"

    return report


# ============================================================
# Agent loop with DeepSeek function calling
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_risk_score",
            "description": "查询A股上市公司的综合风险评分、评级、历史趋势和关键风险因素。输入6位代码或完整代码(如000488或000488.SZ)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "股票代码, 如000488或000488.SZ"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trace_supply_chain",
            "description": "分析公司或行业的供应链上下游传导风险, 识别上游依赖和下游客户行业。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code_or_industry": {"type": "string", "description": "股票代码或申万行业名称"}
                },
                "required": ["code_or_industry"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_macro",
            "description": "解读最新(或指定季度)的宏观经济指标对信用风险的影响, 包括GDP/CPI/PMI/M2/Shibor/社融。",
            "parameters": {
                "type": "object",
                "properties": {
                    "quarter": {"type": "string", "description": "季度标签如2026Q1, 默认最新"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_tariff",
            "description": "模拟美国关税变化对公司或行业的影响, 包括直接成本和下游传导。",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_or_industry": {"type": "string", "description": "股票代码或行业名"},
                    "rate_change_pct": {"type": "number", "description": "关税变化百分比, 如0.10表示加征10%"}
                },
                "required": ["company_or_industry", "rate_change_pct"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "生成公司的完整风险分析报告(Markdown格式), 包含评分/供应链/宏观/风险提示。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "股票代码"}
                },
                "required": ["code"]
            }
        }
    },
]

TOOL_MAP = {
    "query_risk_score": query_risk_score,
    "trace_supply_chain": trace_supply_chain,
    "analyze_macro": analyze_macro,
    "simulate_tariff": simulate_tariff,
    "generate_report": generate_report,
}


def _run_agent_loop():
    """交互式对话循环。"""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\n" + "=" * 60)
    print("  TRACE 风险分析 Agent")
    print("  输入 'help' 查看功能, 'quit' 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见。")
            break
        if user_input.lower() == "help":
            print("\n可用功能:")
            print("  [代码]           查询综合风险 (如 000488 或 000725.SZ)")
            print("  [代码] 供应链     分析上下游传导")
            print("  [行业] 关税 +10%  模拟关税冲击")
            print("  宏观             查看最新宏观解读")
            print("  报告 [代码]       生成完整风险报告")
            continue

        messages.append({"role": "user", "content": user_input})

        # Call DeepSeek with tools
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.3,
            max_tokens=1024,
        )

        msg = resp.choices[0].message

        # Handle tool calls
        if msg.tool_calls:
            messages.append(msg)

            for tc in msg.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                fn = TOOL_MAP.get(func_name)

                if fn:
                    try:
                        result = fn(**func_args)
                    except Exception as e:
                        result = json.dumps({"error": str(e)}, ensure_ascii=False)

                    # Truncate long results
                    if len(result) > 3000:
                        result = result[:3000] + "\n...(truncated)"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

            # Get final response
            resp2 = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            final_msg = resp2.choices[0].message.content
            messages.append({"role": "assistant", "content": final_msg})
            print(f"\nAgent: {final_msg}")

        else:
            messages.append({"role": "assistant", "content": msg.content})
            print(f"\nAgent: {msg.content}")


def main():
    parser = argparse.ArgumentParser(description="TRACE F-09 AI Agent")
    parser.add_argument("-r", "--report", type=str, help="生成指定代码的风险报告")
    parser.add_argument("-q", "--query", type=str, help="单次问答(用|分隔代码, 如 000488.SZ|风险?)")
    args = parser.parse_args()

    if args.report:
        code = _normalize_code(args.report)
        report = generate_report(code)
        print(report)
    elif args.query:
        parts = args.query.split("|")
        code = _normalize_code(parts[0])
        question = parts[1] if len(parts) > 1 else f"{code} 的风险怎么样?"
        print(f"查询: {question}")

        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tools=TOOLS,
            temperature=0.3,
            max_tokens=1024,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn = TOOL_MAP.get(tc.function.name)
                if fn:
                    args_dict = json.loads(tc.function.arguments)
                    # Auto-fill code if needed
                    if tc.function.name != "analyze_macro":
                        args_dict.setdefault("code", code)
                        args_dict.setdefault("code_or_industry", code)
                    result = fn(**args_dict)
                    if len(result) > 4000:
                        result = result[:4000]
                    print(f"\n{tc.function.name}:\n{result}")
        else:
            print(f"\n{msg.content}")
    else:
        _run_agent_loop()


if __name__ == "__main__":
    main()
