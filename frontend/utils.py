"""TRACE 前端工具函数 — 数据加载缓存 + 全局样式 + AI Agent"""
import os
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / ".trace_config.json"
BASE_URL = "https://api.deepseek.com"

# 优先从 .env 加载默认 API Key
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
MODEL = "deepseek-v4-flash"

SYSTEM_PROMPT = """你是 TRACE 金融风险分析助手, 专精于 A 股上市公司信用风险评估、供应链传导分析和宏观经济解读。

你能查询公司风险评分、分析供应链传导、解读宏观指标、模拟关税冲击、生成风险报告。

回复原则: 使用中文, 简洁专业, 引用具体数字。当用户只给股票代码时, 默认返回该公司的综合风险概况。"""

TOOLS = [
    {"type": "function", "function": {"name": "query_risk_score", "description": "查询A股上市公司的综合风险评分、评级、趋势和关键因素", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "股票代码"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "trace_supply_chain", "description": "分析公司或行业的供应链上下游传导风险", "parameters": {"type": "object", "properties": {"code_or_industry": {"type": "string", "description": "股票代码或行业名"}}, "required": ["code_or_industry"]}}},
    {"type": "function", "function": {"name": "analyze_macro", "description": "解读最新宏观经济指标对信用风险的影响", "parameters": {"type": "object", "properties": {"quarter": {"type": "string", "description": "季度如2026Q1, 默认最新"}}, "required": []}}},
    {"type": "function", "function": {"name": "simulate_tariff", "description": "模拟美国关税变化对公司或行业的影响", "parameters": {"type": "object", "properties": {"company_or_industry": {"type": "string"}, "rate_change_pct": {"type": "number", "description": "关税变化如0.10=加征10%"}}, "required": ["company_or_industry", "rate_change_pct"]}}},
    {"type": "function", "function": {"name": "generate_report", "description": "生成公司完整风险分析报告(Markdown)", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]

def inject_css():
    st.markdown("""
    <style>
    .main { background-color: #F8FAFC; }
    .stApp { background-color: #F8FAFC; }
    div[data-testid="stMetric"] {
        background: white; border-radius: 10px; padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .badge-green { background: #22C55E; color: white; padding: 4px 14px; border-radius: 14px; font-weight: 700; display: inline-block; }
    .badge-blue  { background: #3B82F6; color: white; padding: 4px 14px; border-radius: 14px; font-weight: 700; display: inline-block; }
    .badge-orange{ background: #F97316; color: white; padding: 4px 14px; border-radius: 14px; font-weight: 700; display: inline-block; }
    .badge-red   { background: #EF4444; color: white; padding: 4px 14px; border-radius: 14px; font-weight: 700; display: inline-block; }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("## TRACE 🛡️")
        st.caption("Trade-linked Risk Assessment & Contagion Engine")
        st.divider()
        try:
            bf = load_base_feature()
            mn, mx = bf["quarter"].min(), bf["quarter"].max()
            st.info(f"数据周期\n{mn} – {mx}")
        except:
            pass
        st.divider()
        st.caption("Python · XGBoost · DeepSeek · GAT")
        st.caption("Streamlit · Plotly · PyVis")

    # Agent chat (in sidebar, outside the with block so it renders below nav)
    render_agent_chat()

@st.cache_data
def _read_csv(filename):
    path = PROJECT_ROOT / "data" / filename
    if path.exists():
        return pd.read_csv(str(path), encoding="utf-8-sig", low_memory=False)
    st.warning(f"文件未找到: {filename}")
    return pd.DataFrame()

@st.cache_data
def load_macro():            return _read_csv("macro_quarterly.csv")
@st.cache_data
def load_base_feature():     return _read_csv("base_feature.csv")
@st.cache_data
def load_fusion_scores():    return _read_csv("fusion_scores.csv")
@st.cache_data
def load_predictions():      return _read_csv("predictions.csv")
@st.cache_data
def load_io_adjacency():     return _read_csv("io_adjacency.csv")
@st.cache_data
def load_graph_features():   return _read_csv("graph_features.csv")
@st.cache_data
def load_feature_importance(): return _read_csv("feature_importance.csv")
@st.cache_data
def load_company_info():     return _read_csv("company_info.csv")

def get_rating_badge(score):
    if score >= 700:  return "AAA", "badge-green"
    elif score >= 650: return "AA", "badge-green"
    elif score >= 600: return "A", "badge-blue"
    elif score >= 550: return "BBB", "badge-blue"
    elif score >= 500: return "BB", "badge-orange"
    elif score >= 450: return "B", "badge-orange"
    else:              return "C", "badge-red"

def find_company(query, info, scores):
    """支持代码/简称/名称模糊搜索"""
    q = str(query).strip().upper()
    match = info[info["ts_code"] == q]
    if not match.empty:
        return match.iloc[0]["ts_code"]
    if len(q) == 6 and q.isdigit():
        match = info[info["ts_code"].str.startswith(q)]
        if not match.empty:
            return match.iloc[0]["ts_code"]
    match = info[info["name"].str.contains(q, na=False)]
    if not match.empty:
        return match.iloc[0]["ts_code"]
    if q in scores["code"].values:
        return q
    return None


# ============================================================
# AI Agent (sidebar chat)
# ============================================================

def _get_agent_client():
    api_key = st.session_state.get("agent_api_key", "")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def _run_agent_query(user_msg: str) -> str:
    """单轮 Agent 调用: 用户消息 → DeepSeek function calling → 最终回复"""
    client = _get_agent_client()
    if client is None:
        return "请先在侧边栏设置 DeepSeek API Key"

    # Load backend tool functions lazily
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from backend.agent import query_risk_score, trace_supply_chain, analyze_macro, simulate_tariff, generate_report
    tool_map = {
        "query_risk_score": query_risk_score,
        "trace_supply_chain": trace_supply_chain,
        "analyze_macro": analyze_macro,
        "simulate_tariff": simulate_tariff,
        "generate_report": generate_report,
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS,
            temperature=0.3, max_tokens=800,
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls[:3]:  # max 3 tool calls
                fn = tool_map.get(tc.function.name)
                if fn:
                    try:
                        args = json.loads(tc.function.arguments)
                        result = fn(**args)
                        if len(result) > 2500:
                            result = result[:2500]
                    except Exception as e:
                        result = json.dumps({"error": str(e)}, ensure_ascii=False)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            resp2 = client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0.3, max_tokens=800,
            )
            return resp2.choices[0].message.content or "(无回复)"
        else:
            return msg.content or "(无回复)"

    except Exception as e:
        return f"Agent 调用失败: {str(e)[:200]}"


def render_agent_chat():
    """在侧边栏底部嵌入 AI Agent 配置 + 对话。"""
    with st.sidebar:
        st.divider()
        st.caption("🤖 AI 风险分析助手")

        # --- API Key 配置 (持久化) ---
        # 初始化 session state: 优先从文件加载
        if "agent_api_key" not in st.session_state:
            # 优先级: 本地配置 > .env > 空
            saved_key = ""
            if CONFIG_FILE.exists():
                try:
                    saved_key = json.loads(CONFIG_FILE.read_text(encoding="utf-8")).get("api_key", "")
                except:
                    pass
            if not saved_key:
                saved_key = os.getenv("DEEPSEEK_API_KEY", "")
            st.session_state.agent_api_key = saved_key

        # API Key 输入区 (折叠)
        with st.expander("⚙️ API 设置", expanded=not st.session_state.agent_api_key):
            new_key = st.text_input(
                "DeepSeek API Key",
                value=st.session_state.agent_api_key,
                type="password",
                placeholder="sk-...",
                help="密钥仅保存在本地 .trace_config.json",
            )
            if new_key != st.session_state.agent_api_key:
                st.session_state.agent_api_key = new_key
                # 持久化到文件
                CONFIG_FILE.write_text(
                    json.dumps({"api_key": new_key}, ensure_ascii=False),
                    encoding="utf-8",
                )
                st.rerun()

        if not st.session_state.agent_api_key:
            st.caption("👆 请先设置 API Key")
            return

        # --- 对话区 ---
        if "agent_messages" not in st.session_state:
            st.session_state.agent_messages = []

        # 历史消息
        for m in st.session_state.agent_messages[-4:]:
            role_label = "🧑" if m["role"] == "user" else "🤖"
            st.caption(f"{role_label} {m['content'][:200]}")

        # 输入 & 发送
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input(
                " ", placeholder="输入代码或问题...",
                key="agent_q", label_visibility="collapsed",
            )
        with col2:
            send = st.button("发送", key="agent_send", use_container_width=True)

        if send and user_input:
            st.session_state.agent_messages.append({"role": "user", "content": user_input})
            with st.spinner("分析中..."):
                reply = _run_agent_query(user_input)
            st.session_state.agent_messages.append({"role": "assistant", "content": reply})
            st.rerun()
