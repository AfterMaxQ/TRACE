"""AI Agent 对话页面 — 实时思考链 + 工具调用反馈"""
import streamlit as st
import os, json, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_FILE = PROJECT_ROOT / ".trace_config.json"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="AI助手", page_icon="🤖", layout="wide")

st.markdown("""<style>
.think-line { color: #94A3B8; font-size: 13px; font-style: italic; padding: 2px 0; }
.tool-done { color: #22C55E; font-size: 13px; font-weight: 600; padding: 2px 0; }
</style>""", unsafe_allow_html=True)

from utils import inject_css
inject_css()
with st.sidebar:
    st.markdown("## TRACE 🛡️")
    st.caption("Trade-linked Risk Assessment & Contagion Engine")
    st.divider()
    try:
        from utils import load_base_feature
        bf = load_base_feature()
        st.info(f"数据周期\n{bf['quarter'].min()} – {bf['quarter'].max()}")
    except: pass
    st.divider()
    st.caption("Python · XGBoost · DeepSeek · GAT")

st.title("🤖 TRACE AI 风险评估助手")

# ---- API Key ----
if "agent_api_key" not in st.session_state:
    saved = ""
    if CONFIG_FILE.exists():
        try: saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8")).get("api_key", "")
        except: pass
    if not saved: saved = os.getenv("DEEPSEEK_API_KEY", "")
    st.session_state.agent_api_key = saved

with st.expander("⚙️ API 设置", expanded=not st.session_state.agent_api_key):
    new_key = st.text_input("DeepSeek API Key", value=st.session_state.agent_api_key,
        type="password", placeholder="sk-...")
    if new_key != st.session_state.agent_api_key:
        st.session_state.agent_api_key = new_key
        CONFIG_FILE.write_text(json.dumps({"api_key": new_key}, ensure_ascii=False), encoding="utf-8")
        st.rerun()

if not st.session_state.agent_api_key:
    st.info("请先设置 DeepSeek API Key"); st.stop()

from openai import OpenAI
client = OpenAI(api_key=st.session_state.agent_api_key, base_url=BASE_URL)

# ---- Tools ----
SYS = "你是 TRACE 金融风险分析助手。你可以查询公司风险评分/评级/趋势、分析供应链传导、解读宏观指标、模拟关税冲击、生成风险报告。回复使用中文, 简洁专业, 引用具体数字。"

TOOLS = [
    {"type":"function","function":{"name":"query_risk_score","description":"查询公司综合风险评分/评级/趋势/关键因素","parameters":{"type":"object","properties":{"code":{"type":"string","description":"股票代码"}},"required":["code"]}}},
    {"type":"function","function":{"name":"trace_supply_chain","description":"分析公司或行业的供应链上下游传导风险","parameters":{"type":"object","properties":{"code_or_industry":{"type":"string"}},"required":["code_or_industry"]}}},
    {"type":"function","function":{"name":"analyze_macro","description":"解读最新宏观经济指标(GDP/PMI/CPI/M2/Shibor/社融)","parameters":{"type":"object","properties":{"quarter":{"type":"string"}},"required":[]}}},
    {"type":"function","function":{"name":"simulate_tariff","description":"模拟美国关税变化对行业的影响和下游传导","parameters":{"type":"object","properties":{"company_or_industry":{"type":"string"},"rate_change_pct":{"type":"number","description":"如0.10=加征10%"}},"required":["company_or_industry","rate_change_pct"]}}},
    {"type":"function","function":{"name":"generate_report","description":"生成公司的完整风险分析报告(Markdown格式)","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}},
]

import importlib.util
spec = importlib.util.spec_from_file_location("agent", PROJECT_ROOT / "backend" / "agent.py")
agent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_mod)
TOOL_FN = {n: getattr(agent_mod, n) for n in ["query_risk_score","trace_supply_chain","analyze_macro","simulate_tariff","generate_report"]}

# ---- Chat state ----
if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "msg_counter" not in st.session_state:
    st.session_state.msg_counter = 0

# ---- Quick prompts ----
if not st.session_state.msgs:
    st.caption("💡 快速提问：")
    c1,c2,c3,c4 = st.columns(4)
    prompts = ["000488 风险怎么样？", "汽车行业加10%关税影响", "当前宏观环境如何？", "生成 000725 风险报告"]
    for col, p in zip([c1,c2,c3,c4], prompts):
        if col.button(p, key=f"qp_{hash(p) % 10000}", use_container_width=True):
            st.session_state.msgs.append({"role":"user","content":p})
            st.rerun()

# ---- Process pending user message ----
# A user message is "pending" if it's the last message in the list
# Processing happens BEFORE chat_input to avoid widget state conflicts
pending_user_msg = None
if st.session_state.msgs and st.session_state.msgs[-1]["role"] == "user":
    pending_user_msg = st.session_state.msgs[-1]["content"]

if pending_user_msg:
    chain = []

    # Build API messages from all history EXCEPT the pending user msg
    # (we'll add it as the current user prompt)
    api_msgs = [{"role":"system","content":SYS}]
    for m in st.session_state.msgs[:-1]:
        # Only include user and assistant messages for the API
        if m["role"] in ("user", "assistant"):
            api_msgs.append({"role": m["role"], "content": m["content"]})

    # Add the pending user message
    api_msgs.append({"role": "user", "content": pending_user_msg})

    # Process in a chat message container
    with st.chat_message("assistant"):
        status_container = st.status("💭 正在分析...", expanded=True, state="running")

        full = ""
        try:
            # Show history and process live
            with status_container:
                # Step 1: First API call to decide on tools
                st.write("💭 理解问题并选择工具...")
                resp = client.chat.completions.create(
                    model=MODEL, messages=api_msgs, tools=TOOLS,
                    temperature=0.3, max_tokens=800, timeout=45,
                )
                msg = resp.choices[0].message

                if msg.content:
                    chain.append({"type":"think","text":msg.content[:400]})
                    st.write(f"💭 {msg.content[:200]}")

                if msg.tool_calls:
                    st.write(f"🔧 需要调用 {len(msg.tool_calls)} 个工具")
                    chain.append({"type":"think","text":f"调用 {len(msg.tool_calls)} 个工具"})
                    api_msgs.append(msg)

                    # Execute each tool call
                    for tc in msg.tool_calls:
                        fn = TOOL_FN.get(tc.function.name)
                        try:
                            args = json.loads(tc.function.arguments)
                        except Exception:
                            args = {}

                        label = f"调用 `{tc.function.name}`"
                        st.write(f"⏳ {label} — {json.dumps(args, ensure_ascii=False)}")
                        chain.append({"type":"tool_start","name":tc.function.name,
                                       "label":"🔧","args":json.dumps(args, ensure_ascii=False)})

                        try:
                            result = fn(**args)
                            if isinstance(result, str) and len(result) > 5000:
                                result = result[:5000]
                            elif not isinstance(result, str):
                                result = str(result)[:5000]
                        except Exception as e:
                            result = json.dumps({"error": str(e)}, ensure_ascii=False)

                        st.write(f"✅ `{tc.function.name}` 完成 ({len(result)} 字符)")
                        chain.append({"type":"tool_done","name":tc.function.name,
                                       "label":"✅","args":json.dumps(args, ensure_ascii=False),
                                       "result":result})
                        api_msgs.append({"role":"tool","tool_call_id":tc.id,"content":result})

                    # Step 2: Final response with tool results
                    st.write("💭 基于数据生成分析...")
                    chain.append({"type":"think","text":"基于数据生成分析结论..."})

                    resp2 = client.chat.completions.create(
                        model=MODEL, messages=api_msgs, temperature=0.3,
                        max_tokens=1200, timeout=60,
                    )
                    full = resp2.choices[0].message.content or ""
                else:
                    full = msg.content or ""

            status_container.update(label="✅ 分析完成", state="complete", expanded=False)
            st.markdown(full)

        except Exception as e:
            err_msg = str(e)[:300]
            full = f"抱歉，分析过程出错：{err_msg}"
            chain.append({"type":"error","text":err_msg})
            try:
                status_container.update(label="❌ 出错了", state="error")
            except Exception:
                pass
            st.error(full)

        # Append assistant response (don't replace user msg!)
        st.session_state.msgs.append({"role":"assistant","content":full or "(无内容)","chain":chain})
        st.rerun()

# ---- Render history (after any processing) ----
for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            # Render thinking chain FIRST, then the final response
            for step in msg.get("chain", []):
                if step["type"] == "think":
                    st.markdown(f'<div class="think-line">💭 {step["text"]}</div>', unsafe_allow_html=True)
                elif step["type"] == "tool_start":
                    st.markdown(f'<div class="think-line">⏳ {step["label"]} — `{step["name"]}`</div>', unsafe_allow_html=True)
                elif step["type"] == "tool_done":
                    with st.expander(f'{step["label"]} `{step["name"]}` — {step.get("args","")[:80]}', expanded=False):
                        st.caption(str(step.get("result",""))[:500])
                elif step["type"] == "error":
                    st.error(step["text"])
            st.markdown(msg.get("content") or "")

# ---- Clear button ----
if st.session_state.msgs:
    c1, _ = st.columns([1, 8])
    if c1.button("🗑️ 清空对话"):
        st.session_state.msgs = []
        st.session_state.msg_counter = 0
        st.rerun()

# ---- Chat input (must be last to avoid widget state conflicts) ----
# Use a dynamic key based on msg_counter to force fresh widget on each round
input_key = f"chat_input_{st.session_state.msg_counter}"
if prompt := st.chat_input("输入股票代码或问题...", key=input_key):
    st.session_state.msgs.append({"role":"user","content":prompt})
    st.session_state.msg_counter += 1
    st.rerun()
