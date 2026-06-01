"""TRACE 仪表盘入口"""
import streamlit as st

st.set_page_config(page_title="TRACE · 信用风险评估", page_icon="🛡️", layout="wide")

from utils import inject_css, render_sidebar

inject_css()
render_sidebar()

st.title("TRACE 企业信用风险智能评估平台")
st.markdown("""
**Trade-linked Risk Assessment and Contagion Engine** —
融合财务报表、宏观指标、NLP舆情、投入产出表与图神经网络的
多源数据风控平台。

左侧导航选择功能模块:
- **宏观总览** — 核心宏观经济与流动性指标走势
- **公司查询** — 单企业信用评分、违约概率与归因分析
- **风险分布** — PCA降维的行业风险全景散点图
- **供应链网络** — 110个申万行业投入产出传导网络
""")
st.caption("Python 3.10+ · XGBoost · DeepSeek LLM · PyTorch Geometric · Streamlit")
