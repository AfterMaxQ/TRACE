"""宏观指标走势"""
import streamlit as st
import plotly.graph_objects as go
from utils import inject_css, render_sidebar, load_macro

st.set_page_config(page_title="宏观总览", layout="wide")
inject_css()
render_sidebar()

st.header("宏观经济与流动性总览")

df = load_macro()
if df.empty:
    st.warning("macro_quarterly.csv 未找到")
    st.stop()

# 确保列名统一
col_map = {}
for c in df.columns:
    if "quarter" in c.lower(): col_map["quarter"] = c
qcol = col_map.get("quarter", df.columns[0])
df = df.rename(columns={qcol: "quarter"})

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# 顶部指标卡片
c1, c2, c3, c4 = st.columns(4)
with c1:
    v = latest["gdp_yoy"]
    d = f"{v - prev['gdp_yoy']:+.1f} vs上季" if "gdp_yoy" in prev else None
    st.metric("GDP同比", f"{v:.2f}%", delta=d, delta_color="normal" if v > 4.5 else "inverse")
with c2:
    v = latest["pmi"]
    d = f"{v - prev['pmi']:+.1f}" if "pmi" in prev else None
    st.metric("制造业PMI", f"{v:.2f}", delta=d, delta_color="normal" if v > 50 else "inverse")
with c3:
    v = latest["cpi_yoy"]
    st.metric("CPI同比", f"{v:.2f}%", delta="通缩关注" if v < 0.5 else None)
with c4:
    v = latest["m2_yoy"]
    st.metric("M2同比", f"{v:.2f}%")

st.divider()

def _line(df, col, title, hline=None, color="#2563EB"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["quarter"], y=df[col], mode="lines+markers",
        line=dict(color=color, width=2.5), marker=dict(size=5)))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="#94A3B8",
            annotation_text=f"{hline}", annotation_position="bottom right")
    fig.update_layout(title=title, template="plotly_white",
        height=300, margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title=None, yaxis_title=None)
    return fig

r1c1, r1c2, r1c3 = st.columns(3)
with r1c1: st.plotly_chart(_line(df, "gdp_yoy", "GDP同比(%)", hline=5.0), use_container_width=True)
with r1c2: st.plotly_chart(_line(df, "pmi", "制造业PMI", hline=50), use_container_width=True)
with r1c3: st.plotly_chart(_line(df, "cpi_yoy", "CPI同比(%)", hline=2.0), use_container_width=True)

r2c1, r2c2, r2c3 = st.columns(3)
with r2c1: st.plotly_chart(_line(df, "m2_yoy", "M2同比(%)"), use_container_width=True)
with r2c2: st.plotly_chart(_line(df, "shibor_on", "Shibor隔夜(%)", color="#F59E0B"), use_container_width=True)
with r2c3:
    if "shero" in df.columns:
        st.plotly_chart(_line(df, "shero", "社融增量(万亿元)", color="#10B981"), use_container_width=True)
