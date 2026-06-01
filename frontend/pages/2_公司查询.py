"""公司风险查询"""
import streamlit as st
import plotly.graph_objects as go
from utils import (inject_css, render_sidebar, load_fusion_scores,
    load_base_feature, load_company_info, load_feature_importance,
    get_rating_badge, find_company)

st.set_page_config(page_title="公司查询", layout="wide")
inject_css()
render_sidebar()

st.header("公司信用风险查询")

scores = load_fusion_scores()
bf = load_base_feature()
info = load_company_info()
imp = load_feature_importance()

query = st.text_input("输入股票代码或公司名称", placeholder="000488 或 晨鸣纸业")

if query and not scores.empty and not info.empty:
    code = find_company(query, info, scores)
    if code is None:
        st.error("未找到匹配的公司")
        st.stop()

    name = info[info["ts_code"]==code]["name"].values[0] if code in info["ts_code"].values else code
    industry = info[info["ts_code"]==code]["industry"].values[0] if code in info["ts_code"].values else "未知"

    comp = scores[scores["code"]==code].sort_values("quarter")
    if comp.empty:
        st.warning(f"{code} 暂无评分数据")
        st.stop()

    latest = comp.iloc[-1]
    latest_q = str(latest["quarter"])
    score_val = int(latest.get("score", 0))
    prob_val = float(latest["prob"]) if "prob" in latest.index else 0
    rating, badge = get_rating_badge(score_val)

    # Percentile
    curr_all = scores[scores["quarter"]==latest_q]
    if not curr_all.empty and "score" in curr_all.columns:
        pct = (curr_all["score"] < score_val).mean() * 100
    else:
        pct = 50

    ind_avg = None
    if not curr_all.empty and "score" in curr_all.columns:
        ind_codes = info[info["industry"]==industry]["ts_code"].tolist()
        ind_all = curr_all[curr_all["code"].isin(ind_codes)]
        if not ind_all.empty:
            ind_avg = int(ind_all["score"].mean())

    # Layout
    left, right = st.columns([0.55, 0.45])
    with left:
        st.markdown(f"### {name}")
        st.caption(f"{code}  |  {industry}  |  {latest_q}")
        st.markdown(f"""
        <div style="display:flex;align-items:baseline;gap:16px;margin:10px 0">
            <span style="font-size:3.5rem;font-weight:800;color:#1E293B">{score_val}</span>
            <span class="{badge}" style="font-size:1.2rem">{rating}</span>
            <span style="color:#64748B;font-size:0.95rem">
                违约概率 <b style="color:#EF4444">{prob_val*100:.2f}%</b>
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"超越同季度 {pct:.0f}% 的公司" + (f" | 行业均值 {ind_avg}" if ind_avg else ""))

        # Trend
        trend = comp.tail(8)
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=trend["quarter"], y=trend["score"],
            mode="lines+markers", fill="tozeroy", line=dict(color="#3B82F6", width=3)))
        fig_t.update_layout(title="评分趋势", template="plotly_white",
            height=320, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_t, use_container_width=True)

    with right:
        st.markdown("#### 关键特征 vs 行业均值")
        if not imp.empty and not bf.empty:
            comp_q = bf[(bf["code"]==code) & (bf["quarter"]==latest_q)]
            ind_q = bf[(bf["industry"]==industry) & (bf["quarter"]==latest_q)]

            if not comp_q.empty and not ind_q.empty:
                top5 = imp.head(5)["feature"].tolist()
                features, company_vals, industry_vals = [], [], []
                for f in top5:
                    if f in comp_q.columns and f in ind_q.columns:
                        features.append(f)
                        company_vals.append(float(comp_q[f].iloc[0]))
                        industry_vals.append(float(ind_q[f].mean()))

                fig_b = go.Figure()
                fig_b.add_trace(go.Bar(y=features, x=company_vals, name="本公司",
                    orientation="h", marker_color="#3B82F6", text=[f"{v:.3f}" for v in company_vals],
                    textposition="outside"))
                fig_b.add_trace(go.Bar(y=features, x=industry_vals, name="行业均值",
                    orientation="h", marker_color="#CBD5E1"))
                fig_b.update_layout(barmode="group", template="plotly_white",
                    height=350, margin=dict(l=0, r=80, t=30, b=0))
                st.plotly_chart(fig_b, use_container_width=True)

    with st.expander("完整特征重要性排序"):
        if not imp.empty:
            st.dataframe(imp.head(30).style.background_gradient(cmap="Blues", subset=["importance_gain"]),
                use_container_width=True, hide_index=True)
