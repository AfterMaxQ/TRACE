"""PCA 风险分布散点图"""
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import inject_css, render_sidebar, load_base_feature, load_fusion_scores

st.set_page_config(page_title="风险分布", layout="wide")
inject_css()
render_sidebar()

st.header("行业风险全景图谱 (PCA)")

@st.cache_data
def compute_pca(quarter):
    df = load_base_feature()
    scores = load_fusion_scores()
    df_q = df[df["quarter"] == quarter].copy()
    if df_q.empty:
        return pd.DataFrame()

    exclude = ["code", "quarter", "name", "industry", "target"]
    feats = [c for c in df_q.columns if c not in exclude and df_q[c].dtype in ("float64","int64")]
    X = df_q[feats].fillna(0)
    X_s = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    comp = pca.fit_transform(X_s)

    df_q["PC1"] = comp[:, 0]
    df_q["PC2"] = comp[:, 1]

    sq = scores[scores["quarter"] == quarter]
    if "score" in sq.columns:
        sq = sq[["code", "prob", "score"]]
    else:
        sq = sq[["code", "prob"]]

    result = pd.merge(df_q[["code","name","industry","PC1","PC2"]], sq, on="code", how="inner")
    return result, pca.explained_variance_ratio_

bf = load_base_feature()
if bf.empty:
    st.warning("base_feature.csv 未找到")
    st.stop()

quarters = sorted(bf["quarter"].unique(), reverse=True)
sel_q = st.selectbox("选择季度", quarters, index=0)

pca_df, var_ratio = compute_pca(sel_q)
if pca_df.empty:
    st.warning("该季度无数据")
    st.stop()

industries = sorted(pca_df["industry"].dropna().unique())
sel_ind = st.multiselect("筛选行业 (留空=全部)", industries, default=[])

plot_df = pca_df[pca_df["industry"].isin(sel_ind)] if sel_ind else pca_df

st.caption(f"PCA解释方差: PC1={var_ratio[0]:.1%}, PC2={var_ratio[1]:.1%} | 展示 {len(plot_df)} 只股票")

fig = px.scatter(plot_df, x="PC1", y="PC2", color="prob",
    color_continuous_scale="RdYlGn_r",
    hover_name="name",
    hover_data={"code": True, "industry": True, "prob": ":.3f", "score": True if "score" in plot_df.columns else False, "PC1": False, "PC2": False},
    labels={"prob": "违约概率"})
fig.update_layout(template="plotly_white", height=650,
    coloraxis_colorbar=dict(title="风险", tickformat=".1%"))
st.plotly_chart(fig, use_container_width=True)
