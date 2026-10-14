import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from mp import reduce, ui
from mp.regimes import CLASSES, COLORS

ctx = ui.get_ctx()
ui.header("Feature Lab", "Engineered feature groups, cross-asset signals and dimensionality reduction (PCA, t-SNE).", ctx)
R, df, T = ctx.R, ctx.df, ui.T
Xb, tr = R["Xb"], R["idx"]["train"]
ui.note("PCA is fitted on training rows only and then applied to all rows. t-SNE is a visual diagnostic on a subsample and is never used as model input.")

st.subheader("Feature inventory")
inv = pd.DataFrame([(g, len(c), ", ".join(c)) for g, c in R["groups"].items()], columns=["Group", "Count", "Features"])
st.dataframe(inv, hide_index=True, use_container_width=True)
st.caption(f"{Xb.shape[1]} causal features. All rolling statistics use data up to and including day t only (verified by a unit test that perturbs the future).")


@st.cache_data(show_spinner=False)
def _pca(key, var):
    red = reduce.PCAReducer(var).fit(Xb.iloc[tr]); return red, red.transform(Xb)


red, Z = _pca(ctx.key, ctx.cfg["reduction"]["pca_variance"] if "reduction" in ctx.cfg else 0.95)
lab = R["labels"]
st.subheader("PCA")
c1, c2 = st.columns(2)
sc = red.scree()[:20]
fig = go.Figure([go.Bar(x=sc.component, y=sc.explained, name="Explained", marker_color=T["accent"]), go.Scatter(x=sc.component, y=sc.cumulative, name="Cumulative", mode="lines+markers", line=dict(color=T["warn"]))])
fig.add_hline(y=.95, line_dash="dot", line_color=T["muted"]); fig.update_layout(title=dict(text=f"Scree: {red.k} components reach 95% of variance", font=dict(size=12))); c1.plotly_chart(ui.style(fig, 340), use_container_width=True)
a, b = c2.columns(2); pcx, pcy = a.selectbox("X", list(Z.columns[:10]), 0), b.selectbox("Y", list(Z.columns[:10]), 1)
sub = Z.iloc[::2]; fig = go.Figure()
for name, col in COLORS.items():
    m = (lab.reindex(sub.index) == name).to_numpy()
    if m.any(): fig.add_trace(go.Scattergl(x=sub[pcx][m], y=sub[pcy][m], mode="markers", name=ui.short(name), marker=dict(color=col, size=4, opacity=.65)))
c2.plotly_chart(ui.style(fig, 300).update_xaxes(title=pcx).update_yaxes(title=pcy), use_container_width=True)
st.markdown("**Component loadings (what each PC is made of)**")
L = red.loadings(4); st.dataframe(L.reindex(L.abs().max(1).sort_values(ascending=False).index).head(14).style.format("{:.2f}").background_gradient(cmap="RdBu_r", vmin=-.5, vmax=.5), use_container_width=True)
try:
    sil = reduce.silhouette_by_regime(Z.iloc[:, :red.k], lab.reindex(Z.index))
    st.caption(f"Silhouette of the 6 regimes in the {red.k}-PC space: {sil:.2f} (near 0 = heavily overlapping clusters; regimes are rule-defined bands of continuous signals, not natural clusters).")
except Exception:
    pass

st.subheader("t-SNE (non-linear view)")
if st.button("Compute t-SNE (about 20-40 s)"):
    with st.spinner("Embedding..."):
        Xz = ((Xb.iloc[tr] - Xb.iloc[tr].mean()) / Xb.iloc[tr].std()).clip(-5, 5)
        st.session_state["tsne"] = reduce.tsne_embed(Xz, 1200, perplexity=30)
emb = st.session_state.get("tsne")
if emb is not None:
    fig = go.Figure()
    for name, col in COLORS.items():
        m = (lab.reindex(emb.index) == name).to_numpy()
        if m.any(): fig.add_trace(go.Scattergl(x=emb.tsne1[m], y=emb.tsne2[m], mode="markers", name=ui.short(name), marker=dict(color=col, size=5, opacity=.7)))
    st.plotly_chart(ui.style(fig, 420), use_container_width=True)

st.subheader("Cross-asset signals")
if "cross_asset" in R["groups"]:
    cols = R["groups"]["cross_asset"]; a, b = st.columns(2)
    d = R["data"]; fig = go.Figure()
    if "vix" in df:
        fig.add_trace(go.Scatter(x=d.index, y=df["vix"].reindex(d.index), name="VIX", line=dict(color=T["warn"], width=1)))
        fig.add_trace(go.Scatter(x=d.index, y=R["data"][f"vol_20"] * 100 if "vol_20" in d else None, name="Realised vol 20d (%)", line=dict(color=T["accent"], width=1)))
    a.plotly_chart(ui.style(fig.update_layout(title=dict(text="Implied (VIX) vs realised volatility", font=dict(size=12))), 300), use_container_width=True)
    corr = pd.concat([Xb[cols], R["data"][["sig_mom", "sig_vol"]]], axis=1).corr().loc[cols, ["sig_mom", "sig_vol"]]
    b.markdown("**Correlation of cross-asset features with the regime signals**"); b.dataframe(corr.style.format("{:.2f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1), use_container_width=True)
else:
    ui.warn("No cross-asset columns in this dataset (use the Bundled source or add a 'VIX' column to your CSV).")

st.subheader("Correlation structure")
corr = Xb.corr(); order = corr.abs().sum().sort_values(ascending=False).index; corr = corr.loc[order, order]
ui.show(go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1, colorscale="RdBu_r")).update_yaxes(autorange="reversed"), 560, False)
