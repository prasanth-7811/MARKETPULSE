import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from mp import benchmarks as B, errors as E, ui
from mp.regimes import CLASSES

ctx = ui.get_ctx()
ui.header("Benchmarks & Errors", "Hybrid model versus literature baselines, with bootstrap uncertainty and a structured error analysis.", ctx)
R, df, T = ctx.R, ctx.df, ui.T
ui.note("All numbers are on the held-out TEST split. Baselines are re-implementations of method families (Hamilton/HMM, RiskMetrics, GARCH, SMA rule, neural nets) - not replications of any paper's numbers. CIs use a moving-block bootstrap (20-day blocks).")


@st.cache_data(show_spinner=False)
def _bench(key): return B.run_all(R, df, int(ctx.cfg["split"]["seed"]), B=150)


if st.button("Run benchmark suite (about 1-2 min)", type="primary") or st.session_state.get("bench_key") == ctx.key:
    st.session_state["bench_key"] = ctx.key
    with st.spinner("Training baselines and bootstrapping..."): res = _bench(ctx.key)
else:
    st.info("Press the button to train the extra baselines (MLP, PCA+LR, GARCH, EWMA, SMA rule) and compute confidence intervals."); st.stop()

st.subheader("1. Six-class next-day regime task")
m = res["main"]; st.dataframe(m.style.format({c: "{:.3f}" for c in m.columns if c != "model"}), hide_index=True, use_container_width=True)
st.caption("Persistence (tomorrow = today) is very hard to beat on accuracy / macro-F1 because regimes are persistent; probabilistic metrics (log loss, Brier, ECE) separate the models more.")
with st.expander("Paired block-bootstrap tests: Calibrated hybrid minus each baseline (negative log-loss delta = hybrid better)"):
    st.dataframe(res["paired"].style.format({c: "{:.3f}" for c in res["paired"].columns if c != "comparison"}), hide_index=True, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("2. Volatility sub-task"); v = res["volatility"]; st.dataframe(v.style.format({c: "{:.3f}" for c in v.columns if c != "model"}), hide_index=True, use_container_width=True)
with c2:
    st.subheader("3. Trend sub-task"); t = res["trend"]; st.dataframe(t.style.format({c: "{:.3f}" for c in t.columns if c != "model"}), hide_index=True, use_container_width=True)
st.subheader("4. Regime-change (transition) task")
tt = res["transition"]; h = st.selectbox("Horizon", sorted(tt.horizon.unique()), index=0); x = tt[tt.horizon == h].drop(columns="horizon")
st.dataframe(x.style.format({c: "{:.3f}" for c in x.columns if c != "model"}), hide_index=True, use_container_width=True)
with st.expander("Calibration study (validation-fitted calibrators, test-scored)"):
    st.dataframe(res["calibration"].style.format({c: "{:.4f}" for c in ("brier", "log_loss", "ece", "roc_auc")}), hide_index=True, use_container_width=True)
c1, c2 = st.columns(2)
with c1:
    st.subheader("5. Feature-group ablation"); fa = res["feature_ablation"]; fa = fa[fa.split == "test"]
    fig = go.Figure(go.Scatter(x=fa.feature_set, y=fa.macro_f1, mode="lines+markers", line=dict(color=T["accent"]))); fig.update_yaxes(title="test macro-F1"); ui.show(fig, 280, False)
with c2:
    st.subheader("6. PCA input study"); st.dataframe(res["pca"].style.format({"macro_f1": "{:.3f}", "log_loss": "{:.3f}", "accuracy": "{:.3f}"}), hide_index=True, use_container_width=True, height=280)
st.subheader("Literature mapping"); st.dataframe(res["literature"], hide_index=True, use_container_width=True)

st.divider(); st.header("Error analysis")
model = st.selectbox("Model", ["Calibrated hybrid", "Soft-voting ensemble", "XGBoost", "Logistic Regression", "Persistence baseline"])
Ea = E.error_analysis(R, df, model, "test", res["proba_all"])
for line in E.findings(Ea): st.markdown(f"- {line}")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Per-class performance**"); st.dataframe(Ea["per_class"].style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1": "{:.2f}"}), hide_index=True, use_container_width=True)
    st.markdown("**Most frequent confusions**"); st.dataframe(Ea["top_confusions"].style.format({"share_of_errors": "{:.0%}"}), hide_index=True, use_container_width=True)
with c2:
    cm = Ea["confusion"]; z = cm.div(cm.sum(1).clip(lower=1), axis=0)
    fig = go.Figure(go.Heatmap(z=z.values, x=cm.columns, y=cm.index, text=cm.values, texttemplate="%{text}", colorscale="Teal", zmin=0, zmax=1)); fig.update_yaxes(autorange="reversed", title="actual"); fig.update_xaxes(title="predicted"); ui.show(fig, 360, False)
c1, c2, c3 = st.columns(3)
mg = Ea["by_margin"]; c1.plotly_chart(ui.style(go.Figure(go.Bar(x=mg.margin.astype(str), y=mg.error_rate, marker_color=T["accent"])).update_layout(title=dict(text="Error rate by distance to label threshold", font=dict(size=12))), 280, False), use_container_width=True)
if "by_vix" in Ea: bv = Ea["by_vix"]; c2.plotly_chart(ui.style(go.Figure(go.Bar(x=bv.vix_bucket.astype(str), y=bv.error_rate, marker_color=T["warn"])).update_layout(title=dict(text="Error rate by VIX regime", font=dict(size=12))), 280, False), use_container_width=True)
by = Ea["by_year"]; c3.plotly_chart(ui.style(go.Figure([go.Scatter(x=by.year, y=by.model_error, name=model, mode="lines+markers"), go.Scatter(x=by.year, y=by.persistence_error, name="persistence", mode="lines+markers")]).update_layout(title=dict(text="Error rate by year", font=dict(size=12))), 280), use_container_width=True)
c1, c2 = st.columns(2)
c1.markdown("**Error rate vs stated confidence**"); c1.dataframe(Ea["by_confidence"].style.format({"error_rate": "{:.1%}", "mean_conf": "{:.2f}"}), hide_index=True, use_container_width=True)
c2.markdown("**Most confident wrong predictions**"); c2.dataframe(Ea["worst_errors"].style.format({"confidence": "{:.2f}"}), hide_index=True, use_container_width=True)
st.download_button("Download benchmark table", res["main"].to_csv(index=False), "benchmark_main.csv", "text/csv")
