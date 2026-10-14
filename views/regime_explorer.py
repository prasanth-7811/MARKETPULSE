import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from mp import ui
from mp.regimes import COLORS, STATE_PALETTE

ctx = ui.get_ctx()
ui.header("Regime Explorer", "Explore latent market states, historical regime duration, and changes in market behavior.", ctx)
R, cfg, df = ctx.R, ctx.cfg, ctx.df
data, T = R["data"], ui.T
names, K = R["state_names"], len(R["state_names"])
col_of = {n: STATE_PALETTE[i % len(STATE_PALETTE)] for i, n in enumerate(names)}
hs, runs = R["hmm_state"], R["hmm_runs"]
tr_idx = R["split"].eq("train").to_numpy()
st.caption(f"HMM fitted on training rows only: {data.index[tr_idx][0]:%Y-%m-%d} to {data.index[tr_idx][-1]:%Y-%m-%d}. States shown are FILTERED (no look-ahead); smoothing is not used.")

rows = []
for n in names:
    m = (hs == n).to_numpy(); d = runs[runs.regime == n].duration
    rows.append({"Regime": n, "Observations": int(m.sum()), "Share": m.mean(), "Avg daily return": data["ret"][m].mean() if m.any() else np.nan,
                 "Return volatility (daily)": data["ret"][m].std() if m.sum() > 1 else np.nan, "Avg realised vol (ann.)": data[R["hmm_cols"][1]][m].mean() if m.any() else np.nan,
                 "Avg drawdown": data["drawdown"][m].mean() if m.any() else np.nan, "Median duration (days)": d.median() if len(d) else np.nan})
S = pd.DataFrame(rows)
c = st.columns(4)
ui.kpi(c[0], "Detected regimes", str(K), f"{cfg['hmm']['covariance_type']} covariance")
ui.kpi(c[1], "Most frequent", ui.short(S.loc[S.Observations.idxmax(), "Regime"]), ui.pct(S.Share.max()) + " of observations")
ui.kpi(c[2], "Median regime duration", f"{runs.duration.median():.0f} d", f"{len(runs)} regime spells")
ui.kpi(c[3], "Train log-likelihood", f"{R['hmm'].train_loglik:.0f}", "best of 4 EM restarts")

st.subheader("Regime-coloured price")
price = df["adj_close"].reindex(data.index)
fig = ui.regime_traces(go.Figure(), data.index, price.values, hs.values, col_of, extra=R["hmm_filt"][[f"hmm_p{k}" for k in range(K)]].max(axis=1).values)
fig.update_yaxes(title="Adjusted close"); ui.show(fig, 400)

a, b = st.columns(2)
with a:
    st.subheader("HMM transition matrix")
    A = R["hmm"].transmat
    # states ordered as ids; labels shown with ids so duplicates stay readable
    lbl = [f"S{i}: {ui.short(n)}" for i, n in enumerate(names)]
    fig = go.Figure(go.Heatmap(z=A, x=lbl, y=lbl, text=np.round(A, 3), texttemplate="%{text}", colorscale="Teal", zmin=0, zmax=1,
                               hovertemplate="From %{y}<br>To %{x}<br>P=%{z:.3f}<extra></extra>"))
    fig.update_xaxes(title="Next state"); fig.update_yaxes(title="Previous state", autorange="reversed"); ui.show(fig, 340, False)
    ui.note("Rows: previous latent state. Columns: next latent state. This describes learned latent-state dynamics on the training period, not guaranteed future behaviour.")
with b:
    st.subheader("State label mapping (training period)")
    mp_ = R["state_stats"].copy(); mp_.insert(0, "Label", names); mp_.index = [f"S{i}" for i in range(K)]
    mp_ = mp_.rename(columns={"mom": "Mean momentum", "vol": "Mean volatility", "ret": "Mean daily return", "n_train": "Train obs"})
    st.dataframe(mp_.style.format({"Mean momentum": "{:.2%}", "Mean volatility": "{:.1%}", "Mean daily return": "{:.3%}"}), use_container_width=True)
    ui.note("HMM state ids are arbitrary. Labels come from each state's training-period mean momentum (trend) and volatility (vs the median state) and are documented here.")

st.subheader("Regime characteristics")
st.dataframe(S.style.format({"Share": "{:.1%}", "Avg daily return": "{:.3%}", "Return volatility (daily)": "{:.3%}", "Avg realised vol (ann.)": "{:.1%}", "Avg drawdown": "{:.1%}", "Median duration (days)": "{:.0f}"}),
             hide_index=True, use_container_width=True)

st.subheader("Regime distributions")
a, b = st.columns(2)
with a:
    ui.show(go.Figure(go.Bar(x=[ui.short(n) for n in S.Regime], y=S.Observations, marker_color=[col_of[n] for n in S.Regime])).update_layout(title=dict(text="Regime frequency", font=dict(size=12))), 280, False)
    fig = go.Figure()
    for n in names:
        fig.add_trace(go.Box(y=data["ret"][hs == n], name=ui.short(n), marker_color=col_of[n], boxpoints=False))
    ui.show(fig.update_layout(title=dict(text="Daily return by regime", font=dict(size=12))).update_yaxes(tickformat=".1%"), 300, False)
with b:
    fig = go.Figure()
    for n in names:
        fig.add_trace(go.Box(y=runs[runs.regime == n].duration, name=ui.short(n), marker_color=col_of[n]))
    ui.show(fig.update_layout(title=dict(text="Regime duration (days per spell)", font=dict(size=12))), 280, False)
    fig = go.Figure()
    for n in names:
        fig.add_trace(go.Box(y=data[R["hmm_cols"][1]][hs == n], name=ui.short(n), marker_color=col_of[n], boxpoints=False))
    ui.show(fig.update_layout(title=dict(text="Realised volatility by regime", font=dict(size=12))).update_yaxes(tickformat=".0%"), 300, False)

st.subheader("Model controls")
with st.form("hmm_form"):
    c1, c2, c3 = st.columns(3)
    k = c1.slider("Number of HMM states", 2, 6, cfg["hmm"]["n_states"])
    ct = c2.selectbox("Covariance type", ["full", "diag", "tied", "spherical"], ["full", "diag", "tied", "spherical"].index(cfg["hmm"]["covariance_type"]))
    c3.markdown(f"Training period<br>**{cfg['split']['train']:.0%}** of data (edit in Settings)", unsafe_allow_html=True)
    if st.form_submit_button("Retrain models with these HMM settings"):
        ui.set_override({"hmm": {"n_states": k, "covariance_type": ct}}); st.rerun()
st.caption(f"Features used by the HMM: {', '.join(R['hmm_cols'])} (standardised on training rows). HMM states also feed the HMM-augmented ensemble.")
