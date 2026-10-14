import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from mp import ui
from mp.regimes import CLASSES, COLORS
from mp.features import mid

ctx = ui.get_ctx()
ui.header("Market Overview", "Understand current market conditions and model-estimated regime transitions.", ctx)
R, cfg, df = ctx.R, ctx.cfg, ctx.df
data, T = R["data"], ui.T
h = ui.cur_h(ctx); tr = R["transition"][h]; last = len(data) - 1
P, Praw = R["proba"]["Calibrated hybrid"][last], R["proba"]["HMM-augmented ensemble"][last]
cur = int(R["y"][last]); price = df["adj_close"].reindex(data.index)
H = -(P * np.log(np.clip(P, 1e-12, 1))).sum() / np.log(len(CLASSES))
thr = cfg["calibration"]["confidence_threshold"]
level = "High" if P.max() >= thr else "Low"
vw = mid(cfg["features"]["vol_windows"]); vol_now = data[f"vol_{vw}"].iloc[-1]

c = st.columns(4)
ui.kpi(c[0], "Current regime", f"<span style='color:{COLORS[CLASSES[cur]]}'>&#9679;</span> {ui.short(CLASSES[cur])}",
       f"Rule-based label. Model P(same regime next session): {ui.pct(P[cur])}")
ui.kpi(c[1], "Regime confidence", ui.pct(P.max()), f"{level} vs {thr:.0%} threshold. Raw ensemble max: {ui.pct(Praw.max())}. Normalised entropy {H:.2f}", T["accent"])
if tr.get("model") is not None:
    p_now = tr["p_cal"][last]
    ui.kpi(c[2], "Transition probability", ui.pct(p_now), f"P(regime label changes within next {h} trading days), {tr['cal_method']}-calibrated", T["warn"] if p_now > 0.5 else T["accent2"])
else:
    ui.kpi(c[2], "Transition probability", "n/a", tr.get("error", "Unavailable"))
ui.kpi(c[3], "Realized volatility", ui.pct(vol_now), f"{vw}-day window, annualised (x sqrt 252), log returns", T["text"])
ui.note("Maximum class probability is not a full measure of certainty: it can be miscalibrated or unstable. Entropy and the calibrated-vs-raw gap are shown for context.")

L, Rt = st.columns([2, 1])
with L:
    st.subheader("Market regime timeline")
    fig = ui.regime_traces(go.Figure(), data.index, price.values, R["labels"].values, COLORS)
    fig.update_yaxes(title="Adjusted close"); ui.show(fig, 400)
with Rt:
    st.subheader("AI market intelligence")
    st.markdown(f"**{CLASSES[cur]}** &nbsp; <span class='mp-pill'>as of {data.index[-1]:%Y-%m-%d}</span>", unsafe_allow_html=True)
    order = np.argsort(P)
    fig = go.Figure(go.Bar(x=P[order], y=[ui.short(CLASSES[i]) for i in order], orientation="h", marker_color=[COLORS[CLASSES[i]] for i in order],
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", range=[0, 1]); fig.update_layout(title=dict(text="Next-session regime probabilities (calibrated)", font=dict(size=12)))
    ui.show(fig, 240, False)
    st.markdown(f"**Uncertainty:** {'low' if H < .5 else 'moderate' if H < .75 else 'high'} (entropy {H:.2f})  \n**Transition ({h}d):** {ui.pct(tr['p_cal'][last]) if tr.get('model') is not None else 'n/a'}")
    mem = next((m for m in R["models"]["Calibrated hybrid"].members if hasattr(m.est, "feature_importances_")), None)
    if mem is not None:
        imp = pd.Series(mem.est.feature_importances_, index=mem.cols); imp = (imp / imp.sum()).nlargest(5)[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=T["accent"]))
        fig.update_layout(title=dict(text=f"Top features ({mem.name}, built-in importance)", font=dict(size=12))); fig.update_xaxes(tickformat=".0%")
        ui.show(fig, 220, False)
    st.page_link("views/explain.py", label="Open full explanation", icon=":material/psychology:")
    st.page_link("views/regime_explorer.py", label="Open Regime Explorer", icon=":material/explore:")

a, b = st.columns(2)
with a:
    st.subheader("Rolling volatility")
    fig = go.Figure()
    for col in R["groups"]["volatility"]:
        if col.startswith("vol_") and col != "vol_ratio": fig.add_trace(go.Scatter(x=data.index, y=data[col], name=col.replace("vol_", "") + "d", line=dict(width=1.3)))
    fig.update_yaxes(tickformat=".0%", title="Annualised"); ui.show(fig, 300)
with b:
    st.subheader("Daily returns")
    fig = go.Figure(go.Bar(x=data.index, y=data["ret"], marker_color=np.where(data["ret"] >= 0, T["pos"], T["neg"]), hovertemplate="%{x|%Y-%m-%d}: %{y:.2%}<extra></extra>"))
    fig.update_yaxes(tickformat=".1%", title="Log return"); ui.show(fig, 300, False)

a, b = st.columns([2, 1])
with a:
    st.subheader("Recent regime changes")
    lab = R["labels"]; chg = np.where(lab.values[1:] != lab.values[:-1])[0] + 1
    rows = [dict(Date=data.index[i].strftime("%Y-%m-%d"), Previous=ui.short(lab.iloc[i - 1]), New=ui.short(lab.iloc[i]),
                 **{"Model P(new), prior day": ui.pct(R["proba"]["Calibrated hybrid"][i - 1][R["y"][i]]), "Sample": R["split"].iloc[i]}) for i in chg[-12:][::-1]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    ui.note("Changes are in the rule-based regime label. 'Model P(new)' is the probability the calibrated hybrid assigned to the new regime from the previous day's features; 'train' rows are in-sample.")
with b:
    st.subheader("Data quality")
    q = ctx.quality
    st.dataframe(pd.DataFrame({"Check": ["Rows (raw / clean)", "Duplicate dates", "Invalid rows", "Missing cells", "Gaps > 5 days", "Usable rows after warm-up"],
                               "Value": [f"{q['rows_in']} / {q['rows_out']}", q["duplicate_dates"], q["invalid_rows"], sum(q["missing"].values()), q["gaps_over_5d"], len(data)]}),
                 hide_index=True, use_container_width=True)
