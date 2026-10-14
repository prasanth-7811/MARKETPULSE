import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from mp import ui
from mp.explain import TREE, perm_importance, shap_for, subsample
from mp.regimes import CLASSES, COLORS

ctx = ui.get_ctx()
ui.header("Explainable AI", "Understand which input features influenced MarketPulse predictions.", ctx)
R, cfg = ctx.R, ctx.cfg
data, T = R["data"], ui.T
ui.note("SHAP values describe how the fitted model's output depends on each input for these observations. They indicate association within the model, not causation in markets.")


@st.cache_resource(max_entries=8, show_spinner="Computing explanations...")
def _shap(key, model_name, kind, h=0):
    if kind == "trans":
        e = R["transition"][h]; X, rows = R["Xt"], subsample(e["idx"]["test"]); clf = e["model"]
    else:
        X, rows = R["Xb"], subsample(R["idx"]["test"]); clf = R["models"][model_name]
    sv, present = shap_for(clf, X.iloc[rows], binary_positive=(kind == "trans"))
    return sv, present, rows


@st.cache_resource(max_entries=8, show_spinner="Computing permutation importance...")
def _perm(key, model_name):
    rows = subsample(R["idx"]["test"], 600)
    return perm_importance(R["models"][model_name], R["Xb"].iloc[rows], R["y_next"][rows])


sel = [m for m in cfg["models"]["selected"] if m in R["models"]]
model = st.selectbox("Model", sel, index=sel.index("LightGBM") if "LightGBM" in sel else 0)
tree = model in TREE
cols = R["models"][model].cols

st.subheader("Global feature importance")
if tree:
    sv, present, rows = _shap(ctx.key, model, "cls")
    imp = pd.Series(np.abs(sv).mean(axis=(0, 2)), index=cols).nlargest(15)[::-1]; method = "mean |SHAP| over test rows and classes"
else:
    imp = _perm(ctx.key, model).head(15)[::-1]; method = "permutation importance (drop in macro F1) on test rows; SHAP unavailable for this model"
fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=T["accent"])); fig.update_xaxes(title=method); ui.show(fig, 420, False)

if tree:
    st.subheader("SHAP summary")
    present_labels = [CLASSES[i] for i in present]
    cls = st.selectbox("Regime (class) to explain", present_labels, index=0)
    ci = present_labels.index(cls)
    top = pd.Series(np.abs(sv[:, :, ci]).mean(0), index=cols).nlargest(10).index.tolist()[::-1]
    Xs = R["Xb"].iloc[rows][cols]; rng = np.random.default_rng(0); fig = go.Figure()
    for j, f in enumerate(top):
        v = Xs[f].rank(pct=True).to_numpy()
        fig.add_trace(go.Scattergl(x=sv[:, cols.index(f), ci], y=j + rng.uniform(-.25, .25, len(v)), mode="markers", showlegend=False,
                                   marker=dict(size=5, color=v, colorscale=[[0, "#38BDF8"], [1, "#F87171"]], cmin=0, cmax=1, showscale=(j == 0), colorbar=dict(title="Feature value<br>(percentile)", len=.6) if j == 0 else None),
                                   customdata=np.c_[Xs[f].to_numpy()], hovertemplate=f"{f}=%{{customdata[0]:.4f}}<br>SHAP %{{x:.4f}}<extra></extra>"))
    fig.update_yaxes(tickvals=list(range(len(top))), ticktext=top); fig.update_xaxes(title=f"SHAP value (push toward / away from '{ui.short(cls)}')"); ui.show(fig, 420, False)
    st.caption(f"Each dot is a test-period day. Right of zero raises the model's score for {cls}; red dots are days where the feature was high relative to its distribution, blue where it was low.")

st.subheader("Local prediction explanation")
test_idx = R["idx"]["test"]
dsel = st.selectbox("Observation (test period)", [data.index[i] for i in test_idx[::-1]], format_func=lambda d: f"{d:%Y-%m-%d}")
i = int(data.index.get_loc(dsel)); P = R["proba"][model][i]; yk = R["y_next"][i]
c = st.columns(3)
ui.kpi(c[0], "Predicted next regime", ui.short(CLASSES[int(P.argmax())]), f"P = {ui.pct(P.max())}")
ui.kpi(c[1], "Actual next regime", ui.short(CLASSES[yk]) if yk >= 0 else "not yet observable", "correct" if yk == P.argmax() else "incorrect" if yk >= 0 else "")
ui.kpi(c[2], "Confidence", ui.pct(P.max()), "above threshold" if P.max() >= cfg["calibration"]["confidence_threshold"] else "below threshold")
order = np.argsort(P)
ui.show(go.Figure(go.Bar(x=P[order], y=[ui.short(CLASSES[k]) for k in order], orientation="h", marker_color=[COLORS[CLASSES[k]] for k in order])).update_xaxes(tickformat=".0%", title="Class probability"), 240, False)
if tree:
    clf = R["models"][model]; s1, pres = shap_for(clf, R["Xb"].iloc[[i]])
    pred_c = int(P.argmax()); k = list(pres).index(pred_c) if pred_c in pres else 0
    contrib = pd.Series(s1[0, :, k], index=cols); contrib = contrib.reindex(contrib.abs().nlargest(10).index)[::-1]
    fig = go.Figure(go.Bar(x=contrib.values, y=[f"{f} = {R['Xb'].iloc[i][f]:.4f}" for f in contrib.index], orientation="h", marker_color=np.where(contrib.values >= 0, T["pos"], T["neg"])))
    fig.update_xaxes(title=f"SHAP contribution toward '{ui.short(CLASSES[pres[k]])}'"); ui.show(fig, 360, False)
else:
    ui.warn("Per-observation SHAP is not available for this model; select a tree model (Random Forest, XGBoost, LightGBM). Global permutation importance is shown above.")

st.subheader("Transition explanation")
hs = sorted(h for h, e in R["transition"].items() if e.get("model") is not None)
h = st.selectbox("Horizon (trading days)", hs, index=hs.index(ui.cur_h(ctx)) if ui.cur_h(ctx) in hs else 0)
svt, _, trows = _shap(ctx.key, "Transition", "trans", h); tcols = R["transition"][h]["model"].cols
imp_t = pd.Series(np.abs(svt[:, :, 0]).mean(0), index=tcols).nlargest(12)[::-1]
a, b = st.columns(2)
with a: ui.show(go.Figure(go.Bar(x=imp_t.values, y=imp_t.index, orientation="h", marker_color=T["accent2"])).update_xaxes(title="mean |SHAP| (test rows)"), 380, False)
with b:
    s2, _ = shap_for(R["transition"][h]["model"], R["Xt"].iloc[[i]], binary_positive=True)
    ct = pd.Series(s2[0, :, 0], index=tcols); ct = ct.reindex(ct.abs().nlargest(10).index)[::-1]
    ui.show(go.Figure(go.Bar(x=ct.values, y=ct.index, orientation="h", marker_color=np.where(ct.values >= 0, T["warn"], T["accent"]))).update_xaxes(title=f"SHAP toward 'regime change' on {dsel:%Y-%m-%d}"), 380, False)

if tree:
    st.subheader("Regime comparison")
    topf = pd.Series(np.abs(sv).mean(axis=(0, 2)), index=cols).nlargest(12).index
    Z = np.array([[np.abs(sv[:, cols.index(f), k]).mean() for f in topf] for k in range(len(present))])
    fig = go.Figure(go.Heatmap(z=Z, x=list(topf), y=[ui.short(CLASSES[p]) for p in present], colorscale="Teal", hovertemplate="%{y} / %{x}: %{z:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed"); ui.show(fig, 340, False)
    st.caption("Mean |SHAP| of each feature for each regime's output over test rows: which inputs matter most for which regime.")
