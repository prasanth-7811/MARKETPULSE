import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve
from mp import ui
from mp.models import binary_metrics

ctx = ui.get_ctx()
ui.header("Transition Intelligence", "Estimate the likelihood of a market regime transition over a configurable future horizon.", ctx)
R, cfg = ctx.R, ctx.cfg
data, T = R["data"], ui.T
h = ui.cur_h(ctx); tr = R["transition"][h]; last = len(data) - 1
if tr.get("model") is None:
    ui.warn(f"Horizon {h}d is unavailable: {tr.get('error')}"); st.stop()
ui.note("A transition probability is the estimated chance that the regime LABEL changes within the horizon. It is not a probability that the market rises or falls.")

c = st.columns(3)
p_now = tr["p_cal"][last]
ui.kpi(c[0], f"P(regime change within {h} trading days)", ui.pct(p_now), f"Calibration: {tr['cal_method']}. Raw model output: {ui.pct(tr['p_raw'][last])}", T["warn"] if p_now >= cfg["transition"]["threshold"] else T["accent2"])
ev = binary_metrics(tr["y"][tr["idx"]["test"]], tr["p_cal"][tr["idx"]["test"]], cfg["transition"]["threshold"])
ui.kpi(c[1], "Held-out PR-AUC", ui.pct(ev["pr_auc"]) if not np.isnan(ev["pr_auc"]) else "n/a", f"Test prevalence (no-skill PR-AUC): {ui.pct(ev['prevalence'])}")
ui.kpi(c[2], "Held-out Brier score", f"{ev['brier']:.3f}", f"Test events: {ev['events']} of {ev['n']}")
if tr["cal_note"]: ui.warn(tr["cal_note"])

st.subheader("Horizon comparison")
rows = []
for k, e in sorted(R["transition"].items()):
    if e.get("model") is None: rows.append({"Horizon": f"{k}d", "Status": e.get("error")}); continue
    m = binary_metrics(e["y"][e["idx"]["test"]], e["p_cal"][e["idx"]["test"]], cfg["transition"]["threshold"])
    rows.append({"Horizon": f"{k}d", "Current P(change)": e["p_cal"][last], "Test events": m["events"], "Test prevalence": m["prevalence"], "PR-AUC": m["pr_auc"], "ROC-AUC": m["roc_auc"], "Brier": m["brier"]})
st.dataframe(pd.DataFrame(rows).style.format({"Current P(change)": "{:.1%}", "Test prevalence": "{:.1%}", "PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}", "Brier": "{:.3f}"}, na_rep="n/a"), hide_index=True, use_container_width=True)
ui.note("Each horizon has its own separately trained and calibrated model with its own target (change within k days). Probabilities are never rescaled across horizons.")

st.subheader("Historical transition timeline")
sp, x = R["split"].to_numpy(), data.index
fig = go.Figure()
for name, mask, col in (("In-sample prediction (train)", sp == "train", "#64748B"), ("Out-of-sample prediction (validation)", sp == "validation", T["accent"]), ("Out-of-sample prediction (test)", sp == "test", T["accent2"])):
    fig.add_trace(go.Scattergl(x=x[mask], y=tr["p_cal"][mask], mode="lines", name=name, line=dict(color=col, width=1.2)))
lab = tr["valid"] & (tr["y"] == 1)
fig.add_trace(go.Scattergl(x=x[lab], y=np.full(lab.sum(), -0.04), mode="markers", name="Retrospective label: change occurred", marker=dict(color=T["warn"], size=3, symbol="line-ns-open")))
fig.update_yaxes(tickformat=".0%", title="P(change)", range=[-0.08, 1.02]); ui.show(fig, 360)
ui.note("Labels are only observable h days after each date and are drawn retrospectively. Embargo rows between splits are excluded from training and evaluation.")

st.subheader("Evaluation")
c1, c2 = st.columns([1, 2])
split = c1.radio("Evaluation split", ["test", "validation"], horizontal=True)
thr = c2.slider("Classification threshold", 0.05, 0.95, float(cfg["transition"]["threshold"]), 0.01, help="Display-time only; does not retrain.")
idx = tr["idx"][split]; y, p, praw = tr["y"][idx], tr["p_cal"][idx], tr["p_raw"][idx]
m = binary_metrics(y, p, thr)
if m["events"] < 30 or m["n"] - m["events"] < 30: ui.warn(f"Only {m['events']} events / {m['n'] - m['events']} non-events in the {split} split: metrics are unreliable.")
a, b = st.columns(2)
with a:
    if len(np.unique(y)) == 2:
        pr, rc, _ = precision_recall_curve(y, p)
        fig = go.Figure(go.Scatter(x=rc, y=pr, name="Calibrated", line=dict(color=T["accent"]))).add_hline(y=m["prevalence"], line_dash="dot", line_color=T["muted"], annotation_text="no-skill")
        fig.update_xaxes(title="Recall"); fig.update_yaxes(title="Precision"); fig.update_layout(title=dict(text=f"Precision-recall (PR-AUC {m['pr_auc']:.3f})", font=dict(size=12))); ui.show(fig, 300, False)
        fpr, tpr, _ = roc_curve(y, p)
        fig = go.Figure(go.Scatter(x=fpr, y=tpr, line=dict(color=T["accent2"]))).add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color=T["muted"]))
        fig.update_xaxes(title="False positive rate"); fig.update_yaxes(title="True positive rate"); fig.update_layout(title=dict(text=f"ROC (AUC {m['roc_auc']:.3f})", font=dict(size=12))); ui.show(fig, 300, False)
    else: ui.warn("Only one class present in this split; PR and ROC curves are undefined.")
with b:
    cm = m["confusion"]
    fig = go.Figure(go.Heatmap(z=cm, x=["Pred: no change", "Pred: change"], y=["Actual: no change", "Actual: change"], text=cm, texttemplate="%{text}", colorscale="Teal", showscale=False))
    fig.update_yaxes(autorange="reversed"); fig.update_layout(title=dict(text=f"Confusion matrix (threshold {thr:.2f})", font=dict(size=12))); ui.show(fig, 300, False)
    if len(np.unique(y)) == 2:
        fig = go.Figure().add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color=T["muted"]))
        for nm, pp, col in (("Raw", praw, T["muted"]), ("Calibrated", p, T["accent2"])):
            fy, fx = calibration_curve(y, np.clip(pp, 0, 1), n_bins=8, strategy="quantile")
            fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines+markers", name=nm, line=dict(color=col)))
        fig.update_xaxes(title="Mean predicted probability"); fig.update_yaxes(title="Observed frequency"); fig.update_layout(title=dict(text="Reliability diagram", font=dict(size=12))); ui.show(fig, 300)

st.subheader("Event statistics and calibration comparison")
mr = binary_metrics(y, np.clip(praw, 0, 1), thr)
st.dataframe(pd.DataFrame({"Metric": ["Observed transition events", "Event prevalence", "Precision", "Recall", "F1-score", "PR-AUC", "Brier (calibrated)", "Brier (raw)"],
                           "Value": [m["events"], f"{m['prevalence']:.1%}", f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}",
                                     "n/a" if np.isnan(m["pr_auc"]) else f"{m['pr_auc']:.3f}", f"{m['brier']:.4f}", f"{mr['brier']:.4f}"]}), hide_index=True, use_container_width=True)
