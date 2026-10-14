import json, platform
import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
from mp import ui
from mp.models import ALGOS, GRIDS, hyper_search
from mp.regimes import CLASSES

ctx = ui.get_ctx()
ui.header("Model Laboratory", "Compare individual algorithms, ensemble configurations, and hybrid market regime models.", ctx)
R, cfg = ctx.R, ctx.cfg
data, T, M = R["data"], ui.T, R["metrics"]
LOWER = {"log_loss", "brier", "ece"}
NAMES = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro F1", "weighted_f1": "Weighted F1", "log_loss": "Log loss", "brier": "Brier (multiclass)", "ece": "ECE"}
idx, sp = R["idx"], R["split"]
fmt = lambda d: d.style.format({k: "{:.3f}" for k in d.columns if k in NAMES.values()})
ui.note("Task: classify the regime at t+1 from features at t (6 classes). Models are fit on the training split; hyperparameter search uses walk-forward CV inside the training split; temperature calibration uses the validation split. The test split is never used for tuning.")

st.subheader("Model comparison")
split = st.radio("Split", ["test", "validation", "train"], horizontal=True, help="Test = held-out. Train results are in-sample.")
tab = M[M.split == split].drop(columns=["split", "n"]).rename(columns=NAMES).rename(columns={"model": "Model"})
st.dataframe(fmt(tab), hide_index=True, use_container_width=True)
ui.note("Persistence predicts tomorrow = today (probabilities smoothed to 98%). The HMM-based model maps filtered HMM states to next-regime probabilities. No single metric defines a 'winner'; compare columns together. Higher is better except log loss, Brier and ECE.")

st.subheader("Model inspection")
names = list(M.model.unique()); sel = st.selectbox("Model", names, index=names.index("Calibrated hybrid"))
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Hyperparameters / configuration**")
    if sel in ALGOS: st.json(R["params"][sel])
    elif "ensemble" in sel or sel == "Calibrated hybrid":
        st.json({"members": [m.name for m in R["models"][sel].members], "combination": "mean of member probabilities", "temperature": round(R["temperature"], 3) if sel == "Calibrated hybrid" else 1.0})
    elif sel == "HMM-based approach": st.json({"n_states": cfg["hmm"]["n_states"], "covariance_type": cfg["hmm"]["covariance_type"], "mapping": "filtered state probs @ P(next label | state)"})
    else: st.json({"rule": "predict next regime = current regime"})
    aug = sel in ("HMM-augmented ensemble", "Calibrated hybrid")
    st.caption(f"Feature set: {len(R['feature_cols']['aug' if aug else 'base'])} features" + (" (base + HMM filtered/predictive state probabilities)" if aug else "") + f". Train {len(idx['train'])} / validation {len(idx['validation'])} / test {len(idx['test'])} rows; embargo {R['embargo']} rows.")
with c2:
    st.markdown("**Performance by split**")
    st.dataframe(fmt(M[M.model == sel].drop(columns=["model", "n"]).rename(columns=NAMES).set_index("split")), use_container_width=True)

st.subheader("Confusion matrix")
c1, c2, c3 = st.columns(3)
cm_split, cm_model, norm = c1.selectbox("Split ", ["test", "validation", "train"], key="cms"), c2.selectbox("Model ", names, index=names.index(sel), key="cmm"), c3.toggle("Normalise (row %)")
ii = idx[cm_split]; yt, yp = R["y_next"][ii], R["proba"][cm_model][ii].argmax(1)
cm = confusion_matrix(yt, yp, labels=range(len(CLASSES))); z = cm / np.clip(cm.sum(1, keepdims=True), 1, None) if norm else cm
lbl = [ui.short(c) for c in CLASSES]
fig = go.Figure(go.Heatmap(z=z, x=lbl, y=lbl, text=np.round(z, 2) if norm else z, texttemplate="%{text}", colorscale="Teal", hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z}<extra></extra>"))
fig.update_xaxes(title="Predicted", side="bottom"); fig.update_yaxes(title="Actual", autorange="reversed"); ui.show(fig, 420, False)

st.subheader("Performance comparison")
c1, c2 = st.columns(2)
metric = c1.selectbox("Metric", list(NAMES), format_func=NAMES.get, index=2); bs = c2.selectbox("Split  ", ["test", "validation"], key="bs")
d = M[M.split == bs].sort_values(metric, ascending=metric in LOWER)
fig = go.Figure(go.Bar(x=d[metric], y=d.model, orientation="h", marker_color=T["accent"], hovertemplate="%{y}: %{x:.4f}<extra></extra>")); fig.update_yaxes(autorange="reversed"); ui.show(fig, 360, False)
st.caption(f"{NAMES[metric]}: {'lower' if metric in LOWER else 'higher'} is better. Differences on one split may be within noise.")

st.subheader("Ablation study")
A = R["ablation"]; asp = st.radio("Split   ", ["test", "validation"], horizontal=True, key="abl")
st.dataframe(A[A.split == asp].drop(columns="split").rename(columns={**NAMES, "configuration": "Configuration"}).style.format({v: "{:.3f}" for v in NAMES.values()}), hide_index=True, use_container_width=True)
ui.note("Only configurations that were actually trained and evaluated appear. The first three rows use a single LightGBM on cumulative feature groups; the last three reuse the ensembles above.")

st.subheader("Hyperparameter optimisation")
c1, c2, c3 = st.columns(3)
hm = c1.selectbox("Model to tune", [m for m in cfg["models"]["selected"] if m in GRIDS] or ALGOS)
size = int(np.prod([len(v) for v in GRIDS[hm].values()])); n_it = c2.slider("Random-search trials", 2, min(30, size), min(6, size))
c3.markdown(f"Walk-forward CV: **{cfg['split']['cv_splits']}** expanding folds<br>Objective: macro F1 (training split only)", unsafe_allow_html=True)
with st.expander("Search space"): st.json({k: list(map(float, v)) if isinstance(v[0], float) else v for k, v in GRIDS[hm].items()})
ok = st.checkbox(f"I understand this runs about {n_it * cfg['split']['cv_splits']} model fits and may take minutes")
log = st.session_state.setdefault("hpo_log", [])
if st.button("Run search", disabled=not ok):
    with st.spinner("Searching..."):
        hist = hyper_search(hm, R["Xb"].iloc[idx["train"]], R["y_next"][idx["train"]], R["feature_cols"]["base"], len(CLASSES), n_it, cfg["split"]["cv_splits"], cfg["split"]["seed"])
    best = hist.loc[hist.cv_macro_f1.idxmax()] if hist.cv_macro_f1.notna().any() else None
    log.append(dict(model=hm, history=hist, best=None if best is None else {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items() if k not in ("trial", "cv_macro_f1", "cv_std", "seconds")},
                    score=None if best is None else float(best.cv_macro_f1), status="complete"))
if log:
    last = log[-1]; st.markdown(f"**Latest run: {last['model']}** &middot; status {last['status']} &middot; best CV macro F1 {last['score']:.3f}" if last["score"] is not None else "No valid trials.")
    st.dataframe(last["history"], hide_index=True, use_container_width=True)
    if last["best"] and st.button("Apply best parameters and retrain"):
        ui.set_override({"models": {"params": {last["model"]: last["best"]}}}); st.rerun()
    st.caption(f"Experiment history: {len(log)} run(s) this session.")
else: st.caption("Status: no search run yet.")

st.subheader("Results export")
ex = st.selectbox("Model for classification report / predictions", names, index=names.index(sel), key="exm")
rep = pd.DataFrame(classification_report(R["y_next"][idx["test"]], R["proba"][ex][idx["test"]].argmax(1), labels=range(len(CLASSES)), target_names=CLASSES, output_dict=True, zero_division=0)).T
pred = pd.DataFrame(R["proba"][ex], index=data.index, columns=[f"P({c})" for c in CLASSES]).assign(split=sp.values, predicted_next_regime=[CLASSES[i] for i in R["proba"][ex].argmax(1)],
                                                                                              actual_next_regime=[CLASSES[i] if i >= 0 else None for i in R["y_next"]])
meta = dict(app_version="1.0.0", ticker=ctx.ticker, rows=len(data), first=str(data.index[0].date()), last=str(data.index[-1].date()), splits={k: [str(data.index[v[0]].date()), str(data.index[v[-1]].date()), int(len(v))] for k, v in idx.items()},
            embargo_rows=R["embargo"], config={k: v for k, v in cfg.items() if k != "data"}, label_thresholds=R["labeler"].describe(), temperature=R["temperature"], python=platform.python_version())
d = st.columns(5)
d[0].download_button("Model comparison", M.to_csv(index=False), "model_comparison.csv", "text/csv")
d[1].download_button("Classification report", rep.to_csv(), f"classification_report_{ex}.csv", "text/csv")
d[2].download_button("Predictions", pred.to_csv(), f"predictions_{ex}.csv", "text/csv")
d[3].download_button("Hyperparameter results", pd.concat([l["history"].assign(model=l["model"], run=i) for i, l in enumerate(log)]).to_csv(index=False) if log else "", "hyperparameter_results.csv", "text/csv", disabled=not log)
d[4].download_button("Experiment metadata", json.dumps(meta, indent=2, default=str), "experiment_metadata.json", "application/json")
