import json
import streamlit as st
from mp import config as C, ui
from mp.models import ALGOS

ctx = ui.get_ctx()
ui.header("Settings", "Configure features, models, calibration and evaluation. Changes apply only when you press Apply.", ctx)
cfg = ctx.cfg
st.info("Instrument, source, CSV upload, date range and display horizon live in the sidebar so they persist on every page. Applying settings below retrains the models; viewing and filtering never does.")


def parse(txt): return [int(x) for x in txt.replace(" ", "").split(",") if x]


with st.form("settings"):
    st.subheader("Feature engineering")
    a, b, c = st.columns(3)
    rw = a.number_input("Rolling return window", 2, 60, cfg["features"]["return_window"])
    vw = b.text_input("Volatility windows (comma-separated)", ", ".join(map(str, cfg["features"]["vol_windows"])))
    mw = c.text_input("Momentum windows (comma-separated)", ", ".join(map(str, cfg["features"]["mom_windows"])))
    a, b, c = st.columns(3)
    vol_on = a.checkbox("Volume features", cfg["features"]["use_volume"])
    adv_on = a.checkbox("Advanced features (range-vol estimators, skew/kurtosis, trend indicators)", cfg["features"].get("advanced", True))
    xa_on = a.checkbox("Cross-asset features (VIX, Brent) when available", cfg["features"].get("use_cross_asset", True))
    lw = b.number_input("Regime label window (days)", 10, 250, cfg["labels"]["window"])
    q = c.slider("Trend quantiles (bearish below / bullish above)", 0.05, 0.95, tuple(cfg["labels"]["trend_quantiles"]), 0.05)
    st.subheader("HMM")
    a, b, c = st.columns(3)
    k = a.slider("Number of states", 2, 6, cfg["hmm"]["n_states"])
    ct = b.selectbox("Covariance type", ["full", "diag", "tied", "spherical"], ["full", "diag", "tied", "spherical"].index(cfg["hmm"]["covariance_type"]))
    ni = c.number_input("EM iterations", 20, 1000, cfg["hmm"]["n_iter"])
    st.subheader("Machine learning")
    models = st.multiselect("Models in the ensembles", ALGOS, cfg["models"]["selected"])
    pj = st.text_area("Hyperparameter overrides (JSON, e.g. {\"XGBoost\": {\"max_depth\": 3}})", json.dumps(cfg["models"]["params"]) if cfg["models"]["params"] else "{}")
    st.subheader("Transition prediction and calibration")
    a, b, c, d = st.columns(4)
    hz = a.multiselect("Horizons (trading days)", [5, 10, 20], cfg["transition"]["horizons"])
    thr = b.slider("Classification threshold", 0.05, 0.95, float(cfg["transition"]["threshold"]), 0.01)
    cm = c.selectbox("Regime calibration", ["temperature", "none"], ["temperature", "none"].index(cfg["calibration"]["method"]))
    tm = d.selectbox("Transition calibration", ["auto", "isotonic", "platt", "none"], ["auto", "isotonic", "platt", "none"].index(cfg["calibration"]["transition_method"]))
    cth = st.slider("Confidence threshold", 0.2, 1.0, float(cfg["calibration"]["confidence_threshold"]), 0.05)
    st.subheader("Evaluation")
    a, b, c, d = st.columns(4)
    trn = a.slider("Train share", 0.4, 0.8, float(cfg["split"]["train"]), 0.05)
    val = b.slider("Validation share", 0.1, 0.3, float(cfg["split"]["val"]), 0.05)
    cv = c.number_input("Walk-forward CV folds", 2, 10, cfg["split"]["cv_splits"])
    seed = d.number_input("Random seed", 0, 10**6, cfg["split"]["seed"])
    emb = st.number_input("Embargo rows between splits (>= max horizon is enforced)", 0, 100, cfg["split"]["embargo"])
    go_ = st.form_submit_button("Apply and retrain", type="primary")

if go_:
    try:
        params = json.loads(pj or "{}"); assert isinstance(params, dict)
        new = dict(features=dict(return_window=int(rw), vol_windows=parse(vw), mom_windows=parse(mw), use_volume=vol_on, advanced=adv_on, use_cross_asset=xa_on), labels=dict(window=int(lw), trend_quantiles=list(q)),
                   hmm=dict(n_states=int(k), covariance_type=ct, n_iter=int(ni)), models=dict(selected=models, params=params),
                   transition=dict(horizons=sorted(hz), threshold=thr), calibration=dict(method=cm, transition_method=tm, confidence_threshold=cth),
                   split=dict(train=trn, val=val, test=round(1 - trn - val, 6), cv_splits=int(cv), seed=int(seed), embargo=int(emb)))
        errs = C.validate(C.deep_update(cfg, new))
    except Exception as exc:
        errs = [f"Could not parse inputs: {exc}"]; new = None
    if errs:
        for e in errs: st.error(e)
    else:
        ui.set_override(new); st.success("Settings applied. Retraining..."); st.rerun()

if st.button("Reset all settings to defaults"):
    st.session_state["cfg_over"] = {}; st.rerun()
with st.expander("Active configuration"): st.json({k: v for k, v in cfg.items()})
