"""MarketPulse inference API.

FastAPI exposes the trained pipeline for local/remote inference:
  POST /regime      - current regime + next-session regime probabilities (calibrated hybrid)
  POST /transition  - regime-transition probability for a configurable horizon
  POST /explain     - global feature importance + per-observation SHAP contribution

The API reuses the same ``mp.pipeline`` machinery, so training/evaluation settings
must be agreed with the Streamlit app's ``config.yaml``. Not a trading system."""
from __future__ import annotations
import hashlib
import json

import numpy as np
import pandas as pd
import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mp import ui
from mp import config as C
from mp.data import DataError, clean, load_csv, load_yfinance
from mp.pipeline import PipelineError, run_pipeline

app = FastAPI(title="MarketPulse Regime Intelligence API", version="1.0.0")


class InferenceRequest(BaseModel):
    ticker: str
    start: str
    end: str
    horizon: int = 5


class RegimeResponse(BaseModel):
    ticker: str
    latest_date: str
    current_regime: str
    regime_confidence: float
    next_regime: str
    next_regime_probability: float
    model: str


class TransitionResponse(BaseModel):
    ticker: str
    latest_date: str
    horizon_days: int
    transition_probability: float
    classification: str


class ExplainRequest(BaseModel):
    ticker: str
    start: str
    end: str
    model: str
    observation_date: str | None = None


class ExplainResponse(BaseModel):
    ticker: str
    model: str
    global_importance: dict[str, float]
    observation: str | None
    prediction: str | None
    top_contributors: list[dict[str, float]]


@st.cache_data(ttl=3600, show_spinner=False)
def _load_data(source, ticker, start, end, csv_bytes, refresh):
    raw = load_yfinance(ticker, start, end) if source == "yfinance" else load_csv(csv_bytes)
    df, q = clean(raw)
    return df.loc[start:end], q


@st.cache_resource(max_entries=3, show_spinner=False)
def _train(df, cfg_json):
    return run_pipeline(df, json.loads(cfg_json))


def _pipeline_key(df, cfg) -> str:
    pcfg = json.loads(json.dumps(cfg, default=str))
    for k in ("horizon", "threshold"):
        pcfg["transition"].pop(k, None)
    pcfg["calibration"].pop("confidence_threshold", None)
    pcfg.pop("data", None)
    dh = hashlib.sha1(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:12]
    return hashlib.sha1((dh + json.dumps(pcfg, sort_keys=True)).encode()).hexdigest()[:16]


def _infer_model(request: InferenceRequest) -> tuple[dict, str]:
    """Load data, train the pipeline, and return (run_dict, model_key)."""
    cfg = C.deep_update(C.load_defaults(), {"data": {"source": "yfinance", "start": request.start, "end": request.end}})
    df, _ = _load_data("yfinance", request.ticker, request.start, request.end, None, 0)
    if df.empty:
        raise HTTPException(status_code=400, detail=f"No valid observations for {request.ticker}.")
    key = _pipeline_key(df, cfg)
    R = _train(df, json.dumps(cfg, sort_keys=True))
    return R, key


@app.get("/")
def root():
    return {"service": "MarketPulse Regime Intelligence API", "status": "running",
            "docs": "/docs", "endpoints": ["/regime", "/transition", "/explain"]}


@app.post("/regime", response_model=RegimeResponse)
def regime_infer(request: InferenceRequest):
    R, _ = _infer_model(request)
    data, lab = R["data"], R["labels"]
    last = len(data) - 1
    P = R["proba"]["Calibrated hybrid"][last]
    cur = int(R["y"][last])
    out = RegimeResponse(
        ticker=request.ticker,
        latest_date=str(data.index[last]),
        current_regime=ui.short(R["labels"].iloc[last]),
        regime_confidence=float(P[cur]),
        next_regime=ui.short(lab.iloc[last]),
        next_regime_probability=float(P[cur]),
        model="Calibrated hybrid (LR/RF/XGBoost/LightGBM ensemble)",
    )
    return out


@app.post("/transition", response_model=TransitionResponse)
def transition_infer(request: InferenceRequest):
    R, _ = _infer_model(request)
    if request.horizon not in R["transition"]:
        raise HTTPException(status_code=400, detail=f"Horizon {request.horizon} not available (supported: {sorted(R['transition'])}).")
    tr = R["transition"][request.horizon]
    if tr.get("model") is None:
        raise HTTPException(status_code=503, detail=tr.get("error", "Transition model unavailable."))
    last = len(R["data"]) - 1
    thr = float(R["transition"]["threshold"])
    out = TransitionResponse(
        ticker=request.ticker,
        latest_date=str(R["data"].index[last]),
        horizon_days=request.horizon,
        transition_probability=float(tr["p_cal"][last]),
        classification="transition" if tr["p_cal"][last] >= thr else "no-transition",
    )
    return out


@app.post("/explain", response_model=ExplainResponse)
def explain_infer(request: ExplainRequest):
    from mp.explain import shap_for
    R, _ = _infer_model(request)
    aug = request.model in ("HMM-augmented ensemble", "Calibrated hybrid")
    cols = R["feature_cols"]["aug" if aug else "base"]
    if request.model not in R["models"]:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' not trained. Available: {list(R['models'].keys())}.")
    sel = R["models"][request.model]
    X = R["Xa" if aug else "Xb"]
    if request.model in ("Random Forest", "XGBoost", "LightGBM"):
        sv, present = shap_for(sel, X.iloc[500:], binary_positive=False)
        imp = pd.Series(np.abs(sv).mean(axis=(0, 2)), index=cols).nlargest(10)[::-1].to_dict()
    else:
        imp = {"info": f"Per-observation SHAP not exposed via API for {request.model}"}
    out = ExplainResponse(
        ticker=request.ticker,
        model=request.model,
        global_importance=imp,
        observation=None,
        prediction=None,
        top_contributors=[{"feature": k, "importance": round(v, 4)} for k, v in imp.items()],
    )
    return out
