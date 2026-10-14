"""MarketPulse inference API (FastAPI).

Run locally:   uvicorn api.main:app --reload --port 8000        (docs at /docs, demo page at /)
Container:     see Dockerfile (listens on $PORT, default 7860 for Hugging Face Spaces)
"""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from datetime import date as Date
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator
from mp.data import clean, load_bundled
from mp.inference import DISCLAIMER, MIN_ROWS, InferenceError, Predictor

BUNDLE = Path(os.environ.get("MP_BUNDLE", Path(__file__).resolve().parent.parent / "artifacts" / "model_bundle.joblib"))
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try: STATE["predictor"] = Predictor.load(BUNDLE)
    except FileNotFoundError as exc: STATE["error"] = str(exc)
    yield
    STATE.clear()


app = FastAPI(title="MarketPulse Regime API", version="2.0.0", lifespan=lifespan,
              description="Market-regime classification and regime-change probabilities from daily OHLCV history. " + DISCLAIMER)


class Bar(BaseModel):
    date: Date
    open: float = Field(gt=0); high: float = Field(gt=0); low: float = Field(gt=0); close: float = Field(gt=0)
    adj_close: float | None = Field(default=None, gt=0)
    volume: float = Field(default=0, ge=0)
    vix: float | None = Field(default=None, gt=0)
    brent: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _ohlc(self):
        if self.high < self.low: raise ValueError("high must be >= low")
        if not (self.low - 1e-9 <= self.close <= self.high + 1e-9): raise ValueError("close must lie within [low, high]")
        return self


class PredictRequest(BaseModel):
    rows: list[Bar] = Field(min_length=MIN_ROWS, description=f"Daily bars, ascending by date, at least {MIN_ROWS}. Include `vix` if the model requires it (see /model).")


def _predictor() -> Predictor:
    if "predictor" not in STATE: raise HTTPException(503, STATE.get("error", "Model not loaded."))
    return STATE["predictor"]


def _run(pred: Predictor, frame: pd.DataFrame) -> dict:
    try: return pred.predict(frame)
    except InferenceError as exc: raise HTTPException(422, str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if "predictor" in STATE else "model_missing", "detail": STATE.get("error")}


@app.get("/model")
def model_card() -> dict:
    return _predictor().info()


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    frame = pd.DataFrame([r.model_dump() for r in req.rows]).set_index("date")
    frame.index = pd.to_datetime(frame.index)
    if not frame.index.is_monotonic_increasing: raise HTTPException(422, "rows must be sorted ascending by date")
    if frame.index.has_duplicates: raise HTTPException(422, "duplicate dates in rows")
    return _run(_predictor(), frame)


@app.get("/demo")
def demo(as_of: Date | None = None) -> dict:
    """Prediction on the bundled historical S&P 500 + VIX + Brent data (ends 2018-12-21). `as_of` picks any earlier date."""
    raw = load_bundled()
    if as_of is not None: raw = raw.loc[: pd.Timestamp(as_of)]
    return _run(_predictor(), raw.tail(1500))


PAGE = """<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>MarketPulse API</title><style>
body{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:32px auto;padding:0 16px;background:#0B1020;color:#F1F5F9}
a{color:#38BDF8}input,button{font:inherit;padding:8px 12px;border-radius:8px;border:1px solid #293548;background:#151D2E;color:#F1F5F9}
button{cursor:pointer;background:#38BDF8;color:#0B1020;border:0;font-weight:600}pre{background:#151D2E;border:1px solid #293548;border-radius:12px;padding:16px;overflow:auto}
.bar{height:10px;background:#38BDF8;border-radius:5px}.row{display:grid;grid-template-columns:230px 1fr 60px;gap:8px;align-items:center;margin:4px 0}small{color:#94A3B8}</style>
<h1>MarketPulse regime API</h1><p>Classifies the market regime and estimates regime-change probabilities from daily history. <small>Educational tool - not financial advice.</small></p>
<p>Historical demo (bundled S&amp;P 500 data, 1990-2018): as-of date <input id=d type=date value=2018-12-21 min=1995-01-01 max=2018-12-21> <button onclick=run()>Predict</button>
&nbsp; <a href=/docs>API docs</a> &middot; <a href=/model>model card</a></p><div id=out></div><pre id=raw></pre>
<script>
async function run(){const d=document.getElementById('d').value;const r=await fetch('/demo?as_of='+d);const j=await r.json();
document.getElementById('raw').textContent=JSON.stringify(j,null,2);if(!r.ok){document.getElementById('out').innerHTML='<p>'+(j.detail||'error')+'</p>';return}
let h='<h3>'+j.as_of+': current regime <b>'+j.current_regime+'</b> ('+j.days_in_current_regime+' days)</h3><p>Next-day probabilities'+(j.low_confidence?' <small>(low confidence)</small>':'')+'</p>';
for(const [k,v] of Object.entries(j.next_day_regime_probabilities))h+='<div class=row><span>'+k+'</span><div><div class=bar style="width:'+(v*100)+'%"></div></div><span>'+(v*100).toFixed(1)+'%</span></div>';
h+='<p>Regime-change probability: '+Object.entries(j.regime_change_probability).map(([k,v])=>k+'d: '+(v*100).toFixed(0)+'%').join(' &middot; ')+'</p>';document.getElementById('out').innerHTML=h}
run()</script></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str: return PAGE
