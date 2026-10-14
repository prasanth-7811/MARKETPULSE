"""Design system, shared components, sidebar and application context."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from mp import config as C
from mp.data import DataError, clean, freshness, load_bundled, load_csv, load_yf_aux, load_yfinance
from mp.pipeline import PipelineError, run_pipeline

T = dict(bg="#0B1020", side="#0F172A", card="#151D2E", card2="#1B2638", accent="#38BDF8", accent2="#2DD4BF",
         pos="#22C55E", neg="#F87171", warn="#F59E0B", text="#F1F5F9", muted="#94A3B8", border="#293548", mute_bg="#111827")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{{--bg:{T['bg']};--side:{T['side']};--card:{T['card']};--card2:{T['card2']};--accent:{T['accent']};--accent2:{T['accent2']};
--pos:{T['pos']};--neg:{T['neg']};--warn:{T['warn']};--text:{T['text']};--muted:{T['muted']};--border:{T['border']};--mute-bg:{T['mute_bg']};
--r:12px;--s1:8px;--s2:16px;--s3:24px;--s4:32px}}
html,body,.stApp,[class*="css"]{{font-family:Inter,system-ui,-apple-system,'Segoe UI',sans-serif;color:var(--text)}}
.stApp{{background:var(--bg)}} #MainMenu,footer{{visibility:hidden}}
[data-testid="stHeader"]{{background:transparent}}
[data-testid="stSidebar"]{{background:var(--side);border-right:1px solid var(--border)}}
[data-testid="stSidebarNav"] a[aria-current="page"], [data-testid="stSidebarNav"] li a[data-active="true"]{{background:rgba(56,189,248,.12);border-left:2px solid var(--accent)}}
.block-container{{padding-top:var(--s3);max-width:1400px}}
h1{{font-size:30px!important;font-weight:700!important;margin:0!important}} h2,h3{{font-size:19px!important;font-weight:600!important}}
p,li,label,.stMarkdown{{font-size:14px}}
.mp-card{{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:var(--s2);margin-bottom:var(--s2)}}
.mp-kpi{{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:14px var(--s2);min-height:128px}}
.mp-kpi .l{{font-size:12px;color:var(--muted);letter-spacing:.06em;text-transform:uppercase;font-weight:600}}
.mp-kpi .v{{font-size:26px;font-weight:700;margin:6px 0 4px;line-height:1.15}}
.mp-kpi .s{{font-size:12px;color:var(--muted)}}
.mp-head{{display:flex;justify-content:space-between;align-items:flex-start;gap:var(--s2);margin-bottom:var(--s3);border-bottom:1px solid var(--border);padding-bottom:var(--s2);flex-wrap:wrap}}
.mp-head .crumb{{font-size:11px;color:var(--muted);margin-bottom:6px}} .mp-head .d{{color:var(--muted);font-size:14px;margin-top:6px}}
.mp-meta{{text-align:right;font-size:12px;color:var(--muted);line-height:1.7}} .mp-meta b{{color:var(--text);font-weight:600}}
.mp-pill{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:600;border:1px solid var(--border)}}
.mp-note{{font-size:12px;color:var(--muted);border-left:2px solid var(--accent);padding-left:10px;margin:8px 0 16px}}
.mp-warn{{font-size:13px;color:var(--warn);border:1px solid var(--warn);border-radius:8px;padding:8px 12px;margin:8px 0;background:rgba(245,158,11,.07)}}
.dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}}
div[data-testid="stMetric"]{{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:12px}}
.stTabs [data-baseweb="tab"]{{font-size:13px}}
</style>"""


def style(fig: go.Figure, h: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      font=dict(family="Inter, sans-serif", color=T["muted"], size=12), margin=dict(l=8, r=8, t=32, b=8),
                      legend=dict(orientation="h", y=1.12, x=0), hoverlabel=dict(bgcolor=T["card2"]),
                      colorway=[T["accent"], T["accent2"], T["warn"], T["neg"], T["pos"]])
    fig.update_xaxes(gridcolor="#1E293B", zerolinecolor=T["border"], linecolor=T["border"])
    fig.update_yaxes(gridcolor="#1E293B", zerolinecolor=T["border"], linecolor=T["border"])
    return fig


def show(fig, h=360, legend=True): st.plotly_chart(style(fig, h, legend), use_container_width=True)
def pct(x, d=1): return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"
def note(txt): st.markdown(f"<div class='mp-note'>{txt}</div>", unsafe_allow_html=True)
def warn(txt): st.markdown(f"<div class='mp-warn'>{txt}</div>", unsafe_allow_html=True)


def kpi(col, label, value, sub="", color=None):
    col.markdown(f"<div class='mp-kpi'><div class='l'>{label}</div><div class='v' style='color:{color or T['text']}'>{value}</div><div class='s'>{sub}</div></div>", unsafe_allow_html=True)


@dataclass
class Ctx:
    cfg: dict
    status: str = "loading"
    error: str = ""
    df: pd.DataFrame | None = None
    quality: dict = field(default_factory=dict)
    R: dict | None = None
    key: str = ""
    ticker: str = ""
    source: str = ""
    last: pd.Timestamp | None = None
    fresh: str = "Unavailable"


STATUS = {"ready": (T["pos"], "Ready"), "error": (T["neg"], "Error"), "loading": (T["warn"], "Loading"),
          "training": (T["warn"], "Training models")}


def header(title: str, desc: str, ctx: Ctx):
    col, label = STATUS.get(ctx.status, STATUS["error"])
    last = ctx.last.strftime("%Y-%m-%d") if ctx.last is not None else "n/a"
    st.markdown(f"""<div class='mp-head'><div><div class='crumb'>MarketPulse / Financial Market Regime Intelligence</div>
<h1>{title}</h1><div class='d'>{desc}</div></div>
<div class='mp-meta'>Selected Instrument: <b>{ctx.ticker or 'n/a'}</b><br>Latest Observation: <b>{last}</b><br>
Data Status: <b>{ctx.fresh}</b><br><span class='dot' style='background:{col}'></span>System: <b>{label}</b></div></div>""", unsafe_allow_html=True)
    if ctx.status == "error": st.error(ctx.error or "Application error."); st.stop()


def get_ctx() -> Ctx: return st.session_state["ctx"]


def effective_cfg() -> dict:
    cfg = C.deep_update(C.load_defaults(), st.session_state.get("cfg_over", {}))
    cfg["data"]["start"] = str(st.session_state.get("start", cfg["data"]["start"]))
    cfg["data"]["end"] = str(st.session_state.get("end", date.today()))
    cfg["transition"]["horizon"] = int(st.session_state.get("horizon", cfg["transition"]["horizon"]))
    return cfg


def set_override(path: dict):
    st.session_state["cfg_over"] = C.deep_update(st.session_state.get("cfg_over", {}), path)


def sidebar() -> None:
    d = C.load_defaults()
    sb = st.sidebar
    sb.markdown("<div style='font-size:11px;color:#94A3B8;margin:8px 0 4px'>DATA</div>", unsafe_allow_html=True)
    src = sb.radio("Source", ["Bundled", "yfinance", "CSV upload"], key="src", horizontal=True, label_visibility="collapsed",
                   help="Bundled = real S&P 500 + VIX + Brent daily data, 1990-2018 (works offline). yfinance needs internet.")
    if src == "Bundled":
        sb.caption("S&P 500 + VIX + Brent, 1990-12/2018 (see data/bundled/DATA_CARD.md)")
    elif src == "yfinance":
        pre = sb.selectbox("Instrument", d["data"]["presets"] + ["Custom..."], key="preset")
        if pre == "Custom...": sb.text_input("Ticker symbol", "SPY", key="custom_t")
    else:
        sb.file_uploader("OHLCV CSV", type="csv", key="csv")
        sb.text_input("Instrument label", "CSV", key="csv_name")
    sb.date_input("Start date", date.fromisoformat(d["data"]["start"]), key="start", min_value=date(1990, 1, 1))
    sb.date_input("End date", date.today(), key="end", max_value=date.today())
    sb.selectbox("Transition horizon (trading days)", d["transition"]["horizons"], key="horizon")
    if sb.button("Refresh market data", use_container_width=True):
        st.session_state["refresh"] = st.session_state.get("refresh", 0) + 1
    ctx = st.session_state.get("ctx")
    c, lab = STATUS.get(ctx.status if ctx else "loading")
    sb.markdown(f"<div style='font-size:12px;color:#94A3B8;margin-top:16px'><span class='dot' style='background:{c}'></span>{lab} &middot; Source: {src}<br>"
                f"v{d['app']['version']}<br><span style='font-size:11px'>Educational research tool. Not financial advice; not a forecast or trading system. Daily data, not real-time.</span></div>", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def _load(source, ticker, start, end, csv_bytes, refresh):
    if source == "Bundled": raw = load_bundled()
    elif source == "yfinance":
        raw = load_yfinance(ticker, start, end); aux = load_yf_aux(start, end)
        if aux is not None: raw = raw.join(aux, how="left")
    else: raw = load_csv(csv_bytes)
    df, q = clean(raw)
    return df.loc[start:end], q


@st.cache_resource(max_entries=3, show_spinner=False)
def _train(key: str, _df, cfg_json: str):
    return run_pipeline(_df, json.loads(cfg_json))


def _inputs():
    ss = st.session_state
    src = ss.get("src", "Bundled")
    if src == "Bundled": return src, "S&P 500 (bundled)", None
    if src == "yfinance":
        pre = ss.get("preset", "SPY")
        return src, (ss.get("custom_t", "SPY").strip().upper() if pre == "Custom..." else pre), None
    up = ss.get("csv")
    return src, ss.get("csv_name", "CSV"), (up.getvalue() if up else None)


def bootstrap() -> Ctx:
    cfg = effective_cfg()
    src, ticker, csv_bytes = _inputs()
    ctx = Ctx(cfg=cfg, ticker=ticker, source=src)
    st.session_state["ctx"] = ctx
    errs = C.validate(cfg)
    if errs:
        ctx.status, ctx.error = "error", "Invalid configuration: " + " ".join(errs); return ctx
    try:
        if ctx.source == "CSV upload" and not csv_bytes:
            raise DataError("Upload an OHLCV CSV in the sidebar (columns: Date, Open, High, Low, Close, Adj Close, Volume).")
        with st.spinner("Loading market data..."):
            ctx.df, ctx.quality = _load(ctx.source, ctx.ticker, cfg["data"]["start"], cfg["data"]["end"],
                                        csv_bytes, st.session_state.get("refresh", 0))
        if ctx.df.empty: raise DataError("No observations in the selected date range.")
        ctx.last, ctx.fresh = ctx.df.index[-1], freshness(ctx.df.index[-1])
        pcfg = json.loads(json.dumps(cfg, default=str))
        for k in ("horizon", "threshold"): pcfg["transition"].pop(k, None)
        pcfg["calibration"].pop("confidence_threshold", None); pcfg.pop("data", None)
        dh = hashlib.sha1(pd.util.hash_pandas_object(ctx.df).values.tobytes()).hexdigest()[:12]
        ctx.key = hashlib.sha1((dh + json.dumps(pcfg, sort_keys=True)).encode()).hexdigest()[:16]
        ctx.status = "training"
        with st.spinner("Training HMM, classifiers and transition models (cached until data or model settings change)..."):
            ctx.R = _train(ctx.key, ctx.df, json.dumps(pcfg, sort_keys=True))
        ctx.status = "ready"
    except (DataError, PipelineError) as exc:
        ctx.status, ctx.error = "error", str(exc)
    except Exception as exc:  # surfaced, never hidden behind a green status
        ctx.status, ctx.error = "error", f"{type(exc).__name__}: {exc}"
    return ctx


def cur_h(ctx: Ctx) -> int:
    h, avail = ctx.cfg["transition"]["horizon"], sorted(ctx.R["transition"])
    return h if h in avail else avail[0]


def short(c: str) -> str: return c.replace(" Volatility", "").replace(" / ", "/")


def regime_traces(fig, x, y, labels, colors, name_fn=None, size=4, extra=None):
    """One marker trace per regime on top of a thin neutral line; hover shows regime and extras."""
    fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", line=dict(color="#475569", width=1), hoverinfo="skip", showlegend=False))
    lab = pd.Series(np.asarray(labels), index=x)
    for name, col in colors.items():
        m = (lab == name).to_numpy()
        if not m.any(): continue
        cd = None if extra is None else np.asarray(extra)[m]
        fig.add_trace(go.Scattergl(x=np.asarray(x)[m], y=np.asarray(y)[m], mode="markers", name=short(name), marker=dict(color=col, size=size),
                                   customdata=cd, hovertemplate=f"%{{x|%Y-%m-%d}}<br>Price %{{y:.2f}}<br>{short(name)}" + ("<br>Filtered P %{customdata:.1%}" if extra is not None else "") + "<extra></extra>"))
    fig.update_xaxes(rangeselector=dict(buttons=[dict(count=6, label="6M", step="month", stepmode="backward"), dict(count=1, label="1Y", step="year", stepmode="backward"),
                                                  dict(count=5, label="5Y", step="year", stepmode="backward"), dict(step="all", label="All")], bgcolor=T["card2"], font=dict(color=T["text"])))
    return fig
