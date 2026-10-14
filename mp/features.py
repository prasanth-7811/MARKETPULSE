"""Causal feature engineering. Every feature at row t uses data up to and including t only."""
from __future__ import annotations
import numpy as np
import pandas as pd

ANN = 252


def rsi(p: pd.Series, w: int) -> pd.Series:
    d = p.diff()
    up, dn = d.clip(lower=0).rolling(w).mean(), (-d.clip(upper=0)).rolling(w).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_features(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    f = cfg["features"]
    p, r = df["adj_close"], np.log(df["adj_close"]).diff()
    o = pd.DataFrame(index=df.index)
    g: dict[str, list[str]] = {"returns": [], "volatility": [], "momentum": [], "volume": [], "advanced_vol": [], "trend": [], "cross_asset": []}

    def add(name, s, grp):
        o[name] = s; g[grp].append(name)

    add("ret", r, "returns"); add("ret_lag1", r.shift(1), "returns"); add("ret_lag2", r.shift(2), "returns")
    add(f"ret_mean_{f['return_window']}", r.rolling(f["return_window"]).mean(), "returns")
    add("hl_range", (df["high"] - df["low"]) / df["close"], "returns")
    vw = sorted(f["vol_windows"])
    for w in vw: add(f"vol_{w}", r.rolling(w).std() * np.sqrt(ANN), "volatility")
    if len(vw) > 1: add("vol_ratio", o[f"vol_{vw[0]}"] / o[f"vol_{vw[-1]}"], "volatility")
    for w in sorted(f["mom_windows"]): add(f"mom_{w}", p.pct_change(w), "momentum")
    add("rsi", rsi(p, f["rsi_window"]), "momentum")
    add("drawdown", p / p.cummax() - 1, "momentum")
    if f.get("use_volume") and (df["volume"] > 0).any():
        lv = np.log1p(df["volume"])
        add("volume_z", (lv - lv.rolling(60).mean()) / lv.rolling(60).std(), "volume")
    if f.get("advanced", True): _advanced(df, p, r, o, add, f)
    if f.get("use_cross_asset", True): _cross_asset(df, r, o, add, f)
    o = o.replace([np.inf, -np.inf], np.nan)
    return o, {k: v for k, v in g.items() if v}


def _advanced(df, p, r, o, add, f) -> None:
    """Range-based volatility estimators, higher moments, trend/oscillator indicators (all causal)."""
    w = mid(f["vol_windows"])
    hl, co = np.log(df["high"] / df["low"]), np.log(df["close"] / df["open"])
    add(f"parkinson_{w}", np.sqrt((hl ** 2).rolling(w).mean() / (4 * np.log(2))) * np.sqrt(ANN), "advanced_vol")          # Parkinson (1980)
    add(f"garman_klass_{w}", np.sqrt((0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2).rolling(w).mean().clip(lower=0)) * np.sqrt(ANN), "advanced_vol")  # Garman & Klass (1980)
    add("ewma_vol", np.sqrt(r.pow(2).ewm(alpha=0.06, adjust=False).mean()) * np.sqrt(ANN), "advanced_vol")             # RiskMetrics lambda=0.94
    add("vol_of_vol", o[f"vol_{w}"].rolling(60).std(), "advanced_vol")
    add(f"skew_{max(60, w)}", r.rolling(max(60, w)).skew(), "advanced_vol")
    add(f"kurt_{max(60, w)}", r.rolling(max(60, w)).kurt(), "advanced_vol")
    add("sma_gap_50", p / p.rolling(50).mean() - 1, "trend")
    add("sma_gap_200", p / p.rolling(200).mean() - 1, "trend")
    macd = p.ewm(span=12, adjust=False).mean() - p.ewm(span=26, adjust=False).mean()
    add("macd_hist", (macd - macd.ewm(span=9, adjust=False).mean()) / p, "trend")
    m20, s20 = p.rolling(20).mean(), p.rolling(20).std()
    add("boll_pctb", (p - (m20 - 2 * s20)) / (4 * s20), "trend")
    add("ret_autocorr_60", r.rolling(60).corr(r.shift(1)), "trend")
    add("up_day_ratio_20", (r > 0).astype(float).rolling(20).mean(), "trend")


def _cross_asset(df, r, o, add, f) -> None:
    """Implied-volatility and commodity covariates when present in the data (VIX, Brent)."""
    w = mid(f["vol_windows"])
    if "vix" in df and df["vix"].notna().sum() > 250:
        v = df["vix"]
        add("vix", v, "cross_asset")
        add("vix_chg_5", np.log(v).diff(5), "cross_asset")
        add("vix_z_60", (np.log(v) - np.log(v).rolling(60).mean()) / np.log(v).rolling(60).std(), "cross_asset")
        add("vrp", v / 100 - o[f"vol_{w}"], "cross_asset")        # implied minus realised volatility (variance-risk-premium proxy)
    if "brent" in df and df["brent"].notna().sum() > 250:
        b = np.log(df["brent"].where(df["brent"] > 0)).shift(1)   # lagged 1 day: spot series is published with a delay
        add("brent_ret_20", b.diff(20), "cross_asset")
        add("brent_vol_20", b.diff().rolling(20).std() * np.sqrt(ANN), "cross_asset")


def mid(ws): ws = sorted(ws); return ws[len(ws) // 2]
