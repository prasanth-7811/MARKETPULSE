"""Data acquisition (yfinance / CSV), cleaning and quality reporting."""
from __future__ import annotations
import io
from datetime import date, timedelta
import numpy as np
import pandas as pd

from pathlib import Path

COLS = ["open", "high", "low", "close", "adj_close", "volume"]
AUX = ["vix", "brent"]                      # optional cross-asset columns (implied vol, crude oil)
BUNDLED = Path(__file__).resolve().parent.parent / "data" / "bundled" / "market_daily.csv"


class DataError(Exception):
    pass


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalise a raw OHLCV frame; return (clean_df, quality_report)."""
    df = raw.copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None: df.index = df.index.tz_localize(None)
    df.index.name = "date"
    for c in COLS:
        if c not in df: df[c] = np.nan
    aux_present = [c for c in AUX if c in df]
    df = df[COLS + aux_present].apply(pd.to_numeric, errors="coerce")
    q = {"rows_in": len(df), "missing": df[COLS].isna().sum().to_dict(), "aux_columns": aux_present,
         "duplicate_dates": int(df.index.duplicated().sum())}
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["adj_close"] = df["adj_close"].fillna(df["close"])
    for c in ("open", "high", "low"): df[c] = df[c].fillna(df["close"])
    bad = (df[["open", "high", "low", "close", "adj_close"]] <= 0).any(axis=1) | (df["high"] < df["low"]) | (df["volume"] < 0)
    q["invalid_rows"] = int(bad.sum())
    df = df[~bad].dropna(subset=["close", "adj_close"])
    df["volume"] = df["volume"].fillna(0)
    for c in aux_present:                    # cross-asset series: forward-fill short holiday mismatches only (<= 5 rows)
        q[f"missing_{c}"] = int(df[c].isna().sum()); df[c] = df[c].ffill(limit=5)
    q.update(rows_out=len(df), has_volume=bool((df["volume"] > 0).any()),
             gaps_over_5d=int((df.index.to_series().diff().dt.days > 5).sum()))
    if df.empty: raise DataError("No valid observations after cleaning.")
    return df, q


def load_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataError("yfinance is not installed.") from exc
    end_x = (pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        raw = yf.download(ticker, start=start, end=end_x, auto_adjust=False, progress=False)
    except Exception as exc:
        raise DataError(f"Download failed for {ticker}: {exc}") from exc
    if raw is None or raw.empty: raise DataError(f"No data returned for '{ticker}'. Check the symbol and network access.")
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    return raw


def load_csv(content: bytes) -> pd.DataFrame:
    try:
        raw = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise DataError(f"Could not parse CSV: {exc}") from exc
    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
    dcol = next((c for c in raw.columns if c in ("date", "datetime", "timestamp")), None)
    if dcol is None or "close" not in raw.columns:
        raise DataError("CSV needs at least 'Date' and 'Close' columns.")
    raw[dcol] = pd.to_datetime(raw[dcol], errors="coerce")
    return raw.dropna(subset=[dcol]).set_index(dcol)


def freshness(last: pd.Timestamp) -> str:
    age = (pd.Timestamp(date.today()) - last.normalize()).days
    if age <= 4: return "Current (daily close)"
    if age <= 30: return "Delayed"
    return "Historical"


def load_bundled() -> pd.DataFrame:
    """Bundled real daily dataset: S&P 500 OHLCV + VIX + Brent, 1990-2018 (see data/bundled/DATA_CARD.md)."""
    if not BUNDLED.exists(): raise DataError(f"Bundled dataset missing: {BUNDLED}. Run scripts/build_bundled_data.py")
    return pd.read_csv(BUNDLED, index_col=0, parse_dates=True)


def load_yf_aux(start: str, end: str) -> pd.DataFrame | None:
    """Best-effort VIX download to enrich a yfinance instrument; returns None (never fake data) on failure."""
    try:
        return load_yfinance("^VIX", start, end)["close"].rename("vix").to_frame()
    except DataError:
        return None
