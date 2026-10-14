"""Rebuild data/bundled/market_daily.csv from public GitHub-hosted sources (provenance in data/bundled/DATA_CARD.md).

Usage:  python scripts/build_bundled_data.py
Columns: date, open, high, low, close, adj_close, volume (S&P 500) + vix (CBOE VIX close) + brent (Brent spot, USD/bbl).
Window: 1990-01-02 .. 2018-12-21 (S&P 500 source ends there; VIX starts 1990).
"""
import io, sys, urllib.request
from pathlib import Path
import pandas as pd

SRC = {
    "gspc": "https://raw.githubusercontent.com/vijinho/sp500/master/csv/sp500.csv",
    "vix": "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv",
    "brent": "https://raw.githubusercontent.com/datasets/oil-prices/main/data/brent-daily.csv",
}
OUT = Path(__file__).resolve().parent.parent / "data" / "bundled" / "market_daily.csv"


def get(url: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(urllib.request.urlopen(url, timeout=60).read()))


def main() -> None:
    g = get(SRC["gspc"]); g["Date"] = pd.to_datetime(g["Date"]); g = g.set_index("Date").drop(columns=["id"])
    g.columns = ["open", "high", "low", "close", "adj_close", "volume"]
    v = get(SRC["vix"]); v["DATE"] = pd.to_datetime(v["DATE"]); v = v.set_index("DATE")["CLOSE"].rename("vix")
    b = get(SRC["brent"]); b["Date"] = pd.to_datetime(b["Date"]); b = b.set_index("Date")["Price"].rename("brent")
    df = g.loc["1990-01-01":].join(v, how="left").join(b, how="left")
    df[["vix", "brent"]] = df[["vix", "brent"]].ffill(limit=3)      # align holiday mismatches only (<=3 days)
    df = df.dropna(subset=["vix"]); df.index.name = "date"
    OUT.parent.mkdir(parents=True, exist_ok=True); df.to_csv(OUT, float_format="%.6g")
    print(f"wrote {OUT}  rows={len(df)}  {df.index[0].date()}..{df.index[-1].date()}  missing=\n{df.isna().sum().to_string()}")


if __name__ == "__main__":
    sys.exit(main())
