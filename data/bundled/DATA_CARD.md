# Bundled dataset - `market_daily.csv`

| Item | Detail |
|---|---|
| Rows / span | 7,302 trading days, 1990-01-02 to 2018-12-21 |
| Columns | `date, open, high, low, close, adj_close, volume` (S&P 500 index), `vix` (CBOE VIX close), `brent` (Brent spot, USD/bbl) |
| S&P 500 source | Yahoo Finance ^GSPC daily history as redistributed in `vijinho/sp500` (GitHub, `csv/sp500.csv`, ends 2018-12-21). Licence of the redistribution is not stated; the file is included for coursework/reproducibility - re-download for any other use. |
| VIX source | CBOE VIX history via `datasets/finance-vix` (ODC-PDDL-1.0) |
| Brent source | U.S. EIA Europe Brent spot via `datasets/oil-prices` (ODC-PDDL-1.0) |
| Rebuild | `python scripts/build_bundled_data.py` (needs internet) |

## Cleaning applied
* Rows before 1990 dropped (VIX begins 1990). VIX/Brent aligned to S&P trading days by a left join; short holiday mismatches forward-filled (<= 3 rows at build, <= 5 in `mp.data.clean`). Rows without VIX are dropped (none).
* 78 early Brent observations are missing and stay missing (Brent features use a 1-day lag and are NaN-warmed, so those rows are excluded from modelling rather than imputed).
* `adj_close == close` for the index (no dividend adjustment). `volume` is index volume (reliable from 1990).
* Validation done at load (`mp.data.clean`): duplicate dates, non-positive prices, high < low, negative volume.

## Known limitations
* Ends in Dec-2018: the test window (2013-05 to 2018-12) therefore excludes the 2020 crash and 2022 bear market. Use the `yfinance` source or your own CSV to evaluate on later data.
* Data are index levels, not tradable prices; nothing here is investment advice.
