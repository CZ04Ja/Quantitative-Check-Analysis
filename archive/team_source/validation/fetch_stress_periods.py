"""Pull historical stress-test windows from Alpaca IEX feed.

Universe matches V3: AAPL, AMD, MSFT, NVDA, XLK.

Windows (chosen for known regime stress, NOT for performance):
    COVID_CRASH    : 2020-02-01 → 2020-05-31  (vol explosion + drawdown)
    POST_COVID     : 2020-06-01 → 2020-12-31  (sharp recovery)
    TECH_BEAR_2022 : 2022-01-01 → 2022-12-31  (tech -35% slow grind)
    BANKING_2023   : 2023-02-15 → 2023-04-30  (SVB, regional banks)
    TECH_2018Q4    : 2018-10-01 → 2019-01-31  (Q4 selloff + recovery)
    POST_BEAR_2023 : 2023-05-01 → 2023-12-31  (rebound year)

If any window fails (e.g. data not available), it is skipped.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "").strip()
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "").strip()
ALPACA_FEED = os.environ.get("ALPACA_FEED", "iex")

UNIVERSE = ["AAPL", "AMD", "MSFT", "NVDA", "XLK"]

WINDOWS = {
    "COVID_CRASH":    ("2020-02-01", "2020-05-31"),
    "POST_COVID":     ("2020-06-01", "2020-12-31"),
    "TECH_BEAR_2022": ("2022-01-01", "2022-12-31"),
    "BANKING_2023":   ("2023-02-15", "2023-04-30"),
    "TECH_2018Q4":    ("2018-10-01", "2019-01-31"),
    "POST_BEAR_2023": ("2023-05-01", "2023-12-31"),
}

OUT_DIR = ROOT / "data_qtc" / "stress"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pull(start: str, end: str, label: str) -> pd.DataFrame:
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    start_dt = pd.Timestamp(start, tz="UTC").to_pydatetime()
    end_dt = (pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).to_pydatetime()
    req = StockBarsRequest(
        symbol_or_symbols=UNIVERSE,
        timeframe=TimeFrame.Minute,
        start=start_dt, end=end_dt, feed=ALPACA_FEED,
    )
    print(f"[{label}] requesting {start} → {end}")
    try:
        resp = client.get_stock_bars(req)
    except Exception as exc:
        print(f"[{label}] FAILED: {exc}"); return pd.DataFrame()
    big = resp.df
    if big is None or big.empty:
        print(f"[{label}] EMPTY"); return pd.DataFrame()
    big = big.reset_index()
    big["ticker"] = big["symbol"]
    keep = ["timestamp", "symbol", "ticker", "open", "high", "low", "close", "volume"]
    big = big[keep].sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    print(f"[{label}] rows={len(big):,}  per-symbol={big['symbol'].value_counts().to_dict()}")
    return big


def main():
    for label, (s, e) in WINDOWS.items():
        df = pull(s, e, label)
        if df.empty:
            continue
        path = OUT_DIR / f"{label}.parquet"
        df.to_parquet(path)
        print(f"[{label}] saved -> {path}\n")


if __name__ == "__main__":
    main()
