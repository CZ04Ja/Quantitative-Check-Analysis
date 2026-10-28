"""Pull pristine out-of-sample minute bars from Alpaca (IEX feed).

Two windows are OOS w.r.t. the original research panel
(`data_qtc/combined/research_panel.parquet`, 2025-01-02 → 2026-03-31):

    BACK_OOS : 2024-01-02 → 2024-12-31  (12 months, pre-research)
    FWD_OOS  : 2026-04-01 → today       (post-research; today is 2026-05-04)

For each window we save a long-format parquet with the minimum columns the
strategy needs:  timestamp, symbol, open, high, low, close, volume.

This script does NOT touch the original research_panel; nothing here
overwrites design data. We only ADD pristine windows.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
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

UNIVERSE = ["AAPL", "AMD", "MSFT", "NVDA", "XLK", "QQQ"]

OUT_DIR = ROOT / "data_qtc" / "oos"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pull(start: str, end: str, label: str) -> pd.DataFrame:
    """Single Alpaca multi-symbol minute-bar request."""
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    start_dt = pd.Timestamp(start, tz="UTC").to_pydatetime()
    end_dt = (pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).to_pydatetime()
    req = StockBarsRequest(
        symbol_or_symbols=UNIVERSE,
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
        feed=ALPACA_FEED,
    )
    print(f"[{label}] requesting {start} → {end}, {len(UNIVERSE)} symbols, feed={ALPACA_FEED}")
    resp = client.get_stock_bars(req)
    big = resp.df
    if big is None or big.empty:
        print(f"[{label}] EMPTY")
        return pd.DataFrame()
    big = big.reset_index()
    big = big.rename(columns={"symbol": "symbol", "timestamp": "timestamp"})
    big["ticker"] = big["symbol"]
    keep = ["timestamp", "symbol", "ticker", "open", "high", "low", "close", "volume"]
    big = big[keep].sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    print(f"[{label}] rows={len(big):,} per-symbol={big['symbol'].value_counts().to_dict()}")
    return big


def main() -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    back = pull("2024-01-02", "2024-12-31", "BACK_OOS")
    if not back.empty:
        path = OUT_DIR / "back_oos_2024.parquet"
        back.to_parquet(path)
        print(f"[BACK_OOS] saved -> {path}")

    fwd = pull("2026-04-01", today, "FWD_OOS")
    if not fwd.empty:
        path = OUT_DIR / "fwd_oos_2026q2.parquet"
        fwd.to_parquet(path)
        print(f"[FWD_OOS] saved -> {path}")


if __name__ == "__main__":
    main()
