"""
Pull additional names to expand the universe from 4 → 10.

Universe selection rule (defined BEFORE looking at any performance data,
to avoid post-hoc winner picking):

    "Top 10 US large-cap tech / consumer-tech names by market cap as of
     2024-01-02, all liquid enough for IEX 1-min coverage."

Existing 4: AAPL, AMD, MSFT, NVDA
New 6:      GOOGL, META, AMZN, AVGO, TSLA, ORCL

We deliberately did NOT pick CRM/NFLX/INTC/QCOM despite them being
market-cap-comparable in 2024-01: we wanted the 6 names that are most
clearly index-included so the universe is not retroactively chosen on
performance.

Time range: 2024-01-02 → today (single contiguous pull). For the 4
existing names we ALREADY have BACK_OOS_2024 + research_panel +
FWD_OOS_2026Q2 saved separately, so we only need the 6 new ones here.
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

NEW_SYMBOLS = ["GOOGL", "META", "AMZN", "AVGO", "TSLA", "ORCL"]

OUT_DIR = ROOT / "data_qtc" / "expanded"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pull(start: str, end: str, symbols: list[str], label: str) -> pd.DataFrame:
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    start_dt = pd.Timestamp(start, tz="UTC").to_pydatetime()
    end_dt = (pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).to_pydatetime()
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
        feed=ALPACA_FEED,
    )
    print(f"[{label}] requesting {start} → {end}, {len(symbols)} symbols")
    resp = client.get_stock_bars(req)
    big = resp.df
    if big is None or big.empty:
        print(f"[{label}] EMPTY"); return pd.DataFrame()
    big = big.reset_index()
    big["ticker"] = big["symbol"]
    keep = ["timestamp", "symbol", "ticker", "open", "high", "low", "close", "volume"]
    big = big[keep].sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    print(f"[{label}] rows={len(big):,}")
    print(f"[{label}] per-symbol: {big['symbol'].value_counts().to_dict()}")
    return big


def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df = pull("2024-01-02", today, NEW_SYMBOLS, "EXPANDED")
    if df.empty:
        return
    out_path = OUT_DIR / "expanded_2024_2026.parquet"
    df.to_parquet(out_path)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
