"""Pull SPY (best liquidity proxy) for the 2024 and 2026Q2 windows."""
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


def pull(start: str, end: str, label: str) -> pd.DataFrame:
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    start_dt = pd.Timestamp(start, tz="UTC").to_pydatetime()
    end_dt = (pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).to_pydatetime()
    req = StockBarsRequest(
        symbol_or_symbols=["SPY"],
        timeframe=TimeFrame.Minute,
        start=start_dt, end=end_dt, feed=ALPACA_FEED,
    )
    print(f"[{label}] requesting SPY {start} → {end}")
    resp = client.get_stock_bars(req)
    big = resp.df
    if big is None or big.empty:
        return pd.DataFrame()
    big = big.reset_index()
    big["ticker"] = big["symbol"]
    keep = ["timestamp", "symbol", "ticker", "open", "high", "low", "close", "volume"]
    big = big[keep].sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    print(f"[{label}] rows={len(big):,}")
    return big


def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df_2024 = pull("2024-01-02", "2024-12-31", "SPY 2024")
    df_q2 = pull("2026-04-01", today, "SPY 2026Q2")
    out_path = ROOT / "data_qtc" / "expanded" / "spy_2024_2026.parquet"
    pd.concat([df_2024, df_q2]).to_parquet(out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
