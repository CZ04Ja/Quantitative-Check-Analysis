"""Quick diagnostic: how many aligned bars for each universe size,
each window? Identifies the inner-join data loss problem."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WINDOWS = {
    "BACK_OOS_2024":  ("2024-01-01", "2024-12-31"),
    "IS_2025_2026Q1": ("2025-01-01", "2026-03-31"),
    "FWD_OOS_2026Q2": ("2026-04-01", "2027-01-01"),
}


def _load(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["timestamp", "symbol", "close"]]


def main():
    parts = [
        _load(ROOT / "data_qtc" / "oos" / "back_oos_2024.parquet"),
        _load(ROOT / "data_qtc" / "combined" / "research_panel.parquet"),
        _load(ROOT / "data_qtc" / "oos" / "fwd_oos_2026q2.parquet"),
        _load(ROOT / "data_qtc" / "expanded" / "expanded_2024_2026.parquet"),
    ]
    full = pd.concat(parts, ignore_index=True)
    full = full.drop_duplicates(subset=["symbol", "timestamp"], keep="last")

    universes = {
        "4u":  ["AAPL", "AMD", "MSFT", "NVDA", "XLK"],
        "10u": ["AAPL", "AMD", "MSFT", "NVDA",
                "GOOGL", "META", "AMZN", "AVGO", "TSLA", "ORCL", "XLK"],
    }

    for label, (s, e) in WINDOWS.items():
        s_ts = pd.Timestamp(s, tz="UTC"); e_ts = pd.Timestamp(e, tz="UTC")
        sub = full[(full["timestamp"] >= s_ts) & (full["timestamp"] < e_ts)]
        print(f"\n--- {label} ---")
        for uname, symbols in universes.items():
            df_u = sub[sub["symbol"].isin(symbols)]
            wide = df_u.pivot_table(
                index="timestamp", columns="symbol", values="close"
            )
            wide = wide.reindex(columns=symbols)
            n_total = len(wide)
            n_full = wide.notna().all(axis=1).sum()
            present_per = wide.notna().sum().to_dict()
            print(f"  {uname}: total_idx={n_total:,}  inner_join={n_full:,}  "
                  f"({n_full/max(n_total,1)*100:.1f}% kept)")
            for s_, c_ in sorted(present_per.items()):
                print(f"     {s_}: {c_:,} bars")


if __name__ == "__main__":
    main()
