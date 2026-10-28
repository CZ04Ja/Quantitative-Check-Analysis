"""
OOS validation harness for the Phase-12 v2 strategy.

This script does NOT tune anything. It runs the FROZEN strategy
(`build_final_strategy_v2` with the default parameters from RESEARCH.md)
on three disjoint windows and prints a unified metric report:

    IS       : original research panel  (2025-01-02 → 2026-03-31)
    BACK_OOS : Alpaca-pulled bars       (2024-01-02 → 2024-12-31)
    FWD_OOS  : Alpaca-pulled bars       (2026-04-01 → today)

For each window we report:
    Sharpe, Ann return, MaxDD, Calmar, hit-rate, avg gross,
    daily turnover (×), fraction of bars where the regime gate is on,
    number of "fresh entries" (a proxy for trade count), monthly Sharpe,
    and the proxy used (XLK with QQQ fallback).

Outputs are written to:
    data_qtc/validation/oos_summary.json
    data_qtc/validation/oos_monthly.csv
    data_qtc/validation/oos_weights_<window>.parquet
    data_qtc/validation/oos_pnl_<window>.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.xsec_strategy import (
    backtest_weights,
    build_final_strategy_v2,
    portfolio_metrics,
)


TARGETS = ["AAPL", "AMD", "MSFT", "NVDA"]
PRIMARY_PROXY = "XLK"
FALLBACK_PROXY = "QQQ"

# Frozen v2 defaults — identical to research config.
PARAMS = dict(
    signal_horizon=30,
    ts_window=390,
    z_in=2.0,
    z_out_ratio=0.2,
    hold_bars=60,
    dispersion_fast=5,
    dispersion_slow=60,
    use_breadth_gate=True,
    use_per_stock_gate=True,
    gross=1.0,
)
COST_BPS = 1.0  # match the rest of the project's "1bp/side" reporting

OUT_DIR = ROOT / "data_qtc" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------

def load_research_panel() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data_qtc" / "combined" / "research_panel.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _attach_required_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol pct_change + a placeholder rolling_vol column.
    `build_wide_panels` insists on both. ret_1m is the strategy input;
    rolling_vol is plumbed but unused by `build_final_strategy_v2`."""
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    df["ret_1m"] = df.groupby("symbol", sort=False)["close"].pct_change()
    df["rolling_vol"] = (
        df.groupby("symbol", sort=False)["ret_1m"]
        .transform(lambda s: s.abs().rolling(20, min_periods=5).mean())
    )
    return df


def load_oos(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data_qtc" / "oos" / filename)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = _attach_required_cols(df)
    return df


# ----------------------------------------------------------------------
# Pipeline wrapper
# ----------------------------------------------------------------------

def _slice_universe(panel: pd.DataFrame, proxy: str) -> pd.DataFrame:
    needed = TARGETS + [proxy]
    sub = panel[panel["symbol"].isin(needed)].copy()
    return sub


def _try_run_v2(panel: pd.DataFrame, proxy: str):
    sub = _slice_universe(panel, proxy)
    if sub.empty or len(sub["symbol"].unique()) < len(TARGETS) + 1:
        return None
    return build_final_strategy_v2(sub, TARGETS, proxy_symbol=proxy, **PARAMS)


def run_v2_with_proxy_fallback(panel: pd.DataFrame, label: str) -> dict:
    """Try XLK first; fall back to QQQ if XLK has too few aligned bars
    (common on Alpaca free IEX feed for low-volume names)."""
    out = _try_run_v2(panel, PRIMARY_PROXY)
    proxy_used = PRIMARY_PROXY
    if out is None or out["weights_final"].notna().any().sum() == 0 or \
       out["weights_final"].abs().sum().sum() == 0:
        out2 = _try_run_v2(panel, FALLBACK_PROXY)
        if out2 is not None and out2["weights_final"].abs().sum().sum() > 0:
            out = out2
            proxy_used = FALLBACK_PROXY
    if out is None:
        raise RuntimeError(f"[{label}] could not run v2 with either proxy")
    print(f"[{label}] proxy used = {proxy_used}, "
          f"aligned bars = {len(out['returns_wide']):,}")
    return {"out": out, "proxy_used": proxy_used}


# ----------------------------------------------------------------------
# Trade counting
# ----------------------------------------------------------------------

def count_fresh_entries(weights: pd.DataFrame) -> int:
    """A 'fresh entry' = the per-stock weight transitioned from 0 to non-zero
    OR flipped sign. This is the closest analogue to `# trades` for a
    weight-based backtest and matches what would be sent to the broker."""
    sign = np.sign(weights)
    prev = sign.shift(1).fillna(0)
    fresh = ((sign != 0) & (sign != prev)).sum().sum()
    return int(fresh)


def per_symbol_trade_breakdown(weights: pd.DataFrame) -> dict[str, int]:
    sign = np.sign(weights)
    prev = sign.shift(1).fillna(0)
    fresh = ((sign != 0) & (sign != prev)).sum()
    return {c: int(v) for c, v in fresh.items()}


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def evaluate_window(label: str, panel: pd.DataFrame) -> dict:
    print(f"\n=== {label} ===")
    pack = run_v2_with_proxy_fallback(panel, label)
    out = pack["out"]
    proxy = pack["proxy_used"]

    weights = out["weights_final"]
    rets = out["returns_wide"]
    bt = backtest_weights(weights, rets, cost_bps=COST_BPS)
    m = portfolio_metrics(bt)

    # Diagnostics
    n_aligned = len(rets)
    gate_global_on = float(out["regime_mask_combined"].fillna(False).mean()) \
        if "regime_mask_combined" in out else 0.0
    gate_per_stock_on = (
        float(out["regime_mask_per_stock"].fillna(0).mean().mean())
        if out.get("regime_mask_per_stock") is not None else 0.0
    )
    n_trades_total = count_fresh_entries(weights)
    per_sym_trades = per_symbol_trade_breakdown(weights)

    n_days = max(n_aligned / 390.0, 1.0)
    trades_per_day = n_trades_total / n_days

    # Save weights + pnl per window
    bt.to_parquet(OUT_DIR / f"oos_pnl_{label}.parquet")
    weights.to_parquet(OUT_DIR / f"oos_weights_{label}.parquet")

    return {
        "label": label,
        "proxy": proxy,
        "aligned_bars": n_aligned,
        "trading_days_equiv": round(n_days, 1),
        "sharpe": round(m["sharpe"], 3),
        "sortino": round(m["sortino"], 3),
        "ann_return_pct": round(100 * m["ann_return"], 2),
        "max_dd_pct": round(100 * m["max_drawdown"], 2),
        "calmar": round(m["calmar"], 3),
        "bar_hit_rate": round(m["bar_hit_rate"], 4),
        "avg_gross_exposure": round(m["avg_gross_exposure"], 4),
        "daily_turnover_x": round(m["avg_daily_turnover_x"], 3),
        "cost_share": round(m["cost_share_of_gross"], 4),
        "global_gate_on_pct": round(100 * gate_global_on, 2),
        "per_stock_gate_on_pct": round(100 * gate_per_stock_on, 2),
        "fresh_entries_total": n_trades_total,
        "fresh_entries_per_day": round(trades_per_day, 3),
        "fresh_entries_per_symbol": per_sym_trades,
    }


def per_month_breakdown(label: str) -> pd.DataFrame:
    bt = pd.read_parquet(OUT_DIR / f"oos_pnl_{label}.parquet")
    bt.index = pd.to_datetime(bt.index, utc=True)
    bt["__ym"] = bt.index.tz_convert("America/New_York").strftime("%Y-%m")
    grp = bt.groupby("__ym")
    monthly = pd.DataFrame({
        "month": list(grp.groups.keys()),
        "n_bars": grp.size().values,
        "mean_pnl": grp["net_pnl"].mean().values,
        "std_pnl": grp["net_pnl"].std(ddof=0).values,
        "sum_pnl": grp["net_pnl"].sum().values,
        "sum_turnover": grp["turnover"].sum().values,
    })
    monthly["sharpe"] = (
        monthly["mean_pnl"] / monthly["std_pnl"].replace(0, np.nan)
        * np.sqrt(390 * 252)
    )
    monthly["window"] = label
    return monthly


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    is_panel = load_research_panel()
    back_panel = load_oos("back_oos_2024.parquet")
    fwd_panel = load_oos("fwd_oos_2026q2.parquet")

    summary: list[dict] = []
    summary.append(evaluate_window("IS_2025_2026Q1", is_panel))
    summary.append(evaluate_window("BACK_OOS_2024", back_panel))
    summary.append(evaluate_window("FWD_OOS_2026Q2", fwd_panel))

    print("\n=== SUMMARY ===")
    for row in summary:
        print(json.dumps(row, indent=2))

    with open(OUT_DIR / "oos_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    monthlies = []
    for row in summary:
        try:
            monthlies.append(per_month_breakdown(row["label"]))
        except Exception as exc:
            print(f"[{row['label']}] monthly breakdown failed: {exc}")
    if monthlies:
        all_monthly = pd.concat(monthlies, ignore_index=True)
        all_monthly.to_csv(OUT_DIR / "oos_monthly.csv", index=False)
        print("\n=== MONTHLY ===")
        print(all_monthly.to_string(index=False))


if __name__ == "__main__":
    main()
