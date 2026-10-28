"""
True walk-forward sanity check on V3.

Approach:
    For each calendar month from 2024-04 to 2026-04, evaluate V3
    long-only on JUST that month. The strategy parameters are fixed
    (no in-sample tuning); the gate uses the 60-day slow median
    computed STRICTLY from data BEFORE the month start (the
    `dispersion_regime_filter` shifts by 1 bar, so this is enforced).

    The test asks: does V3's edge persist month-by-month across two
    full years, or is the average-Sharpe driven by a few lucky months?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.xsec_strategy import (
    apply_regime_mask, backtest_weights, build_wide_panels,
    dispersion_regime_filter, portfolio_metrics, residualize, ts_zscore,
)

TARGETS = ["AAPL", "AMD", "MSFT", "NVDA"]
PROXY = "XLK"

PARAMS = dict(
    signal_horizon=30, ts_window=390,
    z_in=2.75, z_out_ratio=0.2, hold_bars=120,
    dispersion_fast=5, dispersion_slow=60,
    gross=1.0,
)
OUT_DIR = ROOT / "data_qtc" / "validation"


def state_machine_long_only(signal_z, z_in, z_out_ratio, hold_bars, gross):
    T, N = signal_z.shape
    positions = np.zeros((T, N), dtype=float)
    sz = signal_z.fillna(0).values
    z_exit = z_in * z_out_ratio
    cur = np.zeros(N); held = np.zeros(N, dtype=int)
    for t in range(T):
        for s in range(N):
            zt = sz[t, s]
            if cur[s] != 0:
                held[s] += 1
                exit_cond = (
                    abs(zt) < z_exit
                    or held[s] >= hold_bars
                    or (cur[s] > 0 and zt > 0)
                )
                if exit_cond:
                    cur[s] = 0; held[s] = 0
            if cur[s] == 0 and zt <= -z_in:
                cur[s] = 1; held[s] = 0
            positions[t, s] = cur[s]
    w = pd.DataFrame(positions, index=signal_z.index, columns=signal_z.columns)
    na = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(na, axis=0).mul(gross).fillna(0)


def _load(p):
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = df["symbol"].astype(str)
    return df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]


def load_combined():
    parts = [
        _load(ROOT / "data_qtc" / "oos" / "back_oos_2024.parquet"),
        _load(ROOT / "data_qtc" / "combined" / "research_panel.parquet"),
        _load(ROOT / "data_qtc" / "oos" / "fwd_oos_2026q2.parquet"),
    ]
    full = pd.concat(parts, ignore_index=True)
    full = (full.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
            .sort_values(["symbol", "timestamp"]).reset_index(drop=True))
    full["ret_1m"] = full.groupby("symbol", sort=False)["close"].pct_change()
    full["rolling_vol"] = (
        full.groupby("symbol", sort=False)["ret_1m"]
        .transform(lambda s: s.abs().rolling(20, min_periods=5).mean())
    )
    return full


def main():
    full = load_combined()
    sub = full[full["symbol"].isin(TARGETS + [PROXY])].copy()
    wp = build_wide_panels(sub, TARGETS, PROXY)
    ret = wp["ret_1m"]; proxy = wp["proxy_ret_1m"]
    resid, _ = residualize(ret, proxy, window=PARAMS["ts_window"])
    sig = resid.rolling(PARAMS["signal_horizon"]).sum()
    z = ts_zscore(sig, window=PARAMS["ts_window"])
    disp = dispersion_regime_filter(
        resid, PARAMS["dispersion_fast"], PARAMS["dispersion_slow"]
    )
    w = state_machine_long_only(
        z, PARAMS["z_in"], PARAMS["z_out_ratio"],
        PARAMS["hold_bars"], PARAMS["gross"]
    )
    w = apply_regime_mask(w, disp)
    w.index = pd.to_datetime(w.index, utc=True); ret.index = w.index

    bt_full = backtest_weights(w, ret, cost_bps=1.0)

    bt_full["ym"] = bt_full.index.tz_convert("America/New_York").strftime("%Y-%m")
    monthly = []
    grp = bt_full.groupby("ym")
    for ym, g in grp:
        n = len(g)
        if n < 200:  # skip incomplete months
            continue
        sd = g["net_pnl"].std(ddof=0)
        sharpe = (g["net_pnl"].mean() / sd * np.sqrt(390 * 252)) if sd > 0 else 0.0
        sum_pnl = g["net_pnl"].sum() * 1e4
        n_turn = g["turnover"].sum()
        # trade count
        # (approx: count weight sign-changes within this month)
        w_m = w.loc[g.index]
        sign = np.sign(w_m); prev = sign.shift(1).fillna(0)
        n_trades = int(((sign != 0) & (sign != prev)).sum().sum())
        monthly.append({
            "month": ym, "bars": n,
            "sum_pnl_bps": round(sum_pnl, 2),
            "sharpe": round(float(sharpe), 3),
            "trades": n_trades,
        })
    df = pd.DataFrame(monthly)
    df["positive"] = df["sharpe"] > 0
    print("\n=== MONTH-BY-MONTH (V3 long-only, 1bp) ===")
    print(df.to_string(index=False))
    print(f"\nMonths with Sharpe > 0: {df['positive'].sum()}/{len(df)}  "
          f"({df['positive'].mean()*100:.1f}%)")
    print(f"Median monthly Sharpe: {df['sharpe'].median():.3f}")
    print(f"Mean monthly Sharpe:   {df['sharpe'].mean():.3f}")
    print(f"Std of monthly Sharpe: {df['sharpe'].std(ddof=0):.3f}")
    print(f"Worst month:  {df.loc[df['sharpe'].idxmin(), 'month']} "
          f"sharpe={df['sharpe'].min():.3f}")
    print(f"Best month:   {df.loc[df['sharpe'].idxmax(), 'month']} "
          f"sharpe={df['sharpe'].max():.3f}")
    print(f"Total trades: {df['trades'].sum()}")
    print(f"Total PnL (sum): {df['sum_pnl_bps'].sum():+.2f} bps over "
          f"{len(df)} months")

    df.to_csv(OUT_DIR / "v3_walkforward.csv", index=False)


if __name__ == "__main__":
    main()
