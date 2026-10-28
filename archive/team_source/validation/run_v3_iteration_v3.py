"""
V3 iteration v3 — no ffill (it broke z-score). Use SPY proxy
(best 2024 coverage) + inner-join.

Sub-universes tested:
    4u            : AAPL AMD MSFT NVDA  (original)
    7u            : 4u + GOOGL META AMZN   (high-coverage tech)
    10u           : 4u + 6 added (full)
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
    dispersion_regime_filter, portfolio_metrics, residualize,
    sparse_reversion_state_machine, ts_zscore,
)

UNIVERSES = {
    "4u":  ["AAPL", "AMD", "MSFT", "NVDA"],
    "7u":  ["AAPL", "AMD", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA"],
    "10u": ["AAPL", "AMD", "MSFT", "NVDA",
            "GOOGL", "META", "AMZN", "AVGO", "TSLA", "ORCL"],
}
PROXY = "SPY"

PARAMS = dict(
    signal_horizon=30, ts_window=390,
    z_in=2.75, z_out_ratio=0.2, hold_bars=60,
    dispersion_fast=5, dispersion_slow=60,
    gross=1.0,
)
WINDOWS = {
    "BACK_OOS_2024":  ("2024-01-01", "2024-12-31"),
    "IS_2025_2026Q1": ("2025-01-01", "2026-03-31"),
    "FWD_OOS_2026Q2": ("2026-04-01", "2027-01-01"),
}
OUT_DIR = ROOT / "data_qtc" / "validation"


def _load(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = df["symbol"].astype(str)
    return df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]


def load_combined() -> pd.DataFrame:
    parts = [
        _load(ROOT / "data_qtc" / "oos" / "back_oos_2024.parquet"),
        _load(ROOT / "data_qtc" / "combined" / "research_panel.parquet"),
        _load(ROOT / "data_qtc" / "oos" / "fwd_oos_2026q2.parquet"),
        _load(ROOT / "data_qtc" / "expanded" / "expanded_2024_2026.parquet"),
        _load(ROOT / "data_qtc" / "expanded" / "spy_2024_2026.parquet"),
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


def _slice(df, s, e):
    s_ts = pd.Timestamp(s, tz="UTC"); e_ts = pd.Timestamp(e, tz="UTC")
    return df.loc[(df.index >= s_ts) & (df.index < e_ts)]


def time_of_day_mask(index, skip_first_min=0, skip_last_min=15):
    ts_ny = index.tz_convert("America/New_York")
    mins = ts_ny.hour * 60 + ts_ny.minute
    open_min = 9 * 60 + 30; close_min = 16 * 60
    keep = ((mins >= open_min + skip_first_min) &
            (mins < close_min - skip_last_min))
    return pd.Series(keep, index=index)


def inverse_vol_scale(weights, ret, vol_window=390 * 5,
                       gross=1.0, eps=1e-6):
    vol = ret.rolling(vol_window, min_periods=vol_window // 5).std(ddof=0).shift(1)
    inv = 1.0 / (vol + eps)
    w = weights.mul(inv).fillna(0)
    s = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(s, axis=0).mul(gross).fillna(0)


def build_v1_weights(panel, targets):
    sub = panel[panel["symbol"].isin(targets + [PROXY])].copy()
    wp = build_wide_panels(sub, targets, PROXY)
    ret = wp["ret_1m"]; proxy = wp["proxy_ret_1m"]
    resid, _ = residualize(ret, proxy, window=PARAMS["ts_window"])
    sig = resid.rolling(PARAMS["signal_horizon"]).sum()
    z = ts_zscore(sig, window=PARAMS["ts_window"])
    w_raw = sparse_reversion_state_machine(
        z, z_in=PARAMS["z_in"], z_out_ratio=PARAMS["z_out_ratio"],
        hold_bars=PARAMS["hold_bars"], gross=PARAMS["gross"],
    )
    disp_mask = dispersion_regime_filter(
        resid, PARAMS["dispersion_fast"], PARAMS["dispersion_slow"]
    )
    w = apply_regime_mask(w_raw, disp_mask)
    w.index = pd.to_datetime(w.index, utc=True); ret.index = w.index
    return w, ret


def evaluate(label_w, w, r):
    out = []
    for label, (s, e) in WINDOWS.items():
        w_s = _slice(w, s, e); r_s = _slice(r, s, e)
        if len(w_s) == 0:
            continue
        for cost in [1.0, 3.0]:
            bt = backtest_weights(w_s, r_s, cost_bps=cost)
            m = portfolio_metrics(bt)
            sign = np.sign(w_s); prev = sign.shift(1).fillna(0)
            n_trades = int(((sign != 0) & (sign != prev)).sum().sum())
            n_days = max(len(w_s) / 390.0, 1.0)
            out.append({
                "variant": label_w, "window": label, "cost_bps": cost,
                "sharpe": round(m["sharpe"], 3),
                "ann_pct": round(100 * m["ann_return"], 2),
                "maxdd_pct": round(100 * m["max_drawdown"], 2),
                "calmar": round(m["calmar"], 3),
                "trades": n_trades,
                "trades_per_day": round(n_trades / n_days, 2),
                "cost_share": round(m["cost_share_of_gross"], 3),
                "aligned_bars": len(w_s),
            })
    return out


def main():
    full = load_combined()
    print(f"[combined] rows={len(full):,}")

    rows = []
    for uname, syms in UNIVERSES.items():
        print(f"\n--- V1_{uname} (SPY proxy, inner-join) ---")
        try:
            w, r = build_v1_weights(full, syms)
            rows.extend(evaluate(f"V1_{uname}", w, r))
        except Exception as e:
            print(f"  ERROR for {uname}: {e}"); continue

    # V3 candidates on the WINNING universe (chosen post-hoc only if it
    # Pareto-dominates 4u; otherwise stay on 4u). For now apply variants
    # to every universe and inspect.
    for uname, syms in UNIVERSES.items():
        try:
            w, r = build_v1_weights(full, syms)
        except Exception:
            continue
        # V3a: drop close-15min
        tod = time_of_day_mask(w.index, 0, 15)
        w_v3a = w.mul(tod.astype(float), axis=0)
        rows.extend(evaluate(f"V3a_{uname}_no_close15", w_v3a, r))
        # V3b: V3a + invvol
        w_v3b = inverse_vol_scale(w_v3a, r)
        rows.extend(evaluate(f"V3b_{uname}_no_close15+invvol", w_v3b, r))

    df = pd.DataFrame(rows)
    print("\n=== SHARPE TABLE (1bp) ===")
    pv1 = df[df["cost_bps"] == 1.0].pivot_table(
        index="variant", columns="window", values="sharpe", aggfunc="first")
    print(pv1.to_string())
    print("\n=== SHARPE TABLE (3bp) ===")
    pv3 = df[df["cost_bps"] == 3.0].pivot_table(
        index="variant", columns="window", values="sharpe", aggfunc="first")
    print(pv3.to_string())
    print("\n=== MAXDD TABLE (1bp) ===")
    pv4 = df[df["cost_bps"] == 1.0].pivot_table(
        index="variant", columns="window", values="maxdd_pct", aggfunc="first")
    print(pv4.to_string())
    print("\n=== ALIGNED BARS BY UNIVERSE ===")
    pv5 = df.pivot_table(
        index="variant", columns="window", values="aligned_bars", aggfunc="max")
    print(pv5.to_string())
    print("\n=== TRADE COUNT ===")
    pv6 = df.pivot_table(
        index="variant", columns="window", values="trades", aggfunc="max")
    print(pv6.to_string())

    df.to_csv(OUT_DIR / "v3_iteration_v3.csv", index=False)
    with open(OUT_DIR / "v3_iteration_v3.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
