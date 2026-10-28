"""
Compare V1 (long+short) vs V3 (long-only) across stress windows.

V3's catastrophic loss in TECH_BEAR_2022 (-25% Ann) suggests long-only
is regime-dependent. This script tests whether V1 long+short survives
the same window — and whether either version is honestly recommendable
for a strategy that has to weather any regime.
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

TARGETS = ["AAPL", "AMD", "MSFT", "NVDA"]
PROXY = "XLK"

PARAMS = dict(
    signal_horizon=30, ts_window=390,
    z_in=2.75, z_out_ratio=0.2, hold_bars=120,
    dispersion_fast=5, dispersion_slow=60,
    gross=1.0,
)

STRESS_WINDOWS = ["TECH_BEAR_2022", "POST_BEAR_2023"]
OUT_DIR = ROOT / "data_qtc" / "validation"


def state_machine_long_only(signal_z, **kw):
    z_in = kw["z_in"]; z_out_ratio = kw["z_out_ratio"]
    hold_bars = kw["hold_bars"]; gross = kw["gross"]
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


def load_window(label):
    path = ROOT / "data_qtc" / "stress" / f"{label}.parquet"
    df = _load(path)
    df["ret_1m"] = df.groupby("symbol", sort=False)["close"].pct_change()
    df["rolling_vol"] = (
        df.groupby("symbol", sort=False)["ret_1m"]
        .transform(lambda s: s.abs().rolling(20, min_periods=5).mean())
    )
    return df


def evaluate(panel, kind):
    sub = panel[panel["symbol"].isin(TARGETS + [PROXY])].copy()
    wp = build_wide_panels(sub, TARGETS, PROXY)
    ret = wp["ret_1m"]; proxy = wp["proxy_ret_1m"]
    resid, _ = residualize(ret, proxy, window=PARAMS["ts_window"])
    sig = resid.rolling(PARAMS["signal_horizon"]).sum()
    z = ts_zscore(sig, window=PARAMS["ts_window"])
    disp = dispersion_regime_filter(
        resid, PARAMS["dispersion_fast"], PARAMS["dispersion_slow"]
    )
    if kind == "V1_LS":
        w = sparse_reversion_state_machine(
            z, z_in=PARAMS["z_in"], z_out_ratio=PARAMS["z_out_ratio"],
            hold_bars=60, gross=PARAMS["gross"],
        )
    elif kind == "V1_LS_h120":
        w = sparse_reversion_state_machine(
            z, z_in=PARAMS["z_in"], z_out_ratio=PARAMS["z_out_ratio"],
            hold_bars=120, gross=PARAMS["gross"],
        )
    elif kind == "V3_LO_h120":
        w = state_machine_long_only(z, **PARAMS)
    elif kind == "RAW_NO_GATE":
        w = sparse_reversion_state_machine(
            z, z_in=PARAMS["z_in"], z_out_ratio=PARAMS["z_out_ratio"],
            hold_bars=60, gross=PARAMS["gross"],
        )
        # skip the regime mask
        w.index = pd.to_datetime(w.index, utc=True); ret.index = w.index
        return _score(w, ret, kind)
    else:
        raise ValueError(kind)
    w = apply_regime_mask(w, disp)
    w.index = pd.to_datetime(w.index, utc=True); ret.index = w.index
    return _score(w, ret, kind)


def _score(w, ret, kind):
    bt = backtest_weights(w, ret, cost_bps=1.0)
    m = portfolio_metrics(bt)
    sign = np.sign(w); prev = sign.shift(1).fillna(0)
    n_trades = int(((sign != 0) & (sign != prev)).sum().sum())
    per_sym_pnl = (w.shift(1) * ret).sum(axis=0)
    per_sym_pnl = {k: round(v * 1e4, 2) for k, v in per_sym_pnl.items()}
    return {
        "variant": kind,
        "sharpe_1bp": round(m["sharpe"], 3),
        "ann_pct": round(100 * m["ann_return"], 2),
        "maxdd_pct": round(100 * m["max_drawdown"], 2),
        "calmar": round(m["calmar"], 3),
        "n_trades": n_trades,
        "per_sym_pnl_bps": per_sym_pnl,
    }


def main():
    rows = []
    for win in STRESS_WINDOWS:
        print(f"\n=== {win} ===")
        panel = load_window(win)
        for kind in ["V1_LS", "V1_LS_h120", "V3_LO_h120", "RAW_NO_GATE"]:
            try:
                r = evaluate(panel, kind)
                r["window"] = win
                rows.append(r)
                print(f"  {kind:>14s}: Sharpe={r['sharpe_1bp']:+.2f}  "
                      f"ann={r['ann_pct']:+.2f}%  maxDD={r['maxdd_pct']:.2f}%  "
                      f"trades={r['n_trades']}")
                print(f"  {' ':>14s}  per-sym={r['per_sym_pnl_bps']}")
            except Exception as e:
                print(f"  {kind}: ERROR {e}")
    df = pd.DataFrame(rows)

    # Pivot tables
    print("\n=== SHARPE BY WINDOW × VARIANT ===")
    print(df.pivot_table(index="variant", columns="window",
                          values="sharpe_1bp", aggfunc="first").to_string())
    print("\n=== ANN% BY WINDOW × VARIANT ===")
    print(df.pivot_table(index="variant", columns="window",
                          values="ann_pct", aggfunc="first").to_string())
    print("\n=== MAXDD% BY WINDOW × VARIANT ===")
    print(df.pivot_table(index="variant", columns="window",
                          values="maxdd_pct", aggfunc="first").to_string())

    df.to_csv(OUT_DIR / "stress_variant_compare.csv", index=False)
    with open(OUT_DIR / "stress_variant_compare.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
