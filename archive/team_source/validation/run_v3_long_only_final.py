"""
Final V3 candidate: V1 + z_in=2.75 + LONG-ONLY + hold_bars=120.

This is now the recommended Tiger candidate. Verify with:
    1. Bootstrap CI in all 3 windows
    2. Cost stress 0-10 bps
    3. Per-symbol attribution
    4. Sensitivity sweep over hold_bars (60, 90, 120, 180)
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
    z_in=2.75, z_out_ratio=0.2,
    dispersion_fast=5, dispersion_slow=60,
    gross=1.0,
)
HOLD_GRID = [60, 90, 120, 180]
COST_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
WINDOWS = {
    "BACK_OOS_2024":  ("2024-01-01", "2024-12-31"),
    "IS_2025_2026Q1": ("2025-01-01", "2026-03-31"),
    "FWD_OOS_2026Q2": ("2026-04-01", "2027-01-01"),
}
OUT_DIR = ROOT / "data_qtc" / "validation"


def state_machine_long_only(signal_z, z_in=2.75, z_out_ratio=0.2,
                              hold_bars=60, gross=1.0):
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


def block_bootstrap_sharpe(pnl, block_bars=390 * 2, n_iter=2000, seed=7):
    arr = pnl.values; n = len(arr)
    if n < block_bars * 2:
        return {"n_iter": 0}
    rng = np.random.default_rng(seed)
    n_blocks = n // block_bars
    starts = np.arange(n - block_bars + 1)
    ann = np.sqrt(390 * 252)
    out = np.empty(n_iter)
    for i in range(n_iter):
        sel = rng.choice(starts, size=n_blocks, replace=True)
        chunks = [arr[s:s + block_bars] for s in sel]
        s_arr = np.concatenate(chunks)
        sd = s_arr.std(ddof=0)
        out[i] = (s_arr.mean() / sd * ann) if sd > 0 else 0.0
    return {
        "n_iter": n_iter,
        "median": round(float(np.median(out)), 3),
        "p05": round(float(np.percentile(out, 5)), 3),
        "p25": round(float(np.percentile(out, 25)), 3),
        "p75": round(float(np.percentile(out, 75)), 3),
        "p95": round(float(np.percentile(out, 95)), 3),
        "p_gt_0": round(float((out > 0).mean()), 3),
        "p_gt_1": round(float((out > 1).mean()), 3),
    }


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


def _slice(df, s, e):
    s_ts = pd.Timestamp(s, tz="UTC"); e_ts = pd.Timestamp(e, tz="UTC")
    return df.loc[(df.index >= s_ts) & (df.index < e_ts)]


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

    report: dict = {}

    # --- Hold sensitivity ---
    print("\n=== Hold-bars sensitivity (long-only) ===")
    sens_rows = []
    for hb in HOLD_GRID:
        w = state_machine_long_only(
            z, z_in=PARAMS["z_in"], z_out_ratio=PARAMS["z_out_ratio"],
            hold_bars=hb, gross=PARAMS["gross"],
        )
        w = apply_regime_mask(w, disp)
        w.index = pd.to_datetime(w.index, utc=True); ret.index = w.index
        for label, (s, e) in WINDOWS.items():
            w_s = _slice(w, s, e); r_s = _slice(ret, s, e)
            if len(w_s) == 0:
                continue
            for cost in [1.0, 3.0]:
                bt = backtest_weights(w_s, r_s, cost_bps=cost)
                m = portfolio_metrics(bt)
                sens_rows.append({
                    "hold_bars": hb, "window": label, "cost_bps": cost,
                    "sharpe": round(m["sharpe"], 3),
                    "ann_pct": round(100 * m["ann_return"], 2),
                    "maxdd_pct": round(100 * m["max_drawdown"], 2),
                })
    sens_df = pd.DataFrame(sens_rows)
    print(sens_df.pivot_table(
        index=["hold_bars", "cost_bps"], columns="window",
        values="sharpe").to_string())
    report["hold_sensitivity"] = sens_rows

    # --- FINAL CANDIDATE: hold_120 long_only ---
    print("\n=== FINAL: V3 = V1 + z=2.75 + long-only + hold=120 ===")
    HOLD_FINAL = 120
    w = state_machine_long_only(
        z, z_in=PARAMS["z_in"], z_out_ratio=PARAMS["z_out_ratio"],
        hold_bars=HOLD_FINAL, gross=PARAMS["gross"],
    )
    w = apply_regime_mask(w, disp)
    w.index = pd.to_datetime(w.index, utc=True); ret.index = w.index

    final = {}
    for label, (s, e) in WINDOWS.items():
        w_s = _slice(w, s, e); r_s = _slice(ret, s, e)
        if len(w_s) == 0:
            continue
        bt = backtest_weights(w_s, r_s, cost_bps=1.0)
        m = portfolio_metrics(bt)
        boot = block_bootstrap_sharpe(bt["net_pnl"])
        sign = np.sign(w_s); prev = sign.shift(1).fillna(0)
        n_trades = int(((sign != 0) & (sign != prev)).sum().sum())
        per_sym = ((sign != 0) & (sign != prev)).sum().to_dict()
        per_sym = {k: int(v) for k, v in per_sym.items()}
        per_sym_pnl = (w_s.shift(1) * r_s).sum(axis=0)
        per_sym_pnl = {k: round(v * 1e4, 2) for k, v in per_sym_pnl.items()}
        n_days = max(len(w_s) / 390.0, 1.0)
        final[label] = {
            "aligned_bars": len(w_s),
            "trading_days": round(n_days, 1),
            "sharpe_1bp": round(m["sharpe"], 3),
            "ann_pct_1bp": round(100 * m["ann_return"], 2),
            "maxdd_pct_1bp": round(100 * m["max_drawdown"], 2),
            "calmar_1bp": round(m["calmar"], 3),
            "n_trades": n_trades,
            "trades_per_day": round(n_trades / n_days, 3),
            "trades_per_symbol": per_sym,
            "gross_pnl_per_symbol_bps": per_sym_pnl,
            "bootstrap_1bp": boot,
        }
    report["final_v3"] = {"params": dict(PARAMS, hold_bars=HOLD_FINAL,
                                          state_machine="long_only"), "windows": final}

    # Cost stress
    print("\n=== V3 cost stress ===")
    cost_rows = []
    for label, (s, e) in WINDOWS.items():
        w_s = _slice(w, s, e); r_s = _slice(ret, s, e)
        if len(w_s) == 0:
            continue
        for c in COST_GRID:
            bt = backtest_weights(w_s, r_s, cost_bps=c)
            m = portfolio_metrics(bt)
            cost_rows.append({
                "window": label, "cost_bps": c,
                "sharpe": round(m["sharpe"], 3),
                "ann_pct": round(100 * m["ann_return"], 2),
                "maxdd_pct": round(100 * m["max_drawdown"], 2),
            })
    print(pd.DataFrame(cost_rows).pivot_table(
        index="cost_bps", columns="window", values="sharpe").to_string())
    report["cost_stress"] = cost_rows

    print("\n=== FINAL JSON ===")
    print(json.dumps(report["final_v3"], indent=2))
    with open(OUT_DIR / "v3_long_only_final.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
