"""
V3 iteration v4 — back to 4u + XLK proxy (the only setup with proven OOS
edge), apply ONLY the improvements that don't depend on universe size:

    1. drop close-15min          (consistently negative across windows)
    2. drop close-30min          (more aggressive)
    3. drop open-5min            (microstructure noise)
    4. inverse-vol scaling       (auto-diversifies away NVDA-style risk)
    5. longer hold_bars=120      (lower turnover -> lower cost share)
    6. higher z_in=3.0           (rarer entries)

Each candidate must Pareto-improve V1+z=2.75 baseline in BOTH BACK_OOS
and IS at BOTH 1bp and 3bp. We do NOT optimise on Q2.
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

PARAMS_BASE = dict(
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


def build_v1(panel, params):
    sub = panel[panel["symbol"].isin(TARGETS + [PROXY])].copy()
    wp = build_wide_panels(sub, TARGETS, PROXY)
    ret = wp["ret_1m"]; proxy = wp["proxy_ret_1m"]
    resid, _ = residualize(ret, proxy, window=params["ts_window"])
    sig = resid.rolling(params["signal_horizon"]).sum()
    z = ts_zscore(sig, window=params["ts_window"])
    w_raw = sparse_reversion_state_machine(
        z, z_in=params["z_in"], z_out_ratio=params["z_out_ratio"],
        hold_bars=params["hold_bars"], gross=params["gross"],
    )
    disp_mask = dispersion_regime_filter(
        resid, params["dispersion_fast"], params["dispersion_slow"]
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
            })
    return out


def main():
    full = load_combined()

    rows = []

    # Baseline
    print("--- baseline V1+z=2.75 (4u, XLK) ---")
    w_base, r = build_v1(full, PARAMS_BASE)
    rows.extend(evaluate("V1_4u_baseline", w_base, r))

    # 1) drop close-15
    print("--- drop close-15 ---")
    tod15 = time_of_day_mask(w_base.index, 0, 15)
    w_close15 = w_base.mul(tod15.astype(float), axis=0)
    rows.extend(evaluate("no_close15", w_close15, r))

    # 2) drop close-30
    print("--- drop close-30 ---")
    tod30 = time_of_day_mask(w_base.index, 0, 30)
    w_close30 = w_base.mul(tod30.astype(float), axis=0)
    rows.extend(evaluate("no_close30", w_close30, r))

    # 3) drop open-5
    print("--- drop open-5 ---")
    tod_o5 = time_of_day_mask(w_base.index, 5, 0)
    w_open5 = w_base.mul(tod_o5.astype(float), axis=0)
    rows.extend(evaluate("no_open5", w_open5, r))

    # 4) inverse-vol
    print("--- inverse-vol scaling ---")
    w_iv = inverse_vol_scale(w_base, r)
    rows.extend(evaluate("invvol", w_iv, r))

    # 5) longer hold = 120
    print("--- hold_bars = 120 ---")
    p2 = dict(PARAMS_BASE); p2["hold_bars"] = 120
    w_h120, _ = build_v1(full, p2)
    rows.extend(evaluate("hold_120", w_h120, r))

    # 6) z_in = 3.0
    print("--- z_in = 3.0 ---")
    p3 = dict(PARAMS_BASE); p3["z_in"] = 3.0
    w_z30, _ = build_v1(full, p3)
    rows.extend(evaluate("z30", w_z30, r))

    # COMBOS — only the ones that look likely after seeing the singles
    # We'll add these in a follow-up pass

    df = pd.DataFrame(rows)
    print("\n=== SHARPE TABLE (1bp) ===")
    print(df[df["cost_bps"] == 1.0].pivot_table(
        index="variant", columns="window", values="sharpe").to_string())
    print("\n=== SHARPE TABLE (3bp) ===")
    print(df[df["cost_bps"] == 3.0].pivot_table(
        index="variant", columns="window", values="sharpe").to_string())
    print("\n=== MAXDD (1bp) ===")
    print(df[df["cost_bps"] == 1.0].pivot_table(
        index="variant", columns="window", values="maxdd_pct").to_string())
    print("\n=== TRADE COUNT ===")
    print(df.pivot_table(index="variant", columns="window",
                          values="trades", aggfunc="max").to_string())

    # Pareto check vs baseline
    print("\n=== PARETO ANALYSIS vs V1_4u_baseline ===")
    base = df[df["variant"] == "V1_4u_baseline"].set_index(
        ["window", "cost_bps"])["sharpe"]
    accepts = []
    for var in df["variant"].unique():
        if var == "V1_4u_baseline":
            continue
        cand = df[df["variant"] == var].set_index(["window", "cost_bps"])["sharpe"]
        # Only check IS and BACK_OOS (don't optimize on Q2)
        cells = [("BACK_OOS_2024", 1.0), ("BACK_OOS_2024", 3.0),
                 ("IS_2025_2026Q1", 1.0), ("IS_2025_2026Q1", 3.0)]
        deltas = {c: round(float(cand.get(c, np.nan) - base.get(c, np.nan)), 3)
                  for c in cells}
        n_better = sum(1 for v in deltas.values() if v > 0)
        n_worse = sum(1 for v in deltas.values() if v < 0)
        is_pareto = (n_worse == 0 and n_better >= 1)
        verdict = "ACCEPT" if is_pareto else "REJECT"
        print(f"  {var}: deltas={deltas}  {verdict}")
        if is_pareto:
            accepts.append(var)
    print(f"\n  Accepted (Pareto-positive on IS+BACK): {accepts}")

    df.to_csv(OUT_DIR / "v3_iteration_v4.csv", index=False)
    with open(OUT_DIR / "v3_iteration_v4.json", "w") as f:
        json.dump({"rows": rows, "pareto_accepted": accepts}, f, indent=2)


if __name__ == "__main__":
    main()
