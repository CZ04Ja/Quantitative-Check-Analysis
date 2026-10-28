"""
V3 iteration v5 — risk-control angles.

Try improvements that are RISK CONTROLS rather than alpha tricks:

    A. cap per-symbol weight to ≤ K * gross (K in {0.30, 0.40, 0.50})
       — prevents a single-name event (e.g. NVDA Q2) from sinking the book
    B. long-only (skip all short entries)
       — needed for Tiger if shorting is restricted/expensive
    C. hold_120 (best from v4)
    D. hold_120 + long-only
    E. hold_120 + cap_30
    F. hold_120 + long-only + cap_30 (full conservative)

All on 4u + XLK + z_in=2.75 (the proven base).
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


def state_machine_long_only(signal_z, z_in=2.75, z_out_ratio=0.2,
                              hold_bars=60, gross=1.0):
    """Variant of sparse_reversion_state_machine that ONLY takes long
    entries (z <= -z_in). Short signals (z >= +z_in) are skipped."""
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


def state_machine_full(signal_z, z_in=2.75, z_out_ratio=0.2,
                       hold_bars=60, gross=1.0):
    from src.xsec_strategy import sparse_reversion_state_machine
    return sparse_reversion_state_machine(
        signal_z, z_in=z_in, z_out_ratio=z_out_ratio,
        hold_bars=hold_bars, gross=gross,
    )


def cap_per_symbol(weights: pd.DataFrame, cap: float, gross: float = 1.0):
    """Cap |w_i| ≤ cap * gross per bar; renormalise so total |w| sums to gross."""
    w = weights.copy()
    cap_val = cap * gross
    w_clipped = w.clip(lower=-cap_val, upper=cap_val)
    s = w_clipped.abs().sum(axis=1).replace(0, np.nan)
    return w_clipped.div(s, axis=0).mul(gross).fillna(0)


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


def build_signal_z(panel):
    """Compute the residual z-score panel; return (z, ret, disp_mask)."""
    sub = panel[panel["symbol"].isin(TARGETS + [PROXY])].copy()
    wp = build_wide_panels(sub, TARGETS, PROXY)
    ret = wp["ret_1m"]; proxy = wp["proxy_ret_1m"]
    resid, _ = residualize(ret, proxy, window=PARAMS["ts_window"])
    sig = resid.rolling(PARAMS["signal_horizon"]).sum()
    z = ts_zscore(sig, window=PARAMS["ts_window"])
    disp_mask = dispersion_regime_filter(
        resid, PARAMS["dispersion_fast"], PARAMS["dispersion_slow"]
    )
    return z, ret, resid, disp_mask


def evaluate(label_w, w, r):
    out = []
    w.index = pd.to_datetime(w.index, utc=True); r.index = w.index
    for label, (s, e) in WINDOWS.items():
        w_s = _slice(w, s, e); r_s = _slice(r, s, e)
        if len(w_s) == 0:
            continue
        for cost in [1.0, 3.0, 5.0]:
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
            })
    return out


def main():
    full = load_combined()
    z, r, resid, disp = build_signal_z(full)

    rows = []

    # Baseline (V1+z=2.75)
    print("--- BASELINE V1+z=2.75 ---")
    w0 = state_machine_full(z, z_in=2.75, hold_bars=60)
    w0 = apply_regime_mask(w0, disp)
    rows.extend(evaluate("V1_baseline", w0, r))

    # Cap per-symbol
    for cap in [0.30, 0.40, 0.50]:
        print(f"--- cap_{int(cap*100)}% ---")
        w_cap = cap_per_symbol(w0, cap)
        rows.extend(evaluate(f"cap_{int(cap*100)}", w_cap, r))

    # Long-only baseline
    print("--- long_only ---")
    w_lo = state_machine_long_only(z, z_in=2.75, hold_bars=60)
    w_lo = apply_regime_mask(w_lo, disp)
    rows.extend(evaluate("long_only", w_lo, r))

    # hold_120 baseline (re-run so we get full Q2 too)
    print("--- hold_120 ---")
    w_h = state_machine_full(z, z_in=2.75, hold_bars=120)
    w_h = apply_regime_mask(w_h, disp)
    rows.extend(evaluate("hold_120", w_h, r))

    # hold_120 + long_only
    print("--- hold_120 + long_only ---")
    w_hl = state_machine_long_only(z, z_in=2.75, hold_bars=120)
    w_hl = apply_regime_mask(w_hl, disp)
    rows.extend(evaluate("hold_120_long_only", w_hl, r))

    # hold_120 + cap_30
    print("--- hold_120 + cap_30 ---")
    w_hc = cap_per_symbol(w_h, 0.30)
    rows.extend(evaluate("hold_120_cap_30", w_hc, r))

    # hold_120 + long_only + cap_30 (full conservative)
    print("--- hold_120 + long_only + cap_30 ---")
    w_full = cap_per_symbol(w_hl, 0.30)
    rows.extend(evaluate("hold_120_long_only_cap_30", w_full, r))

    df = pd.DataFrame(rows)

    print("\n=== SHARPE TABLE (1bp) ===")
    print(df[df["cost_bps"] == 1.0].pivot_table(
        index="variant", columns="window", values="sharpe").to_string())
    print("\n=== SHARPE TABLE (3bp) ===")
    print(df[df["cost_bps"] == 3.0].pivot_table(
        index="variant", columns="window", values="sharpe").to_string())
    print("\n=== SHARPE TABLE (5bp) ===")
    print(df[df["cost_bps"] == 5.0].pivot_table(
        index="variant", columns="window", values="sharpe").to_string())
    print("\n=== MAXDD% (1bp) ===")
    print(df[df["cost_bps"] == 1.0].pivot_table(
        index="variant", columns="window", values="maxdd_pct").to_string())

    # Pareto vs baseline (relax: 0 worse cells, ≥1 better cell, on IS+BACK ONLY)
    print("\n=== PARETO vs V1_baseline (IS+BACK only) ===")
    base = df[df["variant"] == "V1_baseline"].set_index(
        ["window", "cost_bps"])["sharpe"]
    accepts = []
    for var in df["variant"].unique():
        if var == "V1_baseline":
            continue
        cand = df[df["variant"] == var].set_index(["window", "cost_bps"])["sharpe"]
        cells = [("BACK_OOS_2024", 1.0), ("BACK_OOS_2024", 3.0),
                 ("IS_2025_2026Q1", 1.0), ("IS_2025_2026Q1", 3.0)]
        deltas = {c: round(float(cand.get(c, 0) - base.get(c, 0)), 3)
                  for c in cells}
        n_better = sum(1 for v in deltas.values() if v > 0)
        n_worse = sum(1 for v in deltas.values() if v < -0.01)
        is_pareto = (n_worse == 0 and n_better >= 1)
        verdict = "ACCEPT" if is_pareto else "REJECT"
        # Also report Q2 delta as info
        q2d = round(float(cand.get(("FWD_OOS_2026Q2", 1.0), 0) -
                          base.get(("FWD_OOS_2026Q2", 1.0), 0)), 3)
        print(f"  {var}: deltas={deltas}  Q2_delta={q2d}  {verdict}")
        if is_pareto:
            accepts.append(var)
    print(f"\n  Accepted: {accepts}")

    df.to_csv(OUT_DIR / "v3_iteration_v5.csv", index=False)
    with open(OUT_DIR / "v3_iteration_v5.json", "w") as f:
        json.dump({"rows": rows, "pareto_accepted": accepts}, f, indent=2)


if __name__ == "__main__":
    main()
