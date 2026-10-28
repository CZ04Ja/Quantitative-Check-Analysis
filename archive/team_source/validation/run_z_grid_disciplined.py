"""
Disciplined z_in grid scan for V1.

Goal: find a parameter setting (if any) that improves BOTH the IS window
AND the BACK_OOS_2024 window simultaneously, at cost = 1 bps and 3 bps.
We only accept a parameter change if it Pareto-improves on the current
default of z_in=2.0 across both samples and both cost levels.

This is the only "optimization" we allow without re-introducing window
data snooping: a candidate that is not Pareto-dominant in this 2x2 grid
is discarded.

Output: data_qtc/validation/z_grid_disciplined.json
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
    apply_regime_mask,
    backtest_weights,
    build_wide_panels,
    dispersion_regime_filter,
    portfolio_metrics,
    residualize,
    sparse_reversion_state_machine,
    ts_zscore,
)

TARGETS = ["AAPL", "AMD", "MSFT", "NVDA"]
PROXY = "XLK"

BASE_PARAMS = dict(
    signal_horizon=30, ts_window=390,
    z_out_ratio=0.2, hold_bars=60,
    dispersion_fast=5, dispersion_slow=60,
    gross=1.0,
)
Z_GRID = [2.0, 2.25, 2.5, 2.75, 3.0]
COST_GRID = [1.0, 3.0]
WINDOWS = {
    "BACK_OOS_2024":  ("2024-01-01", "2024-12-31"),
    "IS_2025_2026Q1": ("2025-01-01", "2026-03-31"),
}

OUT_DIR = ROOT / "data_qtc" / "validation"


def _required_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    if "ret_1m" not in df.columns:
        df["ret_1m"] = df.groupby("symbol", sort=False)["close"].pct_change()
    if "rolling_vol" not in df.columns:
        df["rolling_vol"] = (
            df.groupby("symbol", sort=False)["ret_1m"]
            .transform(lambda s: s.abs().rolling(20, min_periods=5).mean())
        )
    return df


def load_combined() -> pd.DataFrame:
    keep = ["timestamp", "symbol", "open", "high", "low", "close", "volume",
            "ret_1m", "rolling_vol"]

    def _load(p: Path) -> pd.DataFrame:
        df = pd.read_parquet(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = _required_cols(df)
        df["symbol"] = df["symbol"].astype(str)
        return df[[c for c in keep if c in df.columns]]

    parts = [
        _load(ROOT / "data_qtc" / "oos" / "back_oos_2024.parquet"),
        _load(ROOT / "data_qtc" / "combined" / "research_panel.parquet"),
        _load(ROOT / "data_qtc" / "oos" / "fwd_oos_2026q2.parquet"),
    ]
    full = pd.concat(parts, ignore_index=True)
    full = (
        full.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )
    return full


def _slice(df: pd.DataFrame, s: str, e: str) -> pd.DataFrame:
    s_ts = pd.Timestamp(s, tz="UTC")
    e_ts = pd.Timestamp(e, tz="UTC")
    return df.loc[(df.index >= s_ts) & (df.index < e_ts)]


def main() -> None:
    full = load_combined()
    sub = full[full["symbol"].isin(TARGETS + [PROXY])].copy()
    wp = build_wide_panels(sub, TARGETS, PROXY)
    ret = wp["ret_1m"]
    proxy = wp["proxy_ret_1m"]
    resid, _ = residualize(ret, proxy, window=BASE_PARAMS["ts_window"])
    sig = resid.rolling(BASE_PARAMS["signal_horizon"]).sum()
    z = ts_zscore(sig, window=BASE_PARAMS["ts_window"])
    disp_mask = dispersion_regime_filter(
        resid, BASE_PARAMS["dispersion_fast"], BASE_PARAMS["dispersion_slow"]
    )

    rows = []
    for z_in in Z_GRID:
        w_raw = sparse_reversion_state_machine(
            z, z_in=z_in, z_out_ratio=BASE_PARAMS["z_out_ratio"],
            hold_bars=BASE_PARAMS["hold_bars"], gross=BASE_PARAMS["gross"],
        )
        w_v1 = apply_regime_mask(w_raw, disp_mask)
        w_v1.index = pd.to_datetime(w_v1.index, utc=True)
        ret.index = w_v1.index

        for cost in COST_GRID:
            for label, (s, e) in WINDOWS.items():
                w_s = _slice(w_v1, s, e); r_s = _slice(ret, s, e)
                bt = backtest_weights(w_s, r_s, cost_bps=cost)
                m = portfolio_metrics(bt)
                # Count fresh entries
                sign = np.sign(w_s); prev = sign.shift(1).fillna(0)
                n_trades = int(((sign != 0) & (sign != prev)).sum().sum())
                rows.append({
                    "z_in": z_in, "cost_bps": cost, "window": label,
                    "sharpe": round(m["sharpe"], 3),
                    "ann_pct": round(100 * m["ann_return"], 2),
                    "maxdd_pct": round(100 * m["max_drawdown"], 2),
                    "n_trades": n_trades,
                    "cost_share": round(m["cost_share_of_gross"], 3),
                })

    df = pd.DataFrame(rows)

    # Pivot for readable display
    print("\n=== Sharpe by z_in × window × cost ===")
    pv = df.pivot_table(
        index=["z_in", "cost_bps"], columns="window",
        values="sharpe", aggfunc="first",
    )
    print(pv.to_string())

    print("\n=== Trade count by z_in × window ===")
    pv_n = df.pivot_table(
        index="z_in", columns="window", values="n_trades", aggfunc="max",
    )
    print(pv_n.to_string())

    # Pareto check vs baseline z_in=2.0
    print("\n=== Pareto improvements over z_in=2.0 baseline ===")
    base = df[df["z_in"] == 2.0].set_index(["window", "cost_bps"])["sharpe"]
    accepted = []
    for z_in in Z_GRID:
        if z_in == 2.0:
            continue
        cand = df[df["z_in"] == z_in].set_index(["window", "cost_bps"])["sharpe"]
        deltas = (cand - base)
        is_pareto = (deltas >= 0).all() and (deltas > 0).any()
        is_strict = (deltas > 0.05).all()
        verdict = "ACCEPT" if is_strict else ("pareto-tie" if is_pareto else "REJECT")
        print(f"  z_in={z_in}: deltas={deltas.to_dict()} -> {verdict}")
        if verdict == "ACCEPT":
            accepted.append(z_in)

    out = {
        "rows": rows,
        "accepted_pareto": accepted,
        "decision": (
            f"Adopt z_in={accepted[0]}" if accepted
            else "Stay with z_in=2.0 (no candidate Pareto-dominates baseline)"
        ),
    }
    with open(OUT_DIR / "z_grid_disciplined.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== DECISION: {out['decision']} ===")


if __name__ == "__main__":
    main()
