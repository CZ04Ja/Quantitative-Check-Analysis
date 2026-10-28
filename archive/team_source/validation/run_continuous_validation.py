"""
Continuous-panel OOS validation.

Concatenates BACK_OOS_2024 + IS_2025_2026Q1 + FWD_OOS_2026Q2 into a single
long panel, runs the FROZEN v2 strategy ONCE so the regime gate enjoys
continuous warm-up, then slices the resulting PnL into time windows for
metric reporting.

This is the fairest OOS test we can do without changing strategy parameters
or peeking at the OOS windows. The gate is built on `disp_slow=60d` of
trailing data; without continuous concatenation, the FWD_OOS window
(only 5 weeks) cannot wake up at all.

Outputs:
    data_qtc/validation/cont_summary.json
    data_qtc/validation/cont_monthly.csv
    data_qtc/validation/cont_pnl.parquet
    data_qtc/validation/cont_weights.parquet
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
    backtest_weights,
    build_final_strategy_v2,
    portfolio_metrics,
)

TARGETS = ["AAPL", "AMD", "MSFT", "NVDA"]
PROXY = "XLK"

PARAMS = dict(
    signal_horizon=30, ts_window=390,
    z_in=2.0, z_out_ratio=0.2, hold_bars=60,
    dispersion_fast=5, dispersion_slow=60,
    use_breadth_gate=True, use_per_stock_gate=True,
    gross=1.0,
)
COST_BPS = 1.0

OUT_DIR = ROOT / "data_qtc" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = {
    "BACK_OOS_2024":   ("2024-01-01", "2024-12-31"),
    "IS_2025_2026Q1":  ("2025-01-01", "2026-03-31"),
    "FWD_OOS_2026Q2":  ("2026-04-01", "2027-01-01"),
}


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------

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

    back = _load(ROOT / "data_qtc" / "oos" / "back_oos_2024.parquet")
    is_  = _load(ROOT / "data_qtc" / "combined" / "research_panel.parquet")
    fwd  = _load(ROOT / "data_qtc" / "oos" / "fwd_oos_2026q2.parquet")

    full = pd.concat([back, is_, fwd], ignore_index=True)
    full = (
        full.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )

    print(f"[combined] rows={len(full):,}  range={full['timestamp'].min()} → {full['timestamp'].max()}")
    print("[combined] per-symbol bars:")
    print(full["symbol"].value_counts().to_string())
    return full


# ------------------------------------------------------------------
# Sliced metrics
# ------------------------------------------------------------------

def _slice_by_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    return df.loc[(df.index >= s) & (df.index < e)]


def count_fresh_entries(weights: pd.DataFrame) -> int:
    sign = np.sign(weights)
    prev = sign.shift(1).fillna(0)
    return int(((sign != 0) & (sign != prev)).sum().sum())


def per_symbol_trade_breakdown(weights: pd.DataFrame) -> dict[str, int]:
    sign = np.sign(weights)
    prev = sign.shift(1).fillna(0)
    fresh = ((sign != 0) & (sign != prev)).sum()
    return {c: int(v) for c, v in fresh.items()}


def evaluate_slice(label: str, bt_full: pd.DataFrame,
                   weights_full: pd.DataFrame, gate_combined: pd.Series,
                   gate_per_stock: pd.DataFrame, start: str, end: str) -> dict:
    bt = _slice_by_window(bt_full, start, end)
    w  = _slice_by_window(weights_full, start, end)
    if len(bt) == 0:
        return {"label": label, "aligned_bars": 0, "note": "empty slice"}
    gc = _slice_by_window(gate_combined.to_frame("g"), start, end)["g"]
    gp = _slice_by_window(gate_per_stock, start, end) if gate_per_stock is not None else None

    m = portfolio_metrics(bt)
    n_aligned = len(bt)
    n_days = max(n_aligned / 390.0, 1.0)
    n_trades = count_fresh_entries(w)
    return {
        "label": label,
        "start": start, "end": end,
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
        "global_gate_on_pct": round(100 * float(gc.fillna(False).mean()), 2),
        "per_stock_gate_on_pct": (
            round(100 * float(gp.fillna(0).mean().mean()), 2)
            if gp is not None else 0.0
        ),
        "fresh_entries_total": n_trades,
        "fresh_entries_per_day": round(n_trades / n_days, 3),
        "fresh_entries_per_symbol": per_symbol_trade_breakdown(w),
    }


# ------------------------------------------------------------------
# Bootstrap CI for the OOS slices
# ------------------------------------------------------------------

def block_bootstrap_sharpe(
    pnl: pd.Series, block_bars: int = 390 * 2, n_iter: int = 2000, seed: int = 7,
) -> dict:
    """2-day block bootstrap of an annualised Sharpe ratio."""
    arr = pnl.values
    n = len(arr)
    if n < block_bars * 2:
        return {"n_iter": 0, "median": float("nan"), "p05": float("nan"),
                "p95": float("nan"), "p_gt_0": float("nan")}
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
        "median": float(np.median(out)),
        "p05": float(np.percentile(out, 5)),
        "p25": float(np.percentile(out, 25)),
        "p75": float(np.percentile(out, 75)),
        "p95": float(np.percentile(out, 95)),
        "p_gt_0": float((out > 0).mean()),
    }


# ------------------------------------------------------------------
# Per-month
# ------------------------------------------------------------------

def per_month(bt_full: pd.DataFrame) -> pd.DataFrame:
    bt = bt_full.copy()
    bt["__ym"] = bt.index.tz_convert("America/New_York").strftime("%Y-%m")
    grp = bt.groupby("__ym")
    out = pd.DataFrame({
        "month": list(grp.groups.keys()),
        "n_bars": grp.size().values,
        "mean_pnl": grp["net_pnl"].mean().values,
        "std_pnl": grp["net_pnl"].std(ddof=0).values,
        "sum_pnl": grp["net_pnl"].sum().values,
        "sum_turnover": grp["turnover"].sum().values,
    })
    out["sharpe"] = (
        out["mean_pnl"] / out["std_pnl"].replace(0, np.nan)
        * np.sqrt(390 * 252)
    )
    return out


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    full = load_combined()
    sub = full[full["symbol"].isin(TARGETS + [PROXY])].copy()
    print(f"[run] sub rows={len(sub):,}, building v2 ...")
    out = build_final_strategy_v2(sub, TARGETS, proxy_symbol=PROXY, **PARAMS)
    weights = out["weights_final"]
    rets = out["returns_wide"]
    bt = backtest_weights(weights, rets, cost_bps=COST_BPS)
    bt.index = pd.to_datetime(bt.index, utc=True)
    weights.index = bt.index

    bt.to_parquet(OUT_DIR / "cont_pnl.parquet")
    weights.to_parquet(OUT_DIR / "cont_weights.parquet")

    # Slice & report
    summary = []
    for label, (s, e) in WINDOWS.items():
        row = evaluate_slice(
            label, bt, weights,
            out["regime_mask_combined"],
            out.get("regime_mask_per_stock"),
            s, e,
        )
        summary.append(row)
    print("\n=== SLICED METRICS (continuous panel) ===")
    for r in summary:
        print(json.dumps(r, indent=2))

    # Bootstrap on the two truly OOS windows
    boot = {}
    for label, (s, e) in WINDOWS.items():
        if not label.endswith("_OOS_2024") and not label.endswith("_2026Q2"):
            continue
        seg = _slice_by_window(bt, s, e)
        if len(seg) > 0:
            boot[label] = block_bootstrap_sharpe(seg["net_pnl"])
    print("\n=== BOOTSTRAP CI (OOS only) ===")
    for k, v in boot.items():
        print(k, json.dumps(v, indent=2))

    with open(OUT_DIR / "cont_summary.json", "w") as f:
        json.dump({"summary": summary, "bootstrap": boot}, f, indent=2)

    monthly = per_month(bt)
    monthly.to_csv(OUT_DIR / "cont_monthly.csv", index=False)
    print("\n=== MONTHLY (continuous panel) ===")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
