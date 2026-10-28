"""
Final candidate: V1 dispersion-only gate + z_in=2.75 (Pareto-validated).

Compute bootstrap CI in BOTH windows (IS and BACK_OOS_2024) plus full
cost-stress, plus per-symbol trade attribution. This is the candidate
we recommend for Tiger paper-trade.
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
COST_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

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


def block_bootstrap_sharpe(pnl: pd.Series, block_bars: int = 390 * 2,
                            n_iter: int = 2000, seed: int = 7) -> dict:
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


def main() -> None:
    full = load_combined()
    sub = full[full["symbol"].isin(TARGETS + [PROXY])].copy()
    wp = build_wide_panels(sub, TARGETS, PROXY)
    ret = wp["ret_1m"]
    proxy = wp["proxy_ret_1m"]
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

    report: dict = {"params": PARAMS, "windows": {}, "cost_stress": {}}

    for label, (s, e) in WINDOWS.items():
        w_s = _slice(w, s, e); r_s = _slice(ret, s, e)
        if len(w_s) == 0:
            continue
        bt = backtest_weights(w_s, r_s, cost_bps=1.0)
        m = portfolio_metrics(bt)
        boot = block_bootstrap_sharpe(bt["net_pnl"])
        sign = np.sign(w_s); prev = sign.shift(1).fillna(0)
        n_trades = int(((sign != 0) & (sign != prev)).sum().sum())
        per_sym = {c: int(v) for c, v in
                   ((sign != 0) & (sign != prev)).sum().items()}
        n_days = max(len(w_s) / 390.0, 1.0)
        report["windows"][label] = {
            "aligned_bars": len(w_s),
            "trading_days": round(n_days, 1),
            "sharpe_1bp": round(m["sharpe"], 3),
            "ann_pct_1bp": round(100 * m["ann_return"], 2),
            "maxdd_pct_1bp": round(100 * m["max_drawdown"], 2),
            "calmar_1bp": round(m["calmar"], 3),
            "n_trades": n_trades,
            "trades_per_day": round(n_trades / n_days, 3),
            "trades_per_symbol": per_sym,
            "bootstrap_1bp": boot,
        }

    # Cost stress on each window
    for label, (s, e) in WINDOWS.items():
        w_s = _slice(w, s, e); r_s = _slice(ret, s, e)
        if len(w_s) == 0:
            continue
        rows = []
        for c in COST_GRID:
            bt = backtest_weights(w_s, r_s, cost_bps=c)
            m = portfolio_metrics(bt)
            rows.append({
                "cost_bps": c,
                "sharpe": round(m["sharpe"], 3),
                "ann_pct": round(100 * m["ann_return"], 2),
                "maxdd_pct": round(100 * m["max_drawdown"], 2),
                "cost_share": round(m["cost_share_of_gross"], 3),
            })
        report["cost_stress"][label] = rows

    print(json.dumps(report, indent=2))
    with open(OUT_DIR / "v1_z275_final.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
