"""
Ablation diagnostics on the continuous panel.

Goal: figure out WHERE the v2 edge collapses on truly unseen 2024 and
on the longer continuous IS panel. Compares:

    NO_GATE        : raw state-machine, no regime filtering
    DISP_ONLY (v1) : Notebook-03 final, single-channel dispersion gate
    V2_FULL        : Phase-12 v2, breadth + per-stock gates

For each variant we slice the continuous-panel PnL into the same three
windows and report Sharpe / MaxDD / cost share / # trades. This isolates
whether the bleeding comes from the signal (NO_GATE bad), the dispersion
gate (DISP_ONLY bad), or the breadth/per-stock add-on (V2 bad).

Outputs to data_qtc/validation/ablation_summary.json + .csv
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
    breadth_regime_filter,
    build_wide_panels,
    dispersion_regime_filter,
    per_stock_vol_mask,
    portfolio_metrics,
    residualize,
    sparse_reversion_state_machine,
    ts_zscore,
)

TARGETS = ["AAPL", "AMD", "MSFT", "NVDA"]
PROXY = "XLK"

PARAMS = dict(
    signal_horizon=30, ts_window=390,
    z_in=2.0, z_out_ratio=0.2, hold_bars=60,
    dispersion_fast=5, dispersion_slow=60,
    gross=1.0,
)
COST_BPS = 1.0

OUT_DIR = ROOT / "data_qtc" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = {
    "BACK_OOS_2024":  ("2024-01-01", "2024-12-31"),
    "IS_2025_2026Q1": ("2025-01-01", "2026-03-31"),
    "FWD_OOS_2026Q2": ("2026-04-01", "2027-01-01"),
}


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


# ------------------------------------------------------------------
# Build the three ablation weight matrices on the SAME signal
# ------------------------------------------------------------------

def build_all_variants(panel: pd.DataFrame) -> dict[str, dict]:
    sub = panel[panel["symbol"].isin(TARGETS + [PROXY])].copy()
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

    # v1: dispersion-only gate
    disp_mask = dispersion_regime_filter(
        resid, PARAMS["dispersion_fast"], PARAMS["dispersion_slow"]
    )
    w_v1 = apply_regime_mask(w_raw, disp_mask)

    # v2: breadth + per-stock gates
    gates = breadth_regime_filter(
        resid, PARAMS["dispersion_fast"], PARAMS["dispersion_slow"]
    )
    w_v2_global = apply_regime_mask(w_raw, gates["combined"])
    psm = per_stock_vol_mask(
        resid, PARAMS["dispersion_fast"], PARAMS["dispersion_slow"]
    )
    w_v2 = w_v2_global.mul(psm.reindex_like(w_v2_global).fillna(0))

    # no-gate baseline = w_raw
    return {
        "NO_GATE":   {"weights": w_raw, "rets": ret, "extra": {}},
        "DISP_ONLY": {"weights": w_v1, "rets": ret,
                       "extra": {"gate_on_pct": float(disp_mask.fillna(False).mean()) * 100}},
        "V2_FULL":   {"weights": w_v2, "rets": ret,
                       "extra": {
                           "global_on_pct": float(gates["combined"].fillna(False).mean()) * 100,
                           "per_stock_on_pct": float(psm.fillna(0).mean().mean()) * 100,
                       }},
    }


def _slice(df: pd.DataFrame, s: str, e: str) -> pd.DataFrame:
    s_ts = pd.Timestamp(s, tz="UTC")
    e_ts = pd.Timestamp(e, tz="UTC")
    return df.loc[(df.index >= s_ts) & (df.index < e_ts)]


def count_fresh(weights: pd.DataFrame) -> int:
    sign = np.sign(weights)
    prev = sign.shift(1).fillna(0)
    return int(((sign != 0) & (sign != prev)).sum().sum())


def evaluate_variant(label: str, variant: str, w: pd.DataFrame, r: pd.DataFrame,
                     extra: dict, slice_start: str, slice_end: str) -> dict:
    w_s = _slice(w, slice_start, slice_end)
    r_s = _slice(r, slice_start, slice_end)
    if w_s.empty:
        return {"variant": variant, "label": label, "aligned_bars": 0}
    bt = backtest_weights(w_s, r_s, cost_bps=COST_BPS)
    m = portfolio_metrics(bt)
    n_aligned = len(bt)
    n_days = max(n_aligned / 390.0, 1.0)
    n_trades = count_fresh(w_s)
    return {
        "variant": variant,
        "label": label,
        "aligned_bars": n_aligned,
        "trading_days": round(n_days, 1),
        "sharpe": round(m["sharpe"], 3),
        "ann_pct": round(100 * m["ann_return"], 2),
        "maxdd_pct": round(100 * m["max_drawdown"], 2),
        "calmar": round(m["calmar"], 3),
        "cost_share": round(m["cost_share_of_gross"], 4),
        "trades": n_trades,
        "trades_per_day": round(n_trades / n_days, 3),
    }


def main() -> None:
    full = load_combined()
    print(f"[combined] rows={len(full):,}, range={full['timestamp'].min()} → {full['timestamp'].max()}")
    variants = build_all_variants(full)

    rows: list[dict] = []
    for variant_name, pack in variants.items():
        w = pack["weights"]
        r = pack["rets"]
        w.index = pd.to_datetime(w.index, utc=True)
        r.index = w.index
        for label, (s, e) in WINDOWS.items():
            row = evaluate_variant(label, variant_name, w, r, pack["extra"], s, e)
            rows.append(row)

    df = pd.DataFrame(rows)
    print("\n=== ABLATION (continuous panel, then sliced) ===")
    print(df.pivot_table(
        index="label",
        columns="variant",
        values=["sharpe", "ann_pct", "maxdd_pct", "trades", "cost_share"],
        aggfunc="first",
    ).to_string())

    df.to_csv(OUT_DIR / "ablation_summary.csv", index=False)
    with open(OUT_DIR / "ablation_summary.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
