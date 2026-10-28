"""
V3 iteration on the EXPANDED 10-stock universe.

Universe (defined before looking at performance):
    Original 4: AAPL, AMD, MSFT, NVDA
    Added 6:    GOOGL, META, AMZN, AVGO, TSLA, ORCL
    Proxy:      XLK

Each candidate variant must Pareto-improve V1+z=2.75 baseline in BOTH
windows (BACK_OOS_2024 and IS_2025_2026Q1) at BOTH cost levels (1 bps
and 3 bps). We do NOT optimise on FWD_OOS_2026Q2 (only 5 weeks of data,
plus we want to keep one window pristine for final reporting).

Variants tested:
    V1_4u   : original 4-stock V1 (baseline reference)
    V1_10u  : 10-stock V1 with everything else identical
    V3a     : 10-stock V1 + drop close-15min trades
    V3b     : 10-stock V1 + drop close-15min + inverse-vol scaling
    V3c     : V3b + drop first-5-min (open noise)
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

OG_TARGETS = ["AAPL", "AMD", "MSFT", "NVDA"]
EXP_TARGETS = ["AAPL", "AMD", "MSFT", "NVDA",
               "GOOGL", "META", "AMZN", "AVGO", "TSLA", "ORCL"]
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
        _load(ROOT / "data_qtc" / "expanded" / "expanded_2024_2026.parquet"),
    ]
    full = pd.concat(parts, ignore_index=True)
    full = (full.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
            .sort_values(["symbol", "timestamp"]).reset_index(drop=True))
    return full


def _slice(df, s, e):
    s_ts = pd.Timestamp(s, tz="UTC"); e_ts = pd.Timestamp(e, tz="UTC")
    return df.loc[(df.index >= s_ts) & (df.index < e_ts)]


def time_of_day_mask(index: pd.DatetimeIndex,
                     skip_first_min: int = 0,
                     skip_last_min: int = 15) -> pd.Series:
    ts_ny = index.tz_convert("America/New_York")
    mins = ts_ny.hour * 60 + ts_ny.minute
    open_min = 9 * 60 + 30
    close_min = 16 * 60
    keep = ((mins >= open_min + skip_first_min) &
            (mins < close_min - skip_last_min))
    return pd.Series(keep, index=index)


def inverse_vol_scale(weights: pd.DataFrame,
                       ret: pd.DataFrame,
                       vol_window: int = 390 * 5,
                       gross: float = 1.0,
                       eps: float = 1e-6) -> pd.DataFrame:
    """Scale per-stock weight by 1/rolling_vol; renormalise to fixed gross."""
    vol = ret.rolling(vol_window, min_periods=vol_window // 5).std(ddof=0).shift(1)
    inv = 1.0 / (vol + eps)
    w_scaled = weights.mul(inv).fillna(0)
    s = w_scaled.abs().sum(axis=1).replace(0, np.nan)
    return w_scaled.div(s, axis=0).mul(gross).fillna(0)


def build_v1_weights(panel: pd.DataFrame, targets: list[str]) -> tuple:
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
    w_v1 = apply_regime_mask(w_raw, disp_mask)
    w_v1.index = pd.to_datetime(w_v1.index, utc=True)
    ret.index = w_v1.index
    return w_v1, ret


def evaluate(label_w: str, w: pd.DataFrame, r: pd.DataFrame) -> list[dict]:
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
    print(f"[combined] rows={len(full):,}, range={full['timestamp'].min()} → {full['timestamp'].max()}")
    print(f"[combined] symbols available: {sorted(full['symbol'].unique())}")

    rows: list[dict] = []

    # V1_4u baseline (original 4-stock universe)
    print("\n--- V1_4u (4-stock baseline) ---")
    w_4u, r_4u = build_v1_weights(full, OG_TARGETS)
    rows.extend(evaluate("V1_4u", w_4u, r_4u))

    # V1_10u (10-stock universe, otherwise identical)
    print("--- V1_10u (10-stock universe) ---")
    w_10, r_10 = build_v1_weights(full, EXP_TARGETS)
    rows.extend(evaluate("V1_10u", w_10, r_10))

    # V3a: 10u + drop close-15min
    print("--- V3a (10u + drop close-15min) ---")
    tod = time_of_day_mask(w_10.index, skip_first_min=0, skip_last_min=15)
    w_v3a = w_10.mul(tod.astype(float), axis=0)
    rows.extend(evaluate("V3a_no_close15", w_v3a, r_10))

    # V3b: V3a + inverse-vol scaling
    print("--- V3b (V3a + inverse-vol) ---")
    w_v3b = inverse_vol_scale(w_v3a, r_10)
    rows.extend(evaluate("V3b_no_close15+invvol", w_v3b, r_10))

    # V3c: V3b + skip first 5 min
    print("--- V3c (V3b + skip first 5min) ---")
    tod2 = time_of_day_mask(w_10.index, skip_first_min=5, skip_last_min=15)
    w_v3c = inverse_vol_scale(w_10.mul(tod2.astype(float), axis=0), r_10)
    rows.extend(evaluate("V3c_skip_open5+no_close15+invvol", w_v3c, r_10))

    df = pd.DataFrame(rows)
    print("\n=== ALL VARIANTS SHARPE TABLE ===")
    pv = df.pivot_table(
        index=["variant", "cost_bps"], columns="window",
        values="sharpe", aggfunc="first",
    )
    print(pv.to_string())

    print("\n=== ALL VARIANTS MAXDD% TABLE ===")
    pv2 = df.pivot_table(
        index=["variant", "cost_bps"], columns="window",
        values="maxdd_pct", aggfunc="first",
    )
    print(pv2.to_string())

    print("\n=== ALL VARIANTS TRADE COUNT ===")
    pv3 = df.pivot_table(
        index=["variant"], columns="window", values="trades", aggfunc="max",
    )
    print(pv3.to_string())

    df.to_csv(OUT_DIR / "v3_iteration.csv", index=False)
    with open(OUT_DIR / "v3_iteration.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
