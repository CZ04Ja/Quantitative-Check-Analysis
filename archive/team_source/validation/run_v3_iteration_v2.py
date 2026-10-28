"""
V3 iteration v2 — fixes the sparse-IEX inner-join data loss problem.

Two changes from v1:
    a) Proxy switched to QQQ (2024 bar count 91K vs XLK 47K).
    b) Long panel is densified before pivoting: each (symbol, date)
       grid is forward-filled within the NY trading session so a
       missing IEX print becomes a 0-return bar instead of dropping
       the whole row from every symbol.

This is NOT data snooping — it is a data-cleaning step that any
production pipeline would do anyway when blending Alpaca free IEX
with a real broker.
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
PROXY = "QQQ"

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
    ]
    full = pd.concat(parts, ignore_index=True)
    full = (full.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
            .sort_values(["symbol", "timestamp"]).reset_index(drop=True))
    return full


def densify_within_session(panel_long: pd.DataFrame,
                            symbols: list[str]) -> pd.DataFrame:
    """For each (symbol, NY date), reindex to the regular session grid
    09:30 → 16:00 NY (390 bars) and forward-fill close. Recompute
    ret_1m per symbol; first bar of each session has NaN return.
    Drops symbols entirely missing from `panel_long`.
    """
    panel_long = panel_long[panel_long["symbol"].isin(symbols)].copy()
    panel_long["ts_ny"] = panel_long["timestamp"].dt.tz_convert("America/New_York")
    panel_long["date_ny"] = panel_long["ts_ny"].dt.date

    # All trading dates that appear in the data (intersection across
    # symbols): use the union so nothing is dropped, then ffill.
    all_dates = sorted(panel_long["date_ny"].unique())

    out_chunks = []
    for sym in symbols:
        sub = panel_long[panel_long["symbol"] == sym].copy()
        if sub.empty:
            continue
        sub = sub.set_index("ts_ny")
        sub = sub[~sub.index.duplicated(keep="last")]
        # Build the regular minute grid for each session date present
        sym_dates = sorted(sub.index.normalize().unique())
        grids = []
        for d in sym_dates:
            d_pd = pd.Timestamp(d).tz_convert("America/New_York")
            start = d_pd.replace(hour=9, minute=30, second=0, microsecond=0)
            end = d_pd.replace(hour=16, minute=0, second=0, microsecond=0)
            grids.append(pd.date_range(start, end - pd.Timedelta(minutes=1),
                                        freq="1min", tz="America/New_York"))
        full_idx = pd.DatetimeIndex(sorted(set().union(*grids)))
        sub2 = sub.reindex(full_idx)
        # Forward-fill close, NaN out first bar of each session for ret_1m
        sub2["close"] = sub2["close"].ffill()
        sub2["volume"] = sub2["volume"].fillna(0)
        for c in ["open", "high", "low"]:
            sub2[c] = sub2[c].fillna(sub2["close"])
        sub2["timestamp"] = sub2.index.tz_convert("UTC")
        sub2["symbol"] = sym
        sub2["date_ny"] = sub2.index.date
        # ret_1m, but reset first-bar-of-session to NaN
        sub2["ret_1m"] = sub2["close"].pct_change()
        is_first = (sub2.index == sub2.groupby("date_ny").apply(lambda g: g.index.min())
                     .reindex(sub2.index.date).values)
        sub2.loc[is_first, "ret_1m"] = np.nan
        sub2["rolling_vol"] = sub2["ret_1m"].abs().rolling(20, min_periods=5).mean()
        out_chunks.append(sub2.reset_index(drop=True))
    return pd.concat(out_chunks, ignore_index=True)


def _slice(df, s, e):
    s_ts = pd.Timestamp(s, tz="UTC"); e_ts = pd.Timestamp(e, tz="UTC")
    return df.loc[(df.index >= s_ts) & (df.index < e_ts)]


def time_of_day_mask(index, skip_first_min: int = 0,
                     skip_last_min: int = 15):
    ts_ny = index.tz_convert("America/New_York")
    mins = ts_ny.hour * 60 + ts_ny.minute
    open_min = 9 * 60 + 30
    close_min = 16 * 60
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


def build_v1_weights(panel_dense: pd.DataFrame, targets: list[str]) -> tuple:
    sub = panel_dense[panel_dense["symbol"].isin(targets + [PROXY])].copy()
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
            })
    return out


def main():
    full = load_combined()
    print(f"[combined] rows={len(full):,}")
    print("[combined] symbols:", sorted(full['symbol'].unique()))

    print("\n[densify] running session-fill on 4u + 10u universe...")
    panel_dense = densify_within_session(full, EXP_TARGETS + [PROXY])
    print(f"[densify] dense panel rows={len(panel_dense):,}")
    print(f"[densify] per-symbol bars after densify:")
    print(panel_dense["symbol"].value_counts().to_string())

    rows = []
    print("\n--- V1_4u (baseline) on densified data, QQQ proxy ---")
    w4, r4 = build_v1_weights(panel_dense, OG_TARGETS)
    rows.extend(evaluate("V1_4u_dense_QQQ", w4, r4))

    print("--- V1_10u (10 stocks, QQQ proxy, densified) ---")
    w10, r10 = build_v1_weights(panel_dense, EXP_TARGETS)
    rows.extend(evaluate("V1_10u_dense_QQQ", w10, r10))

    print("--- V3a: 10u + drop close-15min ---")
    tod = time_of_day_mask(w10.index, 0, 15)
    w_v3a = w10.mul(tod.astype(float), axis=0)
    rows.extend(evaluate("V3a_10u_no_close15", w_v3a, r10))

    print("--- V3b: V3a + inverse-vol ---")
    w_v3b = inverse_vol_scale(w_v3a, r10)
    rows.extend(evaluate("V3b_10u_no_close15+invvol", w_v3b, r10))

    print("--- V3c: V3b + skip first 5 min ---")
    tod2 = time_of_day_mask(w10.index, 5, 15)
    w_v3c = inverse_vol_scale(w10.mul(tod2.astype(float), axis=0), r10)
    rows.extend(evaluate("V3c_10u_no_open5_no_close15+invvol", w_v3c, r10))

    df = pd.DataFrame(rows)
    print("\n=== SHARPE TABLE ===")
    print(df.pivot_table(
        index=["variant", "cost_bps"], columns="window",
        values="sharpe", aggfunc="first",
    ).to_string())
    print("\n=== MAXDD TABLE ===")
    print(df.pivot_table(
        index=["variant", "cost_bps"], columns="window",
        values="maxdd_pct", aggfunc="first",
    ).to_string())
    print("\n=== TRADE COUNT ===")
    print(df.pivot_table(index="variant", columns="window",
                          values="trades", aggfunc="max").to_string())

    df.to_csv(OUT_DIR / "v3_iteration_v2.csv", index=False)
    with open(OUT_DIR / "v3_iteration_v2.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
