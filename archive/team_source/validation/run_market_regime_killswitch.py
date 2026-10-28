"""
Add a market-regime kill switch to V3.

Idea (well-known, Faber 2007): the strategy only runs when the
underlying sector ETF's daily close is above its N-day moving average.
Below it = the trend is bearish; long-only reversal will likely lose.

We test SMA windows of 50d, 100d, 200d. For each, evaluate V3 across
ALL windows we have data for: 2022 bear, 2023 rebound, 2024 BACK_OOS,
IS 2025-2026Q1, FWD_OOS Q2 2026.

Win criterion: filter must
    a) PROTECT capital in 2022 (turn Sharpe from -2 toward 0)
    b) NOT destroy alpha in 2024 / IS / Q2 / 2023 H2

We do NOT tune the SMA window on a single regime — we report all three
and only adopt one if all are roughly Pareto-positive.
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
    z_in=2.75, z_out_ratio=0.2, hold_bars=120,
    dispersion_fast=5, dispersion_slow=60,
    gross=1.0,
)
SMA_GRID = [50, 100, 200]
OUT_DIR = ROOT / "data_qtc" / "validation"


def state_machine_long_only(signal_z, z_in, z_out_ratio, hold_bars, gross):
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


def market_regime_mask(panel: pd.DataFrame, sma_days: int,
                        proxy: str = "XLK") -> pd.Series:
    """Returns per-minute boolean: True if proxy daily close > N-day SMA
    of daily close, evaluated as of PREVIOUS day's close (no look-ahead).
    """
    px = panel[panel["symbol"] == proxy].sort_values("timestamp").copy()
    if px.empty:
        return pd.Series(dtype=bool)
    px["ts_ny"] = px["timestamp"].dt.tz_convert("America/New_York")
    px["date"] = px["ts_ny"].dt.date
    daily = px.groupby("date")["close"].last().sort_index()
    sma = daily.rolling(sma_days, min_periods=sma_days // 2).mean()
    flag = (daily > sma).astype(float)
    flag = flag.shift(1).fillna(0)  # use yesterday's signal at today
    # Map back to minute bars
    px["regime_on"] = px["date"].map(flag.to_dict()).fillna(0)
    out = px.set_index("timestamp")["regime_on"]
    out = out[~out.index.duplicated(keep="first")]
    return out


def _load(p):
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = df["symbol"].astype(str)
    return df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]


def load_combined() -> pd.DataFrame:
    """Combine ALL windows (stress + standard) into one big panel for
    SMA calc; then we slice. The 200-day SMA needs continuous history."""
    parts = [
        _load(ROOT / "data_qtc" / "stress" / "TECH_BEAR_2022.parquet"),
        _load(ROOT / "data_qtc" / "stress" / "BANKING_2023.parquet"),
        _load(ROOT / "data_qtc" / "stress" / "POST_BEAR_2023.parquet"),
        _load(ROOT / "data_qtc" / "stress" / "POST_COVID.parquet"),
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


WINDOWS_TO_REPORT = {
    "TECH_BEAR_2022":  ("2022-01-01", "2022-12-31"),
    "POST_BEAR_2023":  ("2023-05-01", "2023-12-31"),
    "BACK_OOS_2024":   ("2024-01-01", "2024-12-31"),
    "IS_2025_2026Q1":  ("2025-01-01", "2026-03-31"),
    "FWD_OOS_2026Q2":  ("2026-04-01", "2027-01-01"),
}


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
    w_v3 = state_machine_long_only(
        z, PARAMS["z_in"], PARAMS["z_out_ratio"],
        PARAMS["hold_bars"], PARAMS["gross"]
    )
    w_v3 = apply_regime_mask(w_v3, disp)
    w_v3.index = pd.to_datetime(w_v3.index, utc=True); ret.index = w_v3.index

    # Compute the market regime mask at each SMA window
    masks = {}
    for n in SMA_GRID:
        m = market_regime_mask(full, n)
        masks[n] = m.reindex(w_v3.index, method="ffill").fillna(0)

    rows = []
    for label, (s, e) in WINDOWS_TO_REPORT.items():
        w_b = _slice(w_v3, s, e); r_b = _slice(ret, s, e)
        if len(w_b) == 0:
            continue
        # Baseline (no kill switch)
        bt = backtest_weights(w_b, r_b, cost_bps=1.0)
        m_base = portfolio_metrics(bt)
        rows.append({
            "window": label, "variant": "V3_no_killswitch",
            "sharpe": round(m_base["sharpe"], 3),
            "ann_pct": round(100 * m_base["ann_return"], 2),
            "maxdd_pct": round(100 * m_base["max_drawdown"], 2),
            "regime_on_pct": 100.0,
        })
        # With each SMA kill switch
        for n in SMA_GRID:
            mask = _slice(masks[n].to_frame("k"), s, e)["k"]
            mask_aligned = mask.reindex(w_b.index, method="ffill").fillna(0)
            w_k = w_b.mul(mask_aligned.astype(float), axis=0)
            bt_k = backtest_weights(w_k, r_b, cost_bps=1.0)
            m_k = portfolio_metrics(bt_k)
            rows.append({
                "window": label, "variant": f"V3_kill_sma{n}",
                "sharpe": round(m_k["sharpe"], 3),
                "ann_pct": round(100 * m_k["ann_return"], 2),
                "maxdd_pct": round(100 * m_k["max_drawdown"], 2),
                "regime_on_pct": round(100 * float(mask_aligned.mean()), 2),
            })

    df = pd.DataFrame(rows)
    print("\n=== SHARPE ===")
    print(df.pivot_table(index="variant", columns="window",
                          values="sharpe", aggfunc="first").to_string())
    print("\n=== ANN% ===")
    print(df.pivot_table(index="variant", columns="window",
                          values="ann_pct", aggfunc="first").to_string())
    print("\n=== MAXDD% ===")
    print(df.pivot_table(index="variant", columns="window",
                          values="maxdd_pct", aggfunc="first").to_string())
    print("\n=== REGIME ON % ===")
    print(df.pivot_table(index="variant", columns="window",
                          values="regime_on_pct", aggfunc="first").to_string())

    df.to_csv(OUT_DIR / "killswitch_results.csv", index=False)
    with open(OUT_DIR / "killswitch_results.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
