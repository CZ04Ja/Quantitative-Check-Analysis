"""
Diagnose why V1 + z_in=2.75 lost in 2026 Q2.

Hypotheses to test:
    H1) The 2026 Q2 dispersion regime is genuinely different (lower
        dispersion, gate too open or too closed).
    H2) One specific symbol dominates the loss (single-name failure).
    H3) Trade duration is collapsing (entries unwound too fast).
    H4) Cost-share is anomalously high (turnover up).
    H5) Specific intraday timing (open chaos / close fades) is the
        bleed source.
"""
from __future__ import annotations

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

    def _load(p):
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
    full = (full.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
            .sort_values(["symbol", "timestamp"]).reset_index(drop=True))
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
    w_raw = sparse_reversion_state_machine(
        z, z_in=PARAMS["z_in"], z_out_ratio=PARAMS["z_out_ratio"],
        hold_bars=PARAMS["hold_bars"], gross=PARAMS["gross"],
    )
    disp_mask = dispersion_regime_filter(
        resid, PARAMS["dispersion_fast"], PARAMS["dispersion_slow"]
    )
    w = apply_regime_mask(w_raw, disp_mask)
    w.index = pd.to_datetime(w.index, utc=True); ret.index = w.index
    resid.index = w.index; z.index = w.index

    print("=" * 70)
    print("DIAGNOSTICS — V1 + z_in=2.75, by window")
    print("=" * 70)

    for label, (s, e) in WINDOWS.items():
        print(f"\n--- {label} ---")
        w_s = _slice(w, s, e); r_s = _slice(ret, s, e)
        z_s = _slice(z, s, e); resid_s = _slice(resid, s, e)
        gate_s = _slice(disp_mask.to_frame("g"), s, e)["g"]

        if len(w_s) == 0:
            print("  empty"); continue

        # H1: gate stats
        gate_on = float(gate_s.fillna(False).mean())
        # cross-sectional dispersion (mean of per-stock 5d std of resid)
        roll_std = resid_s.rolling(390 * 5).std(ddof=0).mean(axis=1)
        disp_mean = float(roll_std.mean()) * 1e4  # in bps
        print(f"  H1 regime: gate_on_pct={gate_on*100:.2f}%  "
              f"disp_mean={disp_mean:.2f} bps")

        # |z| distribution
        abs_z = z_s.abs().stack().dropna()
        if len(abs_z) > 0:
            print(f"     |z| dist: p50={abs_z.median():.2f}  "
                  f"p90={abs_z.quantile(0.90):.2f}  "
                  f"p99={abs_z.quantile(0.99):.2f}  "
                  f"max={abs_z.max():.2f}  "
                  f"frac_z>2.75={(abs_z>2.75).mean()*100:.2f}%")

        # H2: per-symbol PnL
        per_sym_pnl = (w_s.shift(1) * r_s).sum(axis=0)
        per_sym_turnover = (w_s - w_s.shift(1)).abs().sum(axis=0)
        per_sym_cost = per_sym_turnover * 1.0 / 1e4
        per_sym_net = per_sym_pnl - per_sym_cost
        print(f"  H2 per-symbol net PnL (units = sum bar returns):")
        for c in per_sym_net.index:
            print(f"     {c}: gross={per_sym_pnl[c]*1e4:+.2f}bps  "
                  f"cost={per_sym_cost[c]*1e4:.2f}bps  "
                  f"net={per_sym_net[c]*1e4:+.2f}bps")

        # H3: avg hold duration (in bars)
        in_pos = (w_s != 0)
        durations = []
        for col in in_pos.columns:
            arr = in_pos[col].values
            cur = 0
            for v in arr:
                if v: cur += 1
                else:
                    if cur > 0: durations.append(cur)
                    cur = 0
            if cur > 0: durations.append(cur)
        if durations:
            d = np.array(durations)
            print(f"  H3 hold duration: n={len(d)}  mean={d.mean():.1f} bars  "
                  f"median={np.median(d):.1f}  max={d.max()}  "
                  f"hit_max={(d==PARAMS['hold_bars']).sum()}/{len(d)}")

        # H4: cost share
        bt = backtest_weights(w_s, r_s, cost_bps=1.0)
        m = portfolio_metrics(bt)
        print(f"  H4 cost: gross_pnl={bt['gross_pnl'].sum()*1e4:+.2f}bps  "
              f"cost={bt['cost'].sum()*1e4:.2f}bps  "
              f"cost_share={m['cost_share_of_gross']:.3f}  "
              f"daily_turnover={m['avg_daily_turnover_x']:.1f}x")

        # H5: intraday breakdown — bin by minute-of-day NY
        bt_h5 = bt.copy()
        bt_h5["min_ny"] = bt_h5.index.tz_convert("America/New_York").hour * 60 + \
                          bt_h5.index.tz_convert("America/New_York").minute
        bins = [(570, 600, "open30"), (600, 945, "midday"),
                (945, 960, "close15")]
        for lo, hi, name in bins:
            mask = (bt_h5["min_ny"] >= lo) & (bt_h5["min_ny"] < hi)
            seg = bt_h5.loc[mask]
            if len(seg) == 0: continue
            p = seg["net_pnl"].sum() * 1e4
            n = len(seg)
            sd = seg["net_pnl"].std(ddof=0)
            sh = seg["net_pnl"].mean()/sd*np.sqrt(390*252) if sd > 0 else 0.0
            print(f"  H5 {name:>8s}: bars={n:>5d}  net={p:+.2f}bps  "
                  f"slice_sharpe={sh:+.2f}")


if __name__ == "__main__":
    main()
