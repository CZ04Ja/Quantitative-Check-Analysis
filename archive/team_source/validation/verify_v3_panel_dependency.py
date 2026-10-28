"""
Verify whether V3's 2024 OOS Sharpe is panel-size-dependent (same pathology
as V2's original overstatement).

Test 4 panel constructions, holding the strategy frozen at V3 settings:

    A) panel = 2024 + IS + Q2                           (the original test)
    B) panel = 2023 H2 + 2024 + IS + Q2                 (add 6 months pre)
    C) panel = 2022 + 2023 + 2024 + IS + Q2            (add 2 years pre)
    D) panel = 2020 H2 + 2022 + 2023 + 2024 + IS + Q2  (max we have)

For each panel, evaluate V3 on the 2024 slice ONLY. If Sharpe varies
materially across panels, V3's apparent OOS edge is not robust.
"""
from __future__ import annotations

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

TARGETS = ["AAPL", "AMD", "MSFT", "NVDA"]; PROXY = "XLK"

PARAMS = dict(
    signal_horizon=30, ts_window=390,
    z_in=2.75, z_out_ratio=0.2, hold_bars=120,
    dispersion_fast=5, dispersion_slow=60,
    gross=1.0,
)

OUT_DIR = ROOT / "data_qtc" / "validation"


def state_machine_long_only(z, z_in=2.75, z_out_ratio=0.2,
                              hold_bars=120, gross=1.0):
    T, N = z.shape
    positions = np.zeros((T, N), dtype=float)
    sz = z.fillna(0).values
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
    w = pd.DataFrame(positions, index=z.index, columns=z.columns)
    na = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(na, axis=0).mul(gross).fillna(0)


def _load(p):
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = df["symbol"].astype(str)
    return df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]


PARTS = {
    "BACK_OOS_2024": ROOT / "data_qtc" / "oos" / "back_oos_2024.parquet",
    "RESEARCH_PANEL": ROOT / "data_qtc" / "combined" / "research_panel.parquet",
    "FWD_Q2": ROOT / "data_qtc" / "oos" / "fwd_oos_2026q2.parquet",
    "POST_BEAR_2023": ROOT / "data_qtc" / "stress" / "POST_BEAR_2023.parquet",
    "BANKING_2023": ROOT / "data_qtc" / "stress" / "BANKING_2023.parquet",
    "TECH_BEAR_2022": ROOT / "data_qtc" / "stress" / "TECH_BEAR_2022.parquet",
    "POST_COVID": ROOT / "data_qtc" / "stress" / "POST_COVID.parquet",
}


def make_panel(parts):
    dfs = [_load(PARTS[p]) for p in parts]
    full = pd.concat(dfs, ignore_index=True)
    full = (full.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
            .sort_values(["symbol", "timestamp"]).reset_index(drop=True))
    full["ret_1m"] = full.groupby("symbol", sort=False)["close"].pct_change()
    full["rolling_vol"] = (
        full.groupby("symbol", sort=False)["ret_1m"]
        .transform(lambda s: s.abs().rolling(20, min_periods=5).mean())
    )
    return full


def run_v3(panel):
    sub = panel[panel["symbol"].isin(TARGETS + [PROXY])].copy()
    wp = build_wide_panels(sub, TARGETS, PROXY)
    ret = wp["ret_1m"]; proxy = wp["proxy_ret_1m"]
    resid, _ = residualize(ret, proxy, window=PARAMS["ts_window"])
    sig = resid.rolling(PARAMS["signal_horizon"]).sum()
    z = ts_zscore(sig, window=PARAMS["ts_window"])
    disp = dispersion_regime_filter(
        resid, PARAMS["dispersion_fast"], PARAMS["dispersion_slow"]
    )
    w = state_machine_long_only(
        z, z_in=PARAMS["z_in"], z_out_ratio=PARAMS["z_out_ratio"],
        hold_bars=PARAMS["hold_bars"], gross=PARAMS["gross"],
    )
    w = apply_regime_mask(w, disp)
    w.index = pd.to_datetime(w.index, utc=True); ret.index = w.index
    return w, ret, disp


def slice_(df, s, e):
    s_ts = pd.Timestamp(s, tz="UTC"); e_ts = pd.Timestamp(e, tz="UTC")
    return df.loc[(df.index >= s_ts) & (df.index < e_ts)]


CONFIGS = {
    "A_2024_only":         ["BACK_OOS_2024", "RESEARCH_PANEL", "FWD_Q2"],
    "B_with_2023H2":       ["POST_BEAR_2023", "BACK_OOS_2024", "RESEARCH_PANEL", "FWD_Q2"],
    "C_with_2022_2023":    ["TECH_BEAR_2022", "BANKING_2023", "POST_BEAR_2023",
                            "BACK_OOS_2024", "RESEARCH_PANEL", "FWD_Q2"],
    "D_with_post_covid":   ["POST_COVID", "TECH_BEAR_2022", "BANKING_2023",
                            "POST_BEAR_2023", "BACK_OOS_2024", "RESEARCH_PANEL", "FWD_Q2"],
}

EVAL_WINDOWS = {
    "BACK_OOS_2024":  ("2024-01-01", "2024-12-31"),
    "IS_2025_2026Q1": ("2025-01-01", "2026-03-31"),
    "FWD_OOS_2026Q2": ("2026-04-01", "2027-01-01"),
}


def main():
    print("\n=== V3 SHARPE ON FROZEN WINDOWS, BY PANEL CONSTRUCTION ===\n")
    results = {win: {} for win in EVAL_WINDOWS}
    gate_results = {win: {} for win in EVAL_WINDOWS}
    for cfg_name, parts in CONFIGS.items():
        print(f"--- panel = {cfg_name}  ({len(parts)} parts) ---")
        full = make_panel(parts)
        w, ret, disp = run_v3(full)
        for label, (s, e) in EVAL_WINDOWS.items():
            w_s = slice_(w, s, e); r_s = slice_(ret, s, e)
            if len(w_s) == 0:
                continue
            disp_s = slice_(disp.to_frame("g"), s, e)["g"]
            bt = backtest_weights(w_s, r_s, cost_bps=1.0)
            m = portfolio_metrics(bt)
            results[label][cfg_name] = round(m["sharpe"], 3)
            gate_results[label][cfg_name] = round(100 * float(disp_s.fillna(False).mean()), 2)
            print(f"  {label}: Sharpe={m['sharpe']:+.3f}  "
                  f"ann={100*m['ann_return']:+.2f}%  "
                  f"maxDD={100*m['max_drawdown']:.2f}%  "
                  f"gate_on={100 * float(disp_s.fillna(False).mean()):.2f}%")
        print()

    print("\n=== SUMMARY: SHARPE BY (window × panel) ===")
    print(pd.DataFrame(results).T.to_string())
    print("\n=== SUMMARY: GATE_ON % BY (window × panel) ===")
    print(pd.DataFrame(gate_results).T.to_string())


if __name__ == "__main__":
    main()
