"""
Stress test V3 (long-only z=2.75 hold=120) across historical regimes.

Each stress window is run standalone with its own gate warm-up. We
report Sharpe, MaxDD, gate-on%, monthly breakdown, per-symbol
attribution, and the WORST SINGLE MONTH across all stress periods.

The single-most-important test is TECH_BEAR_2022: a full year of
slow-grind tech decline (XLK -33%). Long-only is structurally
exposed to this regime; if V3 doesn't preserve capital here, it
should be reconsidered.
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
COST_BPS = 1.0

STRESS_WINDOWS = {
    "POST_COVID":     ("2020-06-01", "2020-12-31"),
    "TECH_BEAR_2022": ("2022-01-01", "2022-12-31"),
    "BANKING_2023":   ("2023-02-15", "2023-04-30"),
    "POST_BEAR_2023": ("2023-05-01", "2023-12-31"),
}
OUT_DIR = ROOT / "data_qtc" / "validation"


def state_machine_long_only(signal_z, z_in=2.75, z_out_ratio=0.2,
                              hold_bars=120, gross=1.0):
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


def _load(p):
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = df["symbol"].astype(str)
    return df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]


def load_window(label: str) -> pd.DataFrame:
    path = ROOT / "data_qtc" / "stress" / f"{label}.parquet"
    df = _load(path)
    df["ret_1m"] = df.groupby("symbol", sort=False)["close"].pct_change()
    df["rolling_vol"] = (
        df.groupby("symbol", sort=False)["ret_1m"]
        .transform(lambda s: s.abs().rolling(20, min_periods=5).mean())
    )
    return df


def block_bootstrap_sharpe(pnl, block_bars=390 * 2, n_iter=2000, seed=7):
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
        "median": round(float(np.median(out)), 3),
        "p05": round(float(np.percentile(out, 5)), 3),
        "p95": round(float(np.percentile(out, 95)), 3),
        "p_gt_0": round(float((out > 0).mean()), 3),
    }


def evaluate_window(label: str, panel: pd.DataFrame, win_start: str,
                    win_end: str) -> dict:
    sub = panel[panel["symbol"].isin(TARGETS + [PROXY])].copy()
    if sub.empty:
        return {"label": label, "note": "no data"}
    wp = build_wide_panels(sub, TARGETS, PROXY)
    ret = wp["ret_1m"]; proxy = wp["proxy_ret_1m"]
    if len(ret) < 25_740:
        warmup_ok = False
        print(f"[{label}] WARN: only {len(ret)} aligned bars — gate warm-up incomplete")
    else:
        warmup_ok = True
    resid, _ = residualize(ret, proxy, window=PARAMS["ts_window"])
    sig = resid.rolling(PARAMS["signal_horizon"]).sum()
    z = ts_zscore(sig, window=PARAMS["ts_window"])
    disp = dispersion_regime_filter(
        resid, PARAMS["dispersion_fast"], PARAMS["dispersion_slow"]
    )
    w = state_machine_long_only(
        z, PARAMS["z_in"], PARAMS["z_out_ratio"],
        PARAMS["hold_bars"], PARAMS["gross"]
    )
    w = apply_regime_mask(w, disp)
    w.index = pd.to_datetime(w.index, utc=True); ret.index = w.index

    bt = backtest_weights(w, ret, cost_bps=COST_BPS)
    m = portfolio_metrics(bt)
    boot = block_bootstrap_sharpe(bt["net_pnl"])
    sign = np.sign(w); prev = sign.shift(1).fillna(0)
    n_trades = int(((sign != 0) & (sign != prev)).sum().sum())
    per_sym_pnl = (w.shift(1) * ret).sum(axis=0)
    per_sym_pnl = {k: round(v * 1e4, 2) for k, v in per_sym_pnl.items()}
    per_sym_trades = ((sign != 0) & (sign != prev)).sum().to_dict()
    per_sym_trades = {k: int(v) for k, v in per_sym_trades.items()}
    n_aligned = len(w)
    n_days = max(n_aligned / 390.0, 1.0)

    # Monthly
    bt2 = bt.copy()
    bt2["ym"] = bt2.index.tz_convert("America/New_York").strftime("%Y-%m")
    grp = bt2.groupby("ym")
    monthly = []
    for ym, g in grp:
        if len(g) < 200:
            continue
        sd = g["net_pnl"].std(ddof=0)
        sh = g["net_pnl"].mean() / sd * np.sqrt(390 * 252) if sd > 0 else 0.0
        w_m = w.loc[g.index]
        sgn = np.sign(w_m); prv = sgn.shift(1).fillna(0)
        ntr = int(((sgn != 0) & (sgn != prv)).sum().sum())
        monthly.append({
            "month": ym,
            "bars": len(g),
            "sum_pnl_bps": round(g["net_pnl"].sum() * 1e4, 2),
            "sharpe": round(float(sh), 3),
            "trades": ntr,
        })

    # Underlying drawdown of XLK over the window (regime context)
    if "XLK" in panel["symbol"].values:
        xlk = panel[panel["symbol"] == "XLK"].sort_values("timestamp")
        xlk_close = xlk["close"]
        if len(xlk_close) > 0:
            xlk_dd = (xlk_close / xlk_close.cummax() - 1).min() * 100
            xlk_total = (xlk_close.iloc[-1] / xlk_close.iloc[0] - 1) * 100
        else:
            xlk_dd = 0; xlk_total = 0
    else:
        xlk_dd = 0; xlk_total = 0

    return {
        "label": label,
        "window": (win_start, win_end),
        "warmup_ok": warmup_ok,
        "aligned_bars": n_aligned,
        "trading_days": round(n_days, 1),
        "underlying_xlk_total_pct": round(xlk_total, 2),
        "underlying_xlk_maxdd_pct": round(xlk_dd, 2),
        "gate_on_pct": round(100 * float(disp.fillna(False).mean()), 2),
        "v3_sharpe_1bp": round(m["sharpe"], 3),
        "v3_ann_pct": round(100 * m["ann_return"], 2),
        "v3_maxdd_pct": round(100 * m["max_drawdown"], 2),
        "v3_calmar": round(m["calmar"], 3),
        "n_trades": n_trades,
        "trades_per_day": round(n_trades / n_days, 3),
        "trades_per_symbol": per_sym_trades,
        "gross_pnl_per_symbol_bps": per_sym_pnl,
        "bootstrap": boot,
        "monthly": monthly,
    }


def main():
    summary = []
    all_monthly = []
    for label, (s, e) in STRESS_WINDOWS.items():
        print(f"\n=== {label} ===")
        try:
            panel = load_window(label)
            row = evaluate_window(label, panel, s, e)
            summary.append(row)
            print(f"  XLK total: {row['underlying_xlk_total_pct']:+.2f}%  "
                  f"XLK maxDD: {row['underlying_xlk_maxdd_pct']:.2f}%")
            print(f"  V3 Sharpe (1bp): {row['v3_sharpe_1bp']:+.2f}  "
                  f"V3 ann: {row['v3_ann_pct']:+.2f}%  V3 maxDD: {row['v3_maxdd_pct']:.2f}%")
            print(f"  gate_on: {row['gate_on_pct']:.2f}%  "
                  f"trades: {row['n_trades']}  per_day: {row['trades_per_day']}")
            print(f"  per-symbol PnL (bps): {row['gross_pnl_per_symbol_bps']}")
            print(f"  bootstrap: {row['bootstrap']}")
            for mr in row["monthly"]:
                mr["window"] = label
                all_monthly.append(mr)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            summary.append({"label": label, "error": str(exc)})

    # Cross-window worst month
    if all_monthly:
        df = pd.DataFrame(all_monthly)
        print("\n=== WORST 5 MONTHS ACROSS STRESS WINDOWS ===")
        worst = df.sort_values("sum_pnl_bps").head(5)
        print(worst.to_string(index=False))
        print("\n=== BEST 5 MONTHS ACROSS STRESS WINDOWS ===")
        best = df.sort_values("sum_pnl_bps", ascending=False).head(5)
        print(best.to_string(index=False))
        print(f"\nTotal months active (with trades): "
              f"{(df['trades'] > 0).sum()} / {len(df)}")
        print(f"Months with positive Sharpe: "
              f"{(df['sharpe'] > 0).sum()} / {len(df)}")
        print(f"Median monthly Sharpe (active months only): "
              f"{df.loc[df['trades'] > 0, 'sharpe'].median():.3f}")
        print(f"Sum of monthly PnL: {df['sum_pnl_bps'].sum():+.2f} bps")
        df.to_csv(OUT_DIR / "stress_monthly.csv", index=False)

    with open(OUT_DIR / "stress_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Compact summary table
    print("\n=== STRESS SUMMARY ===")
    rows_print = []
    for s in summary:
        if "error" in s:
            continue
        rows_print.append({
            "label": s["label"],
            "underlying": f"{s['underlying_xlk_total_pct']:+.1f}%",
            "underlying_dd": f"{s['underlying_xlk_maxdd_pct']:.1f}%",
            "v3_sharpe": s["v3_sharpe_1bp"],
            "v3_ann": f"{s['v3_ann_pct']:+.2f}%",
            "v3_maxdd": f"{s['v3_maxdd_pct']:.2f}%",
            "trades": s["n_trades"],
            "gate_on%": s["gate_on_pct"],
        })
    print(pd.DataFrame(rows_print).to_string(index=False))


if __name__ == "__main__":
    main()
