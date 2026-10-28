"""Combine factors with IC-IR weighting (and optional factor timing), then run
the backtest. Also runs the walk-forward 'paper trading' simulation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mfa.config import load_config
from mfa.data import build_panels, forward_returns
from mfa.evaluation import information_coefficient
from mfa.evaluation.stats import perf_stats
from mfa.portfolio import (
    combine_factors, factor_timing_weights,
    long_short_quantile_weights,
)
from mfa.backtest import BacktestConfig, CostModel, run_backtest, simulate_paper_trading
from mfa.utils import get_logger

log = get_logger("scripts.combine")


def _load_panels(cfg) -> dict:
    out = {}
    for k in ["open", "high", "low", "close", "volume", "vwap", "dollar_volume"]:
        p = cfg.processed_dir / f"panel_{k}.parquet"
        if p.exists():
            out[k] = pd.read_parquet(p)
            out[k].index = pd.to_datetime(out[k].index, utc=True)
    return out


def _load_factors(cfg) -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted((cfg.processed_dir / "factors").glob("*.parquet")):
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True)
        out[p.stem] = df
    return out


def _factor_pnl_panel(factors: dict[str, pd.DataFrame], fwd: pd.DataFrame) -> pd.DataFrame:
    """Compute the daily long-short return of each factor (top - bottom quintile)
    so we can later use it for factor timing."""
    from mfa.evaluation.quantiles import quantile_returns, long_short_returns
    out = {}
    for n, f in factors.items():
        qr = quantile_returns(f, fwd, n_quantiles=5)
        out[n] = long_short_returns(qr)
    return pd.DataFrame(out)


def _factor_ic_panel(factors: dict[str, pd.DataFrame], fwd: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for n, f in factors.items():
        out[n] = information_coefficient(f, fwd)
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    bm = cfg.section("universe").get("benchmark", "SPY")
    panels = build_panels(_load_panels(cfg), benchmark=bm, min_history_days=252)
    factors = _load_factors(cfg)
    if not factors:
        raise RuntimeError("No factor files. Run 02_build_factors.py first.")

    # 1d simple returns for backtest
    close = panels["close"]
    daily_ret = close.pct_change().fillna(0.0)

    # ---------- Factor combination weights -------------------------------
    fwd1 = close.shift(-1) / close - 1.0  # next-day return
    factor_pnl = _factor_pnl_panel(factors, fwd1)
    factor_ic = _factor_ic_panel(factors, fwd1)

    timing_cfg = cfg.section("timing")
    if timing_cfg.get("enabled", True):
        f_weights = factor_timing_weights(
            factor_pnl,
            factor_ic=factor_ic,
            method=timing_cfg.get("signal", "rolling_sharpe"),
            lookback=int(timing_cfg.get("lookback", 63)),
            min_weight=float(timing_cfg.get("min_weight", 0.0)),
            max_weight=float(timing_cfg.get("max_weight", 1.0)),
        )
    else:
        n = len(factors)
        f_weights = pd.DataFrame(1.0 / n, index=daily_ret.index, columns=list(factors.keys()))

    composite = combine_factors(factors, weights=f_weights, standardize=True)

    # ---------- Sizing: long-short quantile portfolio --------------------
    pf_cfg = cfg.section("portfolio")
    target = long_short_quantile_weights(
        composite,
        top_q=float(pf_cfg.get("top_quantile", 0.2)),
        bot_q=float(pf_cfg.get("bottom_quantile", 0.2)),
        gross_leverage=float(pf_cfg.get("gross_leverage", 2.0)),
        max_weight=float(pf_cfg.get("max_weight", 0.10)),
    )

    # ---------- Backtest --------------------------------------------------
    bt_cfg = cfg.section("backtest")
    cm = CostModel(
        commission_bps=float(bt_cfg.get("commission_bps", 1.0)),
        slippage_bps=float(bt_cfg.get("slippage_bps", 2.0)),
        borrow_cost_bps_annual=float(bt_cfg.get("borrow_cost_bps_annual", 50.0)),
    )
    bm_close = panels.get("benchmark_close")
    bm_ret = bm_close.iloc[:, 0].pct_change().fillna(0.0) if bm_close is not None else None
    config = BacktestConfig(
        rebalance=int(pf_cfg.get("rebalance", 5)),
        initial_capital=float(bt_cfg.get("initial_capital", 1_000_000)),
        cost_model=cm,
        benchmark_returns=bm_ret,
    )
    bt = run_backtest(target, daily_ret, config)

    perf = perf_stats(bt.net_returns)
    log.info("Backtest perf: Sharpe=%.2f, ann.ret=%.2f%%, MDD=%.2f%%",
             perf["sharpe"], perf["ann_return"] * 100, perf["max_drawdown"] * 100)

    # Save
    out = cfg.results_root
    bt.equity_net.to_csv(out / "equity_net.csv")
    bt.equity_gross.to_csv(out / "equity_gross.csv")
    bt.net_returns.to_csv(out / "returns_net.csv")
    bt.gross_returns.to_csv(out / "returns_gross.csv")
    bt.weights_held.to_parquet(out / "weights_held.parquet")
    bt.turnover.to_csv(out / "turnover.csv")
    f_weights.to_parquet(out / "factor_weights.parquet")
    composite.to_parquet(out / "composite_signal.parquet")
    pd.Series(perf).to_csv(out / "perf_stats.csv")
    log.info("artifacts saved under %s", out)


if __name__ == "__main__":
    main()
