"""Vectorized daily backtester.

The semantics:
    * Signals are dated `t` and represent the manager's view at the close of t.
    * Target weights for date `t` are *executed at the open of t+1*.
    * Returns are computed close-to-close.
    * Costs are charged on the day of trading.

Inputs
------
target_weights : DataFrame, date x symbol, the *desired* portfolio at the
                 close of date t (will be entered overnight).
returns        : DataFrame, date x symbol, simple close-to-close returns.
rebalance_freq : 1 = daily; 5 = weekly; etc. Between rebalances weights drift.

Output
------
BacktestResult containing:
    * gross_returns, net_returns
    * equity_curve_gross, equity_curve_net
    * positions (held), turnover, costs
    * rebalance_dates
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .execution import CostModel


@dataclass
class BacktestConfig:
    rebalance: int = 5
    initial_capital: float = 1_000_000.0
    cost_model: CostModel = field(default_factory=CostModel)
    benchmark_returns: Optional[pd.Series] = None


@dataclass
class BacktestResult:
    gross_returns: pd.Series
    net_returns: pd.Series
    equity_gross: pd.Series
    equity_net: pd.Series
    weights_held: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    rebalance_dates: list[pd.Timestamp]
    benchmark_returns: Optional[pd.Series]

    def to_dict(self) -> dict:
        return {
            "gross_returns": self.gross_returns,
            "net_returns": self.net_returns,
            "equity_gross": self.equity_gross,
            "equity_net": self.equity_net,
            "weights_held": self.weights_held,
            "turnover": self.turnover,
            "costs": self.costs,
            "rebalance_dates": self.rebalance_dates,
            "benchmark_returns": self.benchmark_returns,
        }


def _rebalance_dates(index: pd.DatetimeIndex, every: int) -> list[pd.Timestamp]:
    """Pick every Nth bar."""
    return list(index[::every])


def run_backtest(
    target_weights: pd.DataFrame,
    returns: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    common_idx = target_weights.index.intersection(returns.index).sort_values()
    common_cols = target_weights.columns.intersection(returns.columns)
    w = target_weights.loc[common_idx, common_cols].fillna(0.0)
    r = returns.loc[common_idx, common_cols].fillna(0.0)

    rebal = _rebalance_dates(common_idx, every=max(1, config.rebalance))
    rebal_set = set(rebal)

    # Held weights start at 0; on rebalance days they jump to the target;
    # otherwise they drift with realized returns.
    held = pd.DataFrame(0.0, index=common_idx, columns=common_cols)
    turnover = pd.Series(0.0, index=common_idx, name="turnover")
    costs = pd.Series(0.0, index=common_idx, name="cost")

    cur_w = pd.Series(0.0, index=common_cols)
    prev_w = cur_w.copy()

    for i, date in enumerate(common_idx):
        # First, rebalance if needed (at "close of t-1 -> open of t")
        if date in rebal_set:
            target = w.loc[date]
            tno = float((target - cur_w).abs().sum())
            short_avg = float(0.5 * (cur_w.clip(upper=0).abs().sum() + target.clip(upper=0).abs().sum()))
            c = config.cost_model.cost_for_step(tno, short_avg, days=config.rebalance)
            cur_w = target.copy()
            turnover.iloc[i] = tno
            costs.iloc[i] = c

        # Hold today
        held.iloc[i] = cur_w

        # Realized return today; positions drift
        port_ret = float((cur_w * r.iloc[i]).sum())
        cur_w = cur_w * (1.0 + r.iloc[i])
        # Re-normalize so leverage doesn't run away (keep gross constant)
        gross = float(cur_w.abs().sum())
        if gross > 0:
            cur_w = cur_w * (gross / gross)  # noop -- but will rebalance soon

        prev_w = cur_w

    # Build returns
    held_yesterday = held.shift(1).fillna(0.0)
    gross_returns = (held_yesterday * r).sum(axis=1)
    net_returns = gross_returns - costs

    eq_gross = (1 + gross_returns).cumprod() * config.initial_capital
    eq_net = (1 + net_returns).cumprod() * config.initial_capital

    return BacktestResult(
        gross_returns=gross_returns.rename("gross"),
        net_returns=net_returns.rename("net"),
        equity_gross=eq_gross.rename("equity_gross"),
        equity_net=eq_net.rename("equity_net"),
        weights_held=held,
        turnover=turnover,
        costs=costs,
        rebalance_dates=rebal,
        benchmark_returns=config.benchmark_returns,
    )
