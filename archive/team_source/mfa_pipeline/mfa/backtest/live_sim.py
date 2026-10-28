"""'Paper-trading' simulator.

This is a thin wrapper over the historical engine that walks forward one
trading day at a time and re-fits the factor combination weights using only
data available *up to the previous bar*. Useful as a stronger out-of-sample
sanity check than a single train/test split.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .engine import BacktestConfig, BacktestResult, run_backtest


@dataclass
class LiveSimResult:
    backtest: BacktestResult
    live_signal: pd.DataFrame
    factor_weights_history: pd.DataFrame


def simulate_paper_trading(
    factors: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    *,
    weight_fn: Callable[[dict[str, pd.DataFrame], pd.DatetimeIndex], pd.DataFrame],
    sizing_fn: Callable[[pd.DataFrame], pd.DataFrame],
    config: BacktestConfig,
    refit_every: int = 21,
    warmup: int = 252,
) -> LiveSimResult:
    """Walk-forward simulation.

    weight_fn(factors_so_far, dates_so_far) -> per-date factor weights for the
        most recent date in `dates_so_far`. Called every `refit_every` bars.
    sizing_fn(composite_signal_panel) -> target_weights (date x symbol).
    """
    common_idx = returns.index.sort_values()
    if len(common_idx) <= warmup:
        raise RuntimeError("Not enough history for warmup.")
    sim_idx = common_idx[warmup:]

    factor_names = list(factors.keys())
    fw_history = pd.DataFrame(index=sim_idx, columns=factor_names, dtype="float64")

    last_weights: pd.Series | None = None
    for i, date in enumerate(sim_idx):
        if i % refit_every == 0:
            dates_so_far = common_idx[: warmup + i + 1]
            facts_slice = {n: f.loc[f.index <= date] for n, f in factors.items()}
            wts = weight_fn(facts_slice, dates_so_far)  # DataFrame date x factor
            last_weights = wts.iloc[-1] if not wts.empty else pd.Series(1.0 / len(factor_names),
                                                                        index=factor_names)
        fw_history.loc[date] = last_weights

    # Build composite signal *out-of-sample*: each date uses ONLY weights known
    # at that date (which were set at the most recent refit boundary).
    cols = sorted(set().union(*[f.columns for f in factors.values()]))
    composite = pd.DataFrame(0.0, index=sim_idx, columns=cols)
    for n, f in factors.items():
        f_aligned = f.reindex(index=sim_idx, columns=cols)
        composite = composite + f_aligned.mul(fw_history[n], axis=0)

    target = sizing_fn(composite)
    bt = run_backtest(target, returns.loc[sim_idx], config)
    return LiveSimResult(backtest=bt, live_signal=composite, factor_weights_history=fw_history)
