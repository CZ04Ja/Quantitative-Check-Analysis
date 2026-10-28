"""Sizing layers that wrap a target-weights matrix.

These functions take a (date x symbol) DataFrame of *raw* target weights (e.g.
output of `long_short_quantile_weights`) and return a *rescaled* DataFrame of
the same shape, applying portfolio-level discipline:

* `target_vol_sizing`        -- scale each row's gross leverage so that the
                                strategy's realised vol over a rolling window
                                lands near a desired annual vol.
* `cap_max_weight`            -- per-name absolute cap.
* `apply_turnover_penalty`    -- shrink today's weights toward yesterday's by
                                a configurable factor to discourage churn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cap_max_weight(weights: pd.DataFrame, max_w: float) -> pd.DataFrame:
    return weights.clip(lower=-max_w, upper=max_w)


def target_vol_sizing(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    target_ann_vol: float = 0.10,
    lookback: int = 63,
    max_leverage: float = 4.0,
) -> pd.DataFrame:
    """Scale each row of weights to target a constant annualised strategy vol.

    The realised strategy return at date t (using *yesterday's* weights) is
        r_t = (held[t-1] * asset_returns[t]).sum()
    We compute the rolling std of `r` over `lookback` bars, then scale the
    weights at date t by  (target_ann_vol / sqrt(252)) / rolling_daily_std.

    The scaling factor itself is shifted by one bar to avoid look-ahead.
    """
    common_idx = weights.index.intersection(asset_returns.index)
    common_cols = weights.columns.intersection(asset_returns.columns)
    w = weights.loc[common_idx, common_cols].fillna(0.0)
    r = asset_returns.loc[common_idx, common_cols].fillna(0.0)

    daily_strat = (w.shift(1) * r).sum(axis=1)
    rolling_std = daily_strat.rolling(lookback, min_periods=max(10, lookback // 3)).std(ddof=0)
    target_daily = target_ann_vol / np.sqrt(252)
    scale = (target_daily / rolling_std.replace(0, np.nan)).clip(upper=max_leverage)
    scale = scale.shift(1).fillna(1.0)
    return w.mul(scale, axis=0)


def apply_turnover_penalty(
    weights: pd.DataFrame,
    shrink: float = 0.5,
) -> pd.DataFrame:
    """Convex combination with yesterday's weights to dampen churn.

    `shrink=0` -> no smoothing (use raw target).
    `shrink=1` -> never trade.
    """
    if shrink <= 0:
        return weights
    if shrink >= 1:
        return weights * 0.0
    smoothed = weights.copy() * 0.0
    prev = pd.Series(0.0, index=weights.columns)
    for date, row in weights.iterrows():
        new = (1 - shrink) * row.fillna(0.0) + shrink * prev
        smoothed.loc[date] = new
        prev = new
    return smoothed
