"""Factor timing -- dynamic per-factor weights based on recent performance.

Look-ahead protection
---------------------
`factor_returns` here is a series where row t is the long-short PnL realised
between t and t+1 (because the underlying forward return was `close[t+1]/close[t]-1`).
Therefore any rolling statistic ending AT date t already peeks at t+1's return.
All score series are explicitly `.shift(1)`-ed before being used as weights so
that the value used for trading at date t depends only on data observable at
the close of t-1.

If you supply a panel that is already point-in-time aligned (PnL[t] uses only
data up to day t), pass `pnl_already_pit=True` to skip the safety shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    mu = returns.rolling(window).mean()
    sd = returns.rolling(window).std(ddof=0).replace(0, np.nan)
    return mu / sd


def factor_timing_weights(
    factor_returns: pd.DataFrame,        # date x factor (long-short PnL)
    factor_ic: pd.DataFrame | None = None,  # date x factor IC, optional
    method: str = "rolling_sharpe",
    lookback: int = 63,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    smooth_halflife: int = 10,
    pnl_already_pit: bool = False,
) -> pd.DataFrame:
    """Compute date x factor weights for a long-only-on-magnitude scheme.

    Methods
    -------
    "equal"          : constant 1/N weights
    "rolling_sharpe" : weight prop. to clipped(rolling Sharpe of factor PnL)
    "rolling_ic"     : weight prop. to clipped(rolling mean of factor IC)
    "momentum"       : weight prop. to clipped(rolling cum return of factor)
    "ic_signed"      : magnitude = |rolling Sharpe of |signed PnL|, sign is
                       fixed by sign(rolling IC over a longer window).
                       This lets persistently anti-IC factors contribute with
                       their sign FLIPPED rather than being clipped to zero.
                       Returned weights are SIGNED (sum-of-abs == 1) instead
                       of all-positive.
    """
    n = factor_returns.shape[1]

    if method == "equal":
        w = pd.DataFrame(1.0 / n, index=factor_returns.index, columns=factor_returns.columns)
        return w

    # --- compute raw score, all of which look at t-included history ---
    if method == "rolling_sharpe":
        score = factor_returns.apply(lambda s: _rolling_sharpe(s, lookback))
    elif method == "rolling_ic":
        if factor_ic is None:
            raise ValueError("factor_ic required for method='rolling_ic'")
        score = factor_ic.rolling(lookback).mean()
    elif method == "momentum":
        score = factor_returns.rolling(lookback).sum()
    elif method == "ic_signed":
        if factor_ic is None:
            raise ValueError("factor_ic required for method='ic_signed'")
        long_ic = factor_ic.rolling(max(lookback, 252)).mean()
        sign = np.sign(long_ic).replace(0, np.nan)
        signed_pnl = factor_returns.mul(sign)
        magnitude = signed_pnl.apply(lambda s: _rolling_sharpe(s, lookback))
        score = magnitude.mul(sign)
    else:
        raise ValueError(method)

    # --- mandatory point-in-time guard: weights at t may NOT use info from t.
    if not pnl_already_pit:
        score = score.shift(1)

    if method == "ic_signed":
        # Signed weights, normalised so sum-of-abs == 1.
        raw = score.fillna(0.0)
        row_abs = raw.abs().sum(axis=1).replace(0, np.nan)
        w = raw.div(row_abs, axis=0).fillna(0.0)
    else:
        raw = score.clip(lower=min_weight, upper=max_weight).fillna(0.0)
        row_sum = raw.sum(axis=1).replace(0, np.nan)
        w = raw.div(row_sum, axis=0).fillna(1.0 / n)

    if smooth_halflife and smooth_halflife > 0:
        w = w.ewm(halflife=smooth_halflife, adjust=False).mean()
    return w
