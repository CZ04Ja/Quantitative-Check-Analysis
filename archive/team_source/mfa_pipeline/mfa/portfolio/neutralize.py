"""Cross-sectional neutralisation helpers.

`market_beta_neutralize_signal`
    For each date, regress the row of signal values on the per-name market
    beta (estimated from a rolling regression of name returns vs the benchmark)
    and return the residuals. This removes the part of the signal that simply
    bets on high-beta vs low-beta names.

The resulting "neutral" signal can be passed to any sizing function unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_beta(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """Per-name rolling beta vs the benchmark over `window` bars.

    All values are point-in-time: beta[t] uses returns up to and including t.
    Apply `.shift(1)` before using as a regressor at date t to avoid look-ahead.
    """
    aligned = asset_returns.join(benchmark_returns.rename("__bm"), how="inner")
    bm = aligned["__bm"]
    rets = aligned.drop(columns="__bm")

    cov = rets.rolling(window).cov(bm)
    var = bm.rolling(window).var().replace(0, np.nan)
    beta = cov.div(var, axis=0)
    return beta


def market_beta_neutralize_signal(
    signal: pd.DataFrame,
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    beta_window: int = 252,
) -> pd.DataFrame:
    """Residualise the cross-sectional signal against name betas.

    For each date t (with at least 5 valid betas), solve the per-row OLS:
        signal_t = a + b * beta_t + e_t
    and return e_t in place of the signal. This makes the signal mechanically
    uncorrelated with market-beta exposure on that date.
    """
    beta = rolling_beta(asset_returns, benchmark_returns, window=beta_window).shift(1)
    common_idx = signal.index.intersection(beta.index)
    common_cols = signal.columns.intersection(beta.columns)
    s = signal.loc[common_idx, common_cols].copy()
    b = beta.loc[common_idx, common_cols]

    out = pd.DataFrame(index=common_idx, columns=common_cols, dtype="float64")
    for date in common_idx:
        y = s.loc[date].values
        x = b.loc[date].values
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 5:
            out.loc[date] = s.loc[date]
            continue
        x1 = x[mask]
        y1 = y[mask]
        X = np.column_stack([np.ones(len(x1)), x1])
        coef, *_ = np.linalg.lstsq(X, y1, rcond=None)
        resid = np.full_like(y, np.nan, dtype=float)
        resid[mask] = y1 - X @ coef
        out.loc[date] = resid
    return out
