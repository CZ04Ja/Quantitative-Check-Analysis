"""Price-based features. Inputs are wide panels (date x symbol)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_return(close: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return np.log(close / close.shift(periods))


def cumulative_return(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Total simple return over the past `window` bars (excluding today)."""
    return close.shift(1) / close.shift(window + 1) - 1.0


def moving_average(close: pd.DataFrame, window: int) -> pd.DataFrame:
    return close.rolling(window=window, min_periods=max(2, window // 2)).mean()


def rolling_volatility(close: pd.DataFrame, window: int, annualize: bool = True) -> pd.DataFrame:
    r = np.log(close / close.shift(1))
    vol = r.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=0)
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def rolling_skew(close: pd.DataFrame, window: int) -> pd.DataFrame:
    r = np.log(close / close.shift(1))
    return r.rolling(window=window, min_periods=max(5, window // 2)).skew()


def rolling_kurt(close: pd.DataFrame, window: int) -> pd.DataFrame:
    r = np.log(close / close.shift(1))
    return r.rolling(window=window, min_periods=max(5, window // 2)).kurt()


def distance_from_high(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """1 - close / rolling_max(close, window) ... 0 means at the high."""
    hi = close.rolling(window=window, min_periods=max(5, window // 2)).max()
    return 1.0 - close / hi


def distance_from_low(close: pd.DataFrame, window: int) -> pd.DataFrame:
    lo = close.rolling(window=window, min_periods=max(5, window // 2)).min()
    return close / lo - 1.0


def rolling_max_drawdown(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Worst drawdown observed inside a rolling window."""
    def _mdd(x: np.ndarray) -> float:
        cm = np.maximum.accumulate(x)
        dd = x / cm - 1.0
        return float(dd.min())
    return close.rolling(window=window, min_periods=max(5, window // 2)).apply(_mdd, raw=True)
