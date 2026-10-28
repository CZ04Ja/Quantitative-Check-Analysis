"""Volume / liquidity features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def dollar_volume(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    return close * volume


def volume_zscore(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volume z-score using a rolling baseline (per symbol)."""
    mu = volume.rolling(window=window, min_periods=max(5, window // 2)).mean()
    sd = volume.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=0)
    return (volume - mu) / sd.replace(0, np.nan)


def amihud_illiquidity(close: pd.DataFrame, volume: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    r"""Amihud (2002) illiquidity = mean( |return_t| / dollar_vol_t ) over `window`.

    A higher value means a less liquid stock.
    """
    ret = (close / close.shift(1) - 1.0).abs()
    dv = (close * volume).replace(0, np.nan)
    raw = (ret / dv) * 1e9  # rescale for human-readable magnitudes
    return raw.rolling(window=window, min_periods=max(5, window // 2)).mean()


def turnover_proxy(volume: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """No shares-outstanding feed on Alpaca free tier, so we proxy turnover by
    today's volume relative to its trailing rolling-window mean."""
    base = volume.rolling(window=window, min_periods=max(5, window // 2)).mean()
    return volume / base.replace(0, np.nan)
