"""'Quality' factors expressible from price/volume only.

Without fundamentals (Alpaca free tier doesn't ship balance-sheet data) we
proxy quality by the *smoothness* and *consistency* of price action:
companies that compound steadily without violent shocks tend to outperform
risk-adjusted in the cross section.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import REGISTRY


@REGISTRY.register(
    name="trend_smoothness_63",
    family="quality",
    description=(
        "Trend smoothness = |cumulative return| / sum(|daily return|) over 63d. "
        "Range (0, 1]. Pure trends -> 1, choppy noise -> 0."
    ),
    default_params={"window": 63},
)
def trend_smoothness(panels, window: int = 63) -> pd.DataFrame:
    c = panels["close"]
    r = np.log(c / c.shift(1))
    cum = r.rolling(window).sum().abs()
    pathlen = r.abs().rolling(window).sum().replace(0, np.nan)
    return cum / pathlen


@REGISTRY.register(
    name="up_day_ratio_63",
    family="quality",
    description="Fraction of up days in trailing 63d. Higher = more consistent rally.",
    default_params={"window": 63},
)
def up_day_ratio(panels, window: int = 63) -> pd.DataFrame:
    c = panels["close"]
    r = c / c.shift(1) - 1.0
    return (r > 0).rolling(window).mean()


@REGISTRY.register(
    name="return_consistency_63",
    family="quality",
    description="mean(return) / std(return) over 63d (per-symbol Sharpe).",
    default_params={"window": 63},
)
def return_sharpe(panels, window: int = 63) -> pd.DataFrame:
    c = panels["close"]
    r = np.log(c / c.shift(1))
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std(ddof=0).replace(0, np.nan)
    return mu / sd
