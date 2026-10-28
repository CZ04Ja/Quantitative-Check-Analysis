"""Microstructure-flavoured factors derived from O/H/L/C only.

Even without intraday data we can split each daily bar into an *overnight*
component (close[t-1] -> open[t]) and an *intraday* component
(open[t] -> close[t]). A wealth of well-documented effects sit here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import REGISTRY


@REGISTRY.register(
    name="overnight_mom_21",
    family="microstructure",
    description=(
        "Sum of 21-day overnight returns. Lou-Polk-Skouras (2019) show overnight "
        "and intraday returns are nearly orthogonal premia."
    ),
    default_params={"window": 21},
)
def overnight_mom(panels, window: int = 21) -> pd.DataFrame:
    o = panels["open"]
    c = panels["close"]
    overnight = np.log(o / c.shift(1))
    return overnight.rolling(window).sum()


@REGISTRY.register(
    name="intraday_mom_21",
    family="microstructure",
    description="Sum of 21-day intraday returns (open->close).",
    default_params={"window": 21},
)
def intraday_mom(panels, window: int = 21) -> pd.DataFrame:
    o = panels["open"]
    c = panels["close"]
    intraday = np.log(c / o)
    return intraday.rolling(window).sum()


@REGISTRY.register(
    name="overnight_minus_intraday",
    family="microstructure",
    description=(
        "21-day overnight return minus intraday return. Captures the cross-sectional "
        "tilt toward overnight-driven names (often higher-quality / passive flows)."
    ),
    default_params={"window": 21},
)
def overnight_vs_intraday(panels, window: int = 21) -> pd.DataFrame:
    o = panels["open"]
    c = panels["close"]
    overnight = np.log(o / c.shift(1)).rolling(window).sum()
    intraday = np.log(c / o).rolling(window).sum()
    return overnight - intraday


@REGISTRY.register(
    name="range_compression_21",
    family="microstructure",
    description=(
        "Mean intraday range (high-low)/close over 21d, NEGATED. Stocks with "
        "compressing intraday range tend to have lower realized vol and benefit "
        "from the low-vol anomaly."
    ),
    default_params={"window": 21},
)
def range_compression(panels, window: int = 21) -> pd.DataFrame:
    h = panels["high"]
    l = panels["low"]
    c = panels["close"]
    rng = (h - l) / c.replace(0, np.nan)
    return -rng.rolling(window).mean()
