"""Liquidity factors. Both classic (Amihud) and a turnover-momentum twist."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import REGISTRY


@REGISTRY.register(
    name="amihud_illiq_21",
    family="liquidity",
    description=(
        "Amihud (2002) illiquidity, 21d window. Higher = less liquid = "
        "illiquidity premium (long signal in cross section)."
    ),
    default_params={"window": 21},
)
def amihud_21(panels, window: int = 21) -> pd.DataFrame:
    c = panels["close"]
    v = panels["volume"]
    ret = (c / c.shift(1) - 1.0).abs()
    dv = (c * v).replace(0, np.nan)
    return (ret / dv * 1e9).rolling(window).mean()


@REGISTRY.register(
    name="dv_momentum_21",
    family="liquidity",
    description=(
        "Dollar-volume momentum: today's 21d mean dollar volume vs 252d baseline. "
        "Captures stocks gathering attention."
    ),
    default_params={"short": 21, "long": 252},
)
def dv_momentum(panels, short: int = 21, long: int = 252) -> pd.DataFrame:
    c = panels["close"]
    v = panels["volume"]
    dv = c * v
    s = dv.rolling(short).mean()
    l = dv.rolling(long).mean().replace(0, np.nan)
    return s / l - 1.0


@REGISTRY.register(
    name="turnover_zscore_63",
    family="liquidity",
    description="Z-score of today's volume relative to its 63d distribution.",
    default_params={"window": 63},
)
def turnover_z(panels, window: int = 63) -> pd.DataFrame:
    v = panels["volume"]
    mu = v.rolling(window).mean()
    sd = v.rolling(window).std(ddof=0).replace(0, np.nan)
    return (v - mu) / sd
