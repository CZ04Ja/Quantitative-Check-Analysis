"""Short-term reversal family.

Sign convention: higher factor value -> long. Since reversal means losers
outperform winners next period, we flip the sign of the past return so that
a *low* past return becomes a *high* factor value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import REGISTRY


@REGISTRY.register(
    name="rev_1",
    family="reversal",
    description="1-day short-term reversal (negate yesterday's return).",
)
def rev_1d(panels) -> pd.DataFrame:
    c = panels["close"]
    return -(c / c.shift(1) - 1.0)


@REGISTRY.register(
    name="rev_5",
    family="reversal",
    description="1-week short-term reversal.",
    default_params={"lookback": 5},
)
def rev_5d(panels, lookback: int = 5) -> pd.DataFrame:
    c = panels["close"]
    return -(c / c.shift(lookback) - 1.0)


@REGISTRY.register(
    name="gap_reversal",
    family="reversal",
    description=(
        "Yesterday's overnight gap (open/prev_close - 1) negated. "
        "Excessive gaps tend to fade intraday and over the following days."
    ),
)
def gap_reversal(panels) -> pd.DataFrame:
    o = panels["open"]
    c = panels["close"]
    gap = o / c.shift(1) - 1.0
    return -gap
