"""Volatility-based factors.

Low-volatility anomaly: low realized vol stocks have *higher* risk-adjusted
returns. We negate the raw vol so high factor value -> low vol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import REGISTRY


@REGISTRY.register(
    name="low_vol_21",
    family="volatility",
    description="Negative of 21-day realized vol (low-vol anomaly).",
    default_params={"window": 21},
)
def low_vol_21(panels, window: int = 21) -> pd.DataFrame:
    c = panels["close"]
    r = np.log(c / c.shift(1))
    vol = r.rolling(window).std(ddof=0) * np.sqrt(252)
    return -vol


@REGISTRY.register(
    name="low_vol_63",
    family="volatility",
    description="Negative of 63-day realized vol.",
    default_params={"window": 63},
)
def low_vol_63(panels, window: int = 63) -> pd.DataFrame:
    c = panels["close"]
    r = np.log(c / c.shift(1))
    vol = r.rolling(window).std(ddof=0) * np.sqrt(252)
    return -vol


@REGISTRY.register(
    name="idio_vol_63",
    family="volatility",
    description=(
        "Idiosyncratic vol = std of residuals from a 63d regression on the "
        "equal-weight market. Negated so low-idio-vol = long signal."
    ),
    default_params={"window": 63},
)
def idio_vol(panels, window: int = 63) -> pd.DataFrame:
    c = panels["close"]
    r = np.log(c / c.shift(1))
    mkt = r.mean(axis=1)
    cov = r.rolling(window).cov(mkt)
    var = mkt.rolling(window).var()
    beta = cov.div(var.replace(0, np.nan), axis=0)
    resid = r.sub(beta.mul(mkt, axis=0), fill_value=np.nan)
    return -resid.rolling(window).std(ddof=0) * np.sqrt(252)
