"""Price momentum family."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import REGISTRY


@REGISTRY.register(
    name="mom_21",
    family="momentum",
    description="1-month price momentum (skip last day to avoid microstructure).",
    default_params={"lookback": 21, "skip": 1},
)
def mom_short(panels, lookback: int = 21, skip: int = 1) -> pd.DataFrame:
    c = panels["close"]
    return c.shift(skip) / c.shift(skip + lookback) - 1.0


@REGISTRY.register(
    name="mom_63",
    family="momentum",
    description="3-month price momentum, skip-1.",
    default_params={"lookback": 63, "skip": 1},
)
def mom_med(panels, lookback: int = 63, skip: int = 1) -> pd.DataFrame:
    c = panels["close"]
    return c.shift(skip) / c.shift(skip + lookback) - 1.0


@REGISTRY.register(
    name="mom_126",
    family="momentum",
    description="6-month price momentum, skip-1.",
    default_params={"lookback": 126, "skip": 1},
)
def mom_long(panels, lookback: int = 126, skip: int = 1) -> pd.DataFrame:
    c = panels["close"]
    return c.shift(skip) / c.shift(skip + lookback) - 1.0


@REGISTRY.register(
    name="mom_252_21",
    family="momentum",
    description="Classic 12-1 momentum (Jegadeesh & Titman 1993).",
    default_params={"long": 252, "skip": 21},
)
def mom_12_1(panels, long: int = 252, skip: int = 21) -> pd.DataFrame:
    c = panels["close"]
    return c.shift(skip) / c.shift(skip + long) - 1.0


@REGISTRY.register(
    name="mom_residual_63",
    family="momentum",
    description="Beta-neutral residual momentum vs. equal-weight market over 63d.",
    default_params={"lookback": 63, "skip": 1},
)
def mom_residual(panels, lookback: int = 63, skip: int = 1) -> pd.DataFrame:
    """Residualize 63-day cumulative return against an equal-weight market
    return computed from the same universe.
    """
    c = panels["close"]
    r = np.log(c / c.shift(1))
    mkt = r.mean(axis=1)
    # Per-symbol rolling regression r = a + b*mkt + e, return e summed over window
    # We'll approximate by: residual_i,t = r_i,t - beta_i,t * mkt_t
    # then sum residuals over the window.
    # Compute rolling beta_i = cov(r_i, mkt) / var(mkt) per symbol.
    cov = r.rolling(lookback).cov(mkt)
    var = mkt.rolling(lookback).var()
    beta = cov.div(var.replace(0, np.nan), axis=0)
    resid = r.sub(beta.mul(mkt, axis=0), fill_value=np.nan)
    return resid.shift(skip).rolling(lookback).sum()
