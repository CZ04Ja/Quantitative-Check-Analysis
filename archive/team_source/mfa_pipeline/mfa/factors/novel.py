"""'Brain-storm' factors -- non-obvious, less-published ideas.

These are not magic. The point is to demonstrate a creative search process:
each factor encodes a *behavioral story* that *might* be priced. The
evaluation layer will tell us, factor by factor, which ones actually pay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import REGISTRY


# -----------------------------------------------------------------------
# 1) Pullback-from-high inside an uptrend
# -----------------------------------------------------------------------
@REGISTRY.register(
    name="pullback_in_uptrend",
    family="novel",
    description=(
        "Distance from 252d high * sign(252d trend). Story: small, healthy "
        "pullbacks inside a long-term uptrend are a buying opportunity."
    ),
    default_params={"long": 252, "pullback_window": 21},
)
def pullback_in_uptrend(panels, long: int = 252, pullback_window: int = 21) -> pd.DataFrame:
    c = panels["close"]
    long_trend = np.sign(c / c.shift(long) - 1.0)
    hi = c.rolling(pullback_window).max()
    pullback = (c / hi - 1.0)              # in [-1, 0], 0 = at the high
    # When trend is up, prefer modest pullbacks (slightly negative pullback).
    # We want the factor to be *higher* for slight pullbacks within an uptrend.
    # So multiply: long_trend * pullback gives positive values during uptrend
    # if the pullback is shallow (close to 0). To rank "shallowest pullback first"
    # we use: long_trend * (-pullback)  -> close to 0 when shallow, large when deep.
    # And then negate so shallow > deep.
    raw = -(long_trend * (-pullback))
    return raw


# -----------------------------------------------------------------------
# 2) Acceleration (2nd derivative of log price)
# -----------------------------------------------------------------------
@REGISTRY.register(
    name="acceleration_21_63",
    family="novel",
    description=(
        "Difference between 21d and 63d cumulative log-return. Positive value "
        "means the recent month is outpacing the trailing quarter."
    ),
    default_params={"short": 21, "long": 63},
)
def acceleration(panels, short: int = 21, long: int = 63) -> pd.DataFrame:
    c = panels["close"]
    r = np.log(c / c.shift(1))
    short_avg = r.rolling(short).mean()
    long_avg = r.rolling(long).mean()
    return short_avg - long_avg


# -----------------------------------------------------------------------
# 3) Volume-Price Confirmation
# -----------------------------------------------------------------------
@REGISTRY.register(
    name="vol_price_confirmation_21",
    family="novel",
    description=(
        "Rolling correlation between daily return and daily volume change "
        "over 21d. Positive when up days come with rising volume (genuine demand)."
    ),
    default_params={"window": 21},
)
def vol_price_confirmation(panels, window: int = 21) -> pd.DataFrame:
    c = panels["close"]
    v = panels["volume"]
    r = np.log(c / c.shift(1))
    dv = np.log((v + 1) / (v.shift(1) + 1))
    # rolling per-symbol correlation
    out = (
        r.rolling(window)
        .corr(dv)
    )
    return out


# -----------------------------------------------------------------------
# 4) Skewness anomaly (lottery effect)
# -----------------------------------------------------------------------
@REGISTRY.register(
    name="negative_skew_63",
    family="novel",
    description=(
        "Negative of trailing 63d daily-return skewness. Positively-skewed "
        "(lottery-like) names tend to under-perform (Bali-Cakici-Whitelaw 2011)."
    ),
    default_params={"window": 63},
)
def negative_skew(panels, window: int = 63) -> pd.DataFrame:
    c = panels["close"]
    r = np.log(c / c.shift(1))
    return -r.rolling(window).skew()


# -----------------------------------------------------------------------
# 5) "Mood" -- distribution-shape factor combining skew & kurt
# -----------------------------------------------------------------------
@REGISTRY.register(
    name="mood_63",
    family="novel",
    description=(
        "Mood = sign(skew) * log(1+kurt). Captures asymmetric tail behaviour. "
        "Empirical sign in this universe is data-driven."
    ),
    default_params={"window": 63},
)
def mood(panels, window: int = 63) -> pd.DataFrame:
    c = panels["close"]
    r = np.log(c / c.shift(1))
    sk = r.rolling(window).skew()
    kt = r.rolling(window).kurt()
    return -(np.sign(sk) * np.log1p(kt.abs()))


# -----------------------------------------------------------------------
# 6) Beta-adjusted maximum daily return ("MAX" factor of Bali et al.)
# -----------------------------------------------------------------------
@REGISTRY.register(
    name="max_return_inverse_21",
    family="novel",
    description=(
        "Negative of max daily return in trailing 21d. Stocks with extreme "
        "single-day spikes attract retail and subsequently underperform."
    ),
    default_params={"window": 21},
)
def max_return_inverse(panels, window: int = 21) -> pd.DataFrame:
    c = panels["close"]
    r = c / c.shift(1) - 1.0
    return -r.rolling(window).max()


# -----------------------------------------------------------------------
# 7) Inversion-of-extremes -- pick names that have NOT been at extremes
# -----------------------------------------------------------------------
@REGISTRY.register(
    name="distance_from_extremes_63",
    family="novel",
    description=(
        "1 - (close - 63d_min) / (63d_max - 63d_min). Higher = closer to the "
        "min of the channel = mean-reversion candidate inside a sideways tape."
    ),
    default_params={"window": 63},
)
def distance_from_extremes(panels, window: int = 63) -> pd.DataFrame:
    c = panels["close"]
    hi = c.rolling(window).max()
    lo = c.rolling(window).min()
    rng = (hi - lo).replace(0, np.nan)
    pos = (c - lo) / rng
    return 1.0 - pos


# -----------------------------------------------------------------------
# 8) Friday-effect (weekly seasonal flow story)
# -----------------------------------------------------------------------
@REGISTRY.register(
    name="weekly_seasonal_friday",
    family="novel",
    description=(
        "Mean Friday return over trailing 26 weeks. Some names exhibit a stable "
        "end-of-week premium driven by retail/passive flows."
    ),
    default_params={"weeks": 26},
)
def friday_seasonal(panels, weeks: int = 26) -> pd.DataFrame:
    c = panels["close"]
    r = c / c.shift(1) - 1.0
    is_friday = (r.index.dayofweek == 4)
    # mask non-Friday rows to NaN, then rolling-mean across the calendar
    fri_arr = np.where(is_friday[:, None], r.values, np.nan)
    fri = pd.DataFrame(fri_arr, index=r.index, columns=r.columns)
    return fri.rolling(window=weeks * 5, min_periods=max(5, weeks // 2)).mean()
