"""Re-usable feature builders.

Two flavors:
    * Time-series features (rolling stats per symbol).
    * Cross-sectional transforms (rank / zscore / neutralize per date).

All feature functions take wide panels and return wide panels of the same
shape, so they can be composed freely.
"""
from .price import (
    log_return,
    cumulative_return,
    rolling_volatility,
    rolling_skew,
    rolling_kurt,
    moving_average,
    distance_from_high,
    distance_from_low,
    rolling_max_drawdown,
)
from .volume import (
    dollar_volume,
    volume_zscore,
    amihud_illiquidity,
    turnover_proxy,
)
from .technical import (
    rsi,
    macd,
    bollinger_bands,
    atr,
)
from .crosssectional import (
    xs_rank,
    xs_zscore,
    xs_winsorize,
    xs_demean,
    xs_neutralize,
)

__all__ = [
    "log_return", "cumulative_return", "rolling_volatility",
    "rolling_skew", "rolling_kurt", "moving_average",
    "distance_from_high", "distance_from_low", "rolling_max_drawdown",
    "dollar_volume", "volume_zscore", "amihud_illiquidity", "turnover_proxy",
    "rsi", "macd", "bollinger_bands", "atr",
    "xs_rank", "xs_zscore", "xs_winsorize", "xs_demean", "xs_neutralize",
]
