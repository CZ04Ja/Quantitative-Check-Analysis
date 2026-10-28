"""Factor zoo.

A factor is anything that maps a dict of OHLCV panels to a single wide
DataFrame (date x symbol) of *raw* values. Cross-sectional standardization
(rank/zscore/winsorize/neutralize) is applied later in the pipeline.

Convention
----------
Higher factor value -> *expected* higher next-period return (long the top,
short the bottom). When a raw signal is naturally inverted (e.g. realized
volatility, where low-vol outperforms), the factor function returns
`-raw_value` so the sign convention is consistent.
"""
from .base import Factor, FactorRegistry, REGISTRY
from . import momentum  # noqa: F401  (registers factors as a side effect)
from . import reversal  # noqa: F401
from . import volatility  # noqa: F401
from . import quality  # noqa: F401
from . import liquidity  # noqa: F401
from . import microstructure  # noqa: F401
from . import novel  # noqa: F401

from .compute import compute_all, compute_subset

__all__ = [
    "Factor", "FactorRegistry", "REGISTRY",
    "compute_all", "compute_subset",
]
