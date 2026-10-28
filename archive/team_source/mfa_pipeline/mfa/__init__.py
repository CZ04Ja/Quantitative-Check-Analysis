"""Multi-Factor Alpha (MFA) Pipeline.

A self-contained, industry-style multi-factor research framework built on top
of Alpaca free historical data. The package is intentionally split into small
modules so any layer can be swapped or reused independently:

    mfa.data       -> universe definition, Alpaca fetcher, preprocessing
    mfa.features   -> reusable feature builders (price/volume/technical/x-sec)
    mfa.factors    -> alpha factors (classic + creative)
    mfa.evaluation -> IC, IR, quantile portfolios, Sharpe, attribution, ...
    mfa.portfolio  -> factor combination, optimization, risk model, timing
    mfa.backtest   -> daily-rebalanced backtest engine with cost model
    mfa.viz        -> all plotting / tearsheet helpers

The package has zero dependency on the parent project: copy/move the
`mfa_pipeline/` folder anywhere and `pip install -r requirements.txt`.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("mfa")
except PackageNotFoundError:  # not installed, running from source
    __version__ = "0.1.0"

__all__ = [
    "data",
    "features",
    "factors",
    "evaluation",
    "portfolio",
    "backtest",
    "viz",
]
