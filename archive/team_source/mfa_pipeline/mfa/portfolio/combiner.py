"""Multi-factor combination."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.crosssectional import xs_rank, xs_winsorize, xs_zscore


def standardize_factor(f: pd.DataFrame, winsor: tuple[float, float] = (0.01, 0.99)
                       ) -> pd.DataFrame:
    """Cross-sectional winsorize + zscore. Fully out-of-sample friendly because
    every transform uses only same-day data."""
    return xs_zscore(xs_winsorize(f, *winsor))


def rank_score(f: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank scaled to [-0.5, +0.5]."""
    return xs_rank(f, pct=True) - 0.5


def combine_factors(
    factors: dict[str, pd.DataFrame],
    weights: dict[str, float] | pd.DataFrame | None = None,
    standardize: bool = True,
) -> pd.DataFrame:
    """Combine many factors into one composite signal.

    Parameters
    ----------
    factors    : dict {name: panel}
    weights    : either
                   * None       -> equal-weight (1/N)
                   * dict       -> static weights {name: w}
                   * DataFrame  -> dynamic weights, index=date, cols=factor names
    standardize: True -> per-row winsorize+zscore each factor first.
    """
    names = list(factors.keys())
    if standardize:
        fs = {n: standardize_factor(factors[n]) for n in names}
    else:
        fs = factors

    # Align all factor panels to the same index/cols.
    idx = sorted(set().union(*[v.index for v in fs.values()]))
    cols = sorted(set().union(*[v.columns for v in fs.values()]))
    fs = {n: v.reindex(index=idx, columns=cols) for n, v in fs.items()}

    # NaN contributions from a single factor must NOT poison the composite
    # (e.g. an all-zero raw factor gets standardized to all NaN). We use
    # `DataFrame.add(..., fill_value=0)` so missing values are treated as
    # zero contribution when at least one side has a value.
    composite = pd.DataFrame(0.0, index=idx, columns=cols)

    if weights is None:
        for n in names:
            contrib = fs[n] * (1.0 / len(names))
            composite = composite.add(contrib, fill_value=0.0)
        return composite

    if isinstance(weights, dict):
        for n in names:
            contrib = fs[n] * float(weights.get(n, 0.0))
            composite = composite.add(contrib, fill_value=0.0)
        return composite

    if isinstance(weights, pd.DataFrame):
        w = weights.reindex(idx).reindex(columns=names).ffill()
        for n in names:
            contrib = fs[n].mul(w[n], axis=0)
            composite = composite.add(contrib, fill_value=0.0)
        return composite

    raise TypeError(f"unsupported weights type: {type(weights)}")
