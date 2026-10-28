"""Cross-sectional transforms applied per date (per row of a wide panel)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def xs_rank(panel: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """Rank within each row. `pct=True` gives ranks scaled into (0, 1)."""
    return panel.rank(axis=1, pct=pct, method="average")


def xs_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    mu = panel.mean(axis=1)
    sd = panel.std(axis=1, ddof=0).replace(0, np.nan)
    return panel.sub(mu, axis=0).div(sd, axis=0)


def xs_winsorize(panel: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    lo = panel.quantile(lower, axis=1)
    hi = panel.quantile(upper, axis=1)
    return panel.clip(lower=lo, upper=hi, axis=0)


def xs_demean(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sub(panel.mean(axis=1), axis=0)


def xs_neutralize(panel: pd.DataFrame, exposures: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """OLS-residualize each row of `panel` against the same-day rows of every
    exposure panel in `exposures`.

    All panels must share the same shape and index.
    """
    if not exposures:
        return panel
    out = panel.copy() * np.nan
    keys = list(exposures.keys())
    # Vectorize per-date with numpy.lstsq.
    for date, row in panel.iterrows():
        y = row.values
        Xcols = []
        for k in keys:
            try:
                Xcols.append(exposures[k].loc[date].values)
            except KeyError:
                Xcols = None
                break
        if Xcols is None:
            continue
        X = np.column_stack([np.ones_like(y)] + Xcols).astype(float)
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if mask.sum() < X.shape[1] + 1:
            continue
        beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        resid = np.full_like(y, np.nan, dtype=float)
        resid[mask] = y[mask] - X[mask] @ beta
        out.loc[date, :] = resid
    return out
