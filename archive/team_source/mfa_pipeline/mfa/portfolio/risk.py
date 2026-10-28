"""Covariance estimators for portfolio optimization."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Plain sample covariance (date x asset returns)."""
    R = returns.dropna(axis=1, how="all").dropna(how="all")
    return R.cov(ddof=0)


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage covariance."""
    R = returns.dropna(axis=1, how="all").dropna()
    if R.empty:
        return pd.DataFrame()
    lw = LedoitWolf().fit(R.values)
    return pd.DataFrame(lw.covariance_, index=R.columns, columns=R.columns)


def ewma_cov(returns: pd.DataFrame, halflife: int = 60) -> pd.DataFrame:
    """EWMA covariance (RiskMetrics style)."""
    R = returns.dropna(axis=1, how="all").dropna()
    if R.empty:
        return pd.DataFrame()
    weights = np.exp(np.log(0.5) / halflife * np.arange(len(R))[::-1])
    weights /= weights.sum()
    Rw = R.values - np.average(R.values, axis=0, weights=weights)
    cov = (Rw.T * weights) @ Rw
    return pd.DataFrame(cov, index=R.columns, columns=R.columns)
