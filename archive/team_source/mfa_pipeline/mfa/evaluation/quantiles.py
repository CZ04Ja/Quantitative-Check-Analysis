"""Quantile-portfolio analytics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import perf_stats


def _quantile_assignments(factor: pd.DataFrame, n_quantiles: int) -> pd.DataFrame:
    """Assign 1..n quantile per row. NaN where factor missing.

    Vectorized using row-wise rank percentile -> ceil to bucket label.
    """
    pct = factor.rank(axis=1, pct=True, method="average")
    # Map rank in (0, 1] to integers 1..N.
    labels = np.ceil(pct.values * n_quantiles)
    labels = np.where(np.isnan(pct.values), np.nan, labels)
    labels = np.clip(labels, 1, n_quantiles)
    return pd.DataFrame(labels, index=factor.index, columns=factor.columns)


def quantile_returns(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_quantiles: int = 5,
    weighting: str = "equal",
) -> pd.DataFrame:
    """Average forward return of each quantile bucket per period.

    weighting: 'equal' (default) or 'rank' (linear within-bucket weights).
    """
    assignments = _quantile_assignments(factor, n_quantiles)
    common_idx = assignments.index.intersection(fwd_ret.index)
    a = assignments.loc[common_idx]
    r = fwd_ret.loc[common_idx]

    out = pd.DataFrame(index=common_idx, columns=[f"Q{i}" for i in range(1, n_quantiles + 1)],
                       dtype="float64")
    for q in range(1, n_quantiles + 1):
        mask = (a == q)
        bucket = r.where(mask)
        if weighting == "equal":
            out[f"Q{q}"] = bucket.mean(axis=1)
        elif weighting == "rank":
            # rank within bucket per row -> linear weights
            ranks = bucket.rank(axis=1)
            weights = ranks.div(ranks.sum(axis=1), axis=0)
            out[f"Q{q}"] = (bucket * weights).sum(axis=1)
        else:
            raise ValueError(weighting)
    return out


def long_short_returns(qret: pd.DataFrame) -> pd.Series:
    """Top quantile minus bottom quantile."""
    cols = [c for c in qret.columns if c.startswith("Q")]
    cols.sort(key=lambda c: int(c[1:]))
    if len(cols) < 2:
        raise ValueError("Need at least 2 quantiles for long-short")
    top, bot = cols[-1], cols[0]
    return (qret[top] - qret[bot]).rename("long_short")


def quantile_summary(qret: pd.DataFrame, fwd_horizon_days: int = 1) -> pd.DataFrame:
    """Summary stats per quantile + long-short. Assumes non-overlapping returns
    aligned to periods of length `fwd_horizon_days`."""
    rows = []
    for c in qret.columns:
        rows.append({"name": c, **perf_stats(qret[c], periods_per_year=252 / fwd_horizon_days)})
    ls = long_short_returns(qret)
    rows.append({"name": "Q_top - Q_bot", **perf_stats(ls, periods_per_year=252 / fwd_horizon_days)})
    return pd.DataFrame(rows).set_index("name")


def factor_turnover(factor: pd.DataFrame, n_quantiles: int = 5) -> pd.Series:
    """Top-quantile membership turnover, day over day. 0 = no churn, 1 = full churn.

    Defined as 1 - |intersection| / |union| of consecutive top-quantile sets.
    """
    a = _quantile_assignments(factor, n_quantiles)
    top = (a == n_quantiles)
    out: dict[pd.Timestamp, float] = {}
    prev = None
    for date, row in top.iterrows():
        cur = set(row[row].index)
        if prev is None:
            out[date] = np.nan
        else:
            union = prev | cur
            inter = prev & cur
            out[date] = 1.0 - (len(inter) / len(union)) if union else np.nan
        prev = cur
    return pd.Series(out, name="top_q_turnover")
