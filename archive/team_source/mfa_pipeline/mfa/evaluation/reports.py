"""End-to-end factor report builder."""
from __future__ import annotations

import pandas as pd

from .ic import information_coefficient, ic_summary, rolling_ic, ic_decay
from .quantiles import quantile_returns, quantile_summary, long_short_returns, factor_turnover
from .stats import perf_stats


def factor_report(
    name: str,
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    close: pd.DataFrame | None = None,
    n_quantiles: int = 5,
    horizons_for_decay: list[int] | None = None,
    fwd_horizon_days: int = 1,
) -> dict:
    """Compute a full set of diagnostics for a single factor.

    Returns a dict with keys:
        ic_series, ic_summary, ic_rolling, ic_decay (optional),
        q_returns, q_summary, ls_returns, ls_summary, turnover.
    """
    ic = information_coefficient(factor, fwd_ret)
    ics = ic_summary(ic)
    ic_roll = rolling_ic(ic, window=21)
    decay = ic_decay(factor, close, horizons_for_decay) if (close is not None and horizons_for_decay) else None

    qret = quantile_returns(factor, fwd_ret, n_quantiles=n_quantiles)
    qsum = quantile_summary(qret, fwd_horizon_days=fwd_horizon_days)
    ls = long_short_returns(qret)
    lss = perf_stats(ls, periods_per_year=252 / fwd_horizon_days)

    to = factor_turnover(factor, n_quantiles=n_quantiles)

    return {
        "name": name,
        "ic_series": ic,
        "ic_summary": ics,
        "ic_rolling": ic_roll,
        "ic_decay": decay,
        "q_returns": qret,
        "q_summary": qsum,
        "ls_returns": ls,
        "ls_summary": lss,
        "turnover": to,
    }
