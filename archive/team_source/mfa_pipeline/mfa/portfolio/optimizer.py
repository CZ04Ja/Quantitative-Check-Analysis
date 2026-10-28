"""Portfolio construction methods. All return *target weights* (date x symbol)
that sum to zero (long-short, dollar-neutral) or one (long-only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import risk as _risk


# ---------------------------------------------------------------------------
# 1) Quantile long-short (the default cross-sectional book)
# ---------------------------------------------------------------------------
def long_short_quantile_weights(
    signal: pd.DataFrame,
    top_q: float = 0.2,
    bot_q: float = 0.2,
    gross_leverage: float = 2.0,
    max_weight: float | None = None,
) -> pd.DataFrame:
    """Equal-weight long top quintile, short bottom quintile, dollar-neutral.

    gross_leverage = 2.0 means $1 long + $1 short per $1 capital.
    """
    out = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for date, row in signal.iterrows():
        s = row.dropna()
        if s.empty:
            continue
        n = len(s)
        n_top = max(1, int(np.ceil(n * top_q)))
        n_bot = max(1, int(np.ceil(n * bot_q)))
        ranks = s.rank(method="first", ascending=False)
        long_names = ranks[ranks <= n_top].index
        short_names = ranks[ranks > n - n_bot].index
        long_w = (gross_leverage / 2) / len(long_names)
        short_w = -(gross_leverage / 2) / len(short_names)
        for nm in long_names:
            out.loc[date, nm] = long_w
        for nm in short_names:
            out.loc[date, nm] = short_w
        if max_weight is not None:
            out.loc[date] = out.loc[date].clip(lower=-max_weight, upper=max_weight)
    return out


# ---------------------------------------------------------------------------
# 2) Mean-variance long-short (signal as expected return, cov as risk)
# ---------------------------------------------------------------------------
def mean_variance_weights(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    cov_method: str = "ledoit_wolf",
    cov_lookback: int = 252,
    risk_aversion: float = 5.0,
    gross_leverage: float = 2.0,
    max_weight: float = 0.10,
) -> pd.DataFrame:
    """Solve max_w  alpha'w - 0.5 * lambda * w' S w
       s.t.  sum(w) = 0,  sum(|w|) <= L,  |w_i| <= max_weight.
    Done numerically via SLSQP at each rebalance date.
    """
    out = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    last_w: pd.Series | None = None
    for date in signal.index:
        if date in rebalance_dates:
            alpha = signal.loc[date].dropna()
            cols = alpha.index
            if len(cols) < 5:
                continue
            window = returns.loc[:date].iloc[-cov_lookback:][cols].dropna(how="all")
            if window.shape[0] < 30 or window.shape[1] < 5:
                continue
            if cov_method == "sample":
                S = _risk.sample_cov(window)
            elif cov_method == "ewma":
                S = _risk.ewma_cov(window, halflife=cov_lookback // 4)
            else:
                S = _risk.ledoit_wolf_cov(window)
            S = S.reindex(index=cols, columns=cols).fillna(0.0).values
            a = alpha.values

            def obj(w):
                return -(a @ w) + 0.5 * risk_aversion * w @ S @ w

            cons = [
                {"type": "eq", "fun": lambda w: np.sum(w)},
                {"type": "ineq", "fun": lambda w: gross_leverage - np.sum(np.abs(w))},
            ]
            bounds = [(-max_weight, max_weight)] * len(cols)
            x0 = np.zeros(len(cols))
            res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"maxiter": 200, "ftol": 1e-8})
            w = pd.Series(res.x, index=cols)
            last_w = w.reindex(signal.columns).fillna(0.0)
        if last_w is not None:
            out.loc[date] = last_w
    return out


# ---------------------------------------------------------------------------
# 3) Risk parity (long-only inverse-vol)
# ---------------------------------------------------------------------------
def risk_parity_weights(
    returns: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    lookback: int = 63,
) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    last_w: pd.Series | None = None
    for date in returns.index:
        if date in rebalance_dates:
            window = returns.loc[:date].iloc[-lookback:]
            vol = window.std(ddof=0)
            inv = 1.0 / vol.replace(0, np.nan)
            inv = inv.dropna()
            if inv.empty:
                continue
            w = inv / inv.sum()
            last_w = w.reindex(returns.columns).fillna(0.0)
        if last_w is not None:
            out.loc[date] = last_w
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def target_weights(
    method: str,
    *,
    signal: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    rebalance_dates: list[pd.Timestamp] | None = None,
    **kwargs,
) -> pd.DataFrame:
    if method == "long_short_quantile":
        assert signal is not None
        return long_short_quantile_weights(signal, **kwargs)
    if method == "mean_variance":
        assert signal is not None and returns is not None and rebalance_dates is not None
        return mean_variance_weights(signal, returns, rebalance_dates, **kwargs)
    if method == "risk_parity":
        assert returns is not None and rebalance_dates is not None
        return risk_parity_weights(returns, rebalance_dates, **kwargs)
    raise ValueError(method)
