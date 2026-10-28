"""Performance statistics. Pure pandas/numpy (no quantstats dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _to_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("expected single column")
        return x.iloc[:, 0]
    return x


def drawdown_curve(returns: pd.Series) -> pd.Series:
    """Drawdown series in [-1, 0]."""
    r = _to_series(returns).dropna()
    eq = (1.0 + r).cumprod()
    cm = eq.cummax()
    return (eq / cm - 1.0).rename("drawdown")


def perf_stats(returns: pd.Series, periods_per_year: float = 252) -> dict[str, float]:
    """Comprehensive performance summary."""
    r = _to_series(returns).dropna()
    if r.empty:
        nan = float("nan")
        return {
            "ann_return": nan, "ann_vol": nan, "sharpe": nan,
            "sortino": nan, "calmar": nan, "max_drawdown": nan,
            "max_dd_duration": nan, "hit_rate": nan, "skew": nan,
            "kurt": nan, "var_95": nan, "cvar_95": nan,
            "best_period": nan, "worst_period": nan,
            "n_periods": 0, "total_return": nan,
        }

    eq = (1.0 + r).cumprod()
    total = float(eq.iloc[-1] - 1.0)
    yrs = len(r) / periods_per_year
    ann_ret = (eq.iloc[-1]) ** (1.0 / yrs) - 1.0 if yrs > 0 else np.nan
    ann_vol = r.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside = r[r < 0]
    dd_vol = downside.std(ddof=0) * np.sqrt(periods_per_year) if not downside.empty else np.nan
    sortino = ann_ret / dd_vol if dd_vol and dd_vol > 0 else np.nan

    dd = drawdown_curve(r)
    mdd = float(dd.min()) if not dd.empty else np.nan
    calmar = ann_ret / abs(mdd) if mdd and mdd < 0 else np.nan

    # Drawdown duration in periods
    if not dd.empty:
        in_dd = dd < 0
        groups = (in_dd != in_dd.shift()).cumsum()
        durations = in_dd.groupby(groups).sum()
        max_dd_dur = int(durations.max()) if not durations.empty else 0
    else:
        max_dd_dur = 0

    var95 = float(np.percentile(r, 5))
    cvar95 = float(r[r <= var95].mean()) if (r <= var95).any() else np.nan

    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino) if sortino is not None else np.nan,
        "calmar": float(calmar) if calmar is not None else np.nan,
        "max_drawdown": float(mdd),
        "max_dd_duration": int(max_dd_dur),
        "hit_rate": float((r > 0).mean()),
        "skew": float(r.skew()),
        "kurt": float(r.kurt()),
        "var_95": var95,
        "cvar_95": cvar95,
        "best_period": float(r.max()),
        "worst_period": float(r.min()),
        "n_periods": int(len(r)),
        "total_return": total,
    }


def rolling_sharpe(returns: pd.Series, window: int = 63, periods_per_year: float = 252
                   ) -> pd.Series:
    r = _to_series(returns)
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std(ddof=0).replace(0, np.nan)
    return (mu / sd) * np.sqrt(periods_per_year)


def annual_returns(returns: pd.Series) -> pd.Series:
    r = _to_series(returns).dropna()
    if r.empty:
        return pd.Series(dtype="float64", name="annual_return")
    yearly = r.resample("YE").apply(lambda s: (1.0 + s).prod() - 1.0)
    return yearly.rename("annual_return")


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Year x Month grid of monthly returns (as DataFrame)."""
    r = _to_series(returns).dropna()
    if r.empty:
        return pd.DataFrame()
    m = (1 + r).resample("ME").prod() - 1
    df = m.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    return df.pivot(index="year", columns="month", values="ret")
