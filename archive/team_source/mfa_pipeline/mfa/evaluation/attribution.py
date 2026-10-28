"""Inter-factor diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def correlation_matrix(factors: dict[str, pd.DataFrame], method: str = "spearman"
                       ) -> pd.DataFrame:
    """Average per-day cross-sectional correlation across factor pairs."""
    names = list(factors.keys())
    n = len(names)
    out = pd.DataFrame(np.eye(n), index=names, columns=names, dtype="float64")
    common_idx = None
    for f in factors.values():
        common_idx = f.index if common_idx is None else common_idx.intersection(f.index)

    for i, a in enumerate(names):
        fa = factors[a].loc[common_idx]
        for j, b in enumerate(names):
            if j <= i:
                continue
            fb = factors[b].loc[common_idx]
            corrs = []
            for date in common_idx:
                ra = fa.loc[date]
                rb = fb.loc[date]
                df = pd.concat([ra, rb], axis=1).dropna()
                if df.shape[0] < 5:
                    continue
                corrs.append(df.iloc[:, 0].corr(df.iloc[:, 1], method=method))
            v = float(np.nanmean(corrs)) if corrs else np.nan
            out.loc[a, b] = v
            out.loc[b, a] = v
    return out


def factor_attribution(strategy_ret: pd.Series, factor_rets: pd.DataFrame
                       ) -> pd.DataFrame:
    """Regress strategy returns on factor returns. Returns betas + t-stats + R^2."""
    df = pd.concat([strategy_ret.rename("y"), factor_rets], axis=1).dropna()
    if df.empty or df.shape[1] < 2:
        return pd.DataFrame()
    y = df["y"]
    X = sm.add_constant(df.drop(columns="y"))
    model = sm.OLS(y, X).fit()
    out = pd.DataFrame({
        "beta": model.params,
        "t_stat": model.tvalues,
        "p_value": model.pvalues,
    })
    out.attrs["r_squared"] = float(model.rsquared)
    out.attrs["adj_r_squared"] = float(model.rsquared_adj)
    return out
