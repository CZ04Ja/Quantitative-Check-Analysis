"""Information Coefficient toolkit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def _cs_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    df = pd.concat([a, b], axis=1).dropna()
    if df.shape[0] < 5:
        return np.nan
    if method == "pearson":
        return float(pearsonr(df.iloc[:, 0], df.iloc[:, 1])[0])
    return float(spearmanr(df.iloc[:, 0], df.iloc[:, 1])[0])


def information_coefficient(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    method: str = "spearman",
    min_xsec_obs: int = 10,
) -> pd.Series:
    """Per-day cross-sectional rank correlation between factor and fwd return.

    Vectorized: Spearman = Pearson on ranks, and per-row Pearson is computed
    with broadcasted matrix ops. ~100x faster than the per-date python loop.
    """
    common_idx = factor.index.intersection(fwd_ret.index)
    common_cols = factor.columns.intersection(fwd_ret.columns)
    f = factor.loc[common_idx, common_cols]
    r = fwd_ret.loc[common_idx, common_cols]

    if method == "spearman":
        f_use = f.rank(axis=1)
        r_use = r.rank(axis=1)
    elif method == "pearson":
        f_use = f
        r_use = r
    else:
        raise ValueError(method)

    # Per-row Pearson correlation, ignoring NaNs.
    a = f_use.values.astype(float)
    b = r_use.values.astype(float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = np.where(mask, a, np.nan)
    b = np.where(mask, b, np.nan)
    n = mask.sum(axis=1)
    sum_a = np.nansum(a, axis=1)
    sum_b = np.nansum(b, axis=1)
    sum_ab = np.nansum(a * b, axis=1)
    sum_aa = np.nansum(a * a, axis=1)
    sum_bb = np.nansum(b * b, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        num = sum_ab - sum_a * sum_b / n
        den = np.sqrt((sum_aa - sum_a ** 2 / n) * (sum_bb - sum_b ** 2 / n))
        ic = np.where(den > 0, num / den, np.nan)
    ic = np.where(n >= min_xsec_obs, ic, np.nan)
    return pd.Series(ic, index=f.index, name="ic")


def rolling_ic(ic: pd.Series, window: int = 21) -> pd.DataFrame:
    return pd.DataFrame({
        "ic_rolling_mean": ic.rolling(window).mean(),
        "ic_rolling_std": ic.rolling(window).std(ddof=0),
        "ic_rolling_ir": ic.rolling(window).mean() / ic.rolling(window).std(ddof=0).replace(0, np.nan),
    })


def ic_decay(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizons: list[int],
    method: str = "spearman",
) -> pd.Series:
    """IC at multiple forward horizons (decay curve)."""
    out = {}
    for h in horizons:
        fwd = close.shift(-h) / close - 1.0
        ic = information_coefficient(factor, fwd, method=method)
        out[h] = ic.mean()
    return pd.Series(out).rename("ic_mean").rename_axis("horizon_days")


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """Summary statistics for an IC series."""
    s = ic.dropna()
    if s.empty:
        return {k: np.nan for k in [
            "ic_mean", "ic_std", "ic_ir", "ic_t_stat",
            "ic_hit_rate", "ic_skew", "ic_kurt", "n_periods",
        ]}
    mu = s.mean()
    sd = s.std(ddof=0)
    n = len(s)
    return {
        "ic_mean": float(mu),
        "ic_std": float(sd),
        "ic_ir": float(mu / sd) if sd > 0 else np.nan,
        # t-stat for "is mean IC zero?": IR * sqrt(n)
        "ic_t_stat": float((mu / sd) * np.sqrt(n)) if sd > 0 else np.nan,
        "ic_hit_rate": float((s > 0).mean()),
        "ic_skew": float(s.skew()),
        "ic_kurt": float(s.kurt()),
        "n_periods": int(n),
    }
