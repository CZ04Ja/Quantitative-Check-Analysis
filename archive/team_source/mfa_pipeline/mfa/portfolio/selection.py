"""Anti-overfitting helpers: factor selection on a sealed training fold.

Discipline
----------
1. The user picks a `train_end` date BEFORE looking at any test results.
2. `select_factors_train` looks at IC t-stats COMPUTED ONLY ON TRAIN data and
   returns the names that pass a multiple-testing-aware threshold.
3. Everything downstream uses ONLY those factors. Test fold is touched once,
   at the very end, to compute the OOS metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import forward_returns
from ..evaluation import information_coefficient


def split_dates(
    dates: pd.DatetimeIndex,
    train_end: str,
    val_end: str | None = None,
) -> dict[str, pd.DatetimeIndex]:
    """Split a DatetimeIndex into train / (val) / test."""
    te = pd.Timestamp(train_end, tz=dates.tz)
    out = {"train": dates[dates <= te]}
    if val_end is not None:
        ve = pd.Timestamp(val_end, tz=dates.tz)
        out["val"] = dates[(dates > te) & (dates <= ve)]
        out["test"] = dates[dates > ve]
    else:
        out["test"] = dates[dates > te]
    return out


def factor_train_summary(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    train_end: str,
    horizon: int = 5,
) -> pd.DataFrame:
    """Per-factor IC summary computed on TRAIN ONLY.

    The result columns include `ic_t_stat`, `ic_mean`, `ic_ir`, `n_periods`
    and a `bonferroni_pass` flag for |t| > t_alpha with alpha=0.10/N (so we
    insist on factors that survive a multiple-testing correction).
    """
    fwd_panel = forward_returns(close, [horizon])[horizon]
    train_idx = fwd_panel.index <= pd.Timestamp(train_end, tz=fwd_panel.index.tz)

    n_factors = len(factors)
    # Two-sided alpha=0.10 with Bonferroni: alpha_corr = 0.10 / N
    # threshold t = z(1 - alpha/2) ~ z(1 - 0.05/N). For N=29, ~3.04
    z_thresh = float(np.abs(np.quantile(np.random.RandomState(0).standard_normal(2_000_000),
                                        1.0 - 0.10 / (2 * n_factors))))
    rows = []
    for name, f in factors.items():
        ic = information_coefficient(
            f.loc[train_idx], fwd_panel.loc[train_idx]
        )
        ic = ic.dropna()
        if ic.empty:
            rows.append({"factor": name, "ic_mean": np.nan, "ic_t_stat": np.nan,
                         "n_periods": 0, "bonferroni_pass": False})
            continue
        mu, sd = ic.mean(), ic.std(ddof=0)
        ir = mu / sd if sd > 0 else np.nan
        t = (mu / sd) * np.sqrt(len(ic)) if sd > 0 else np.nan
        rows.append({
            "factor": name,
            "ic_mean": float(mu), "ic_ir": float(ir),
            "ic_t_stat": float(t),
            "n_periods": int(len(ic)),
            "bonferroni_pass": bool(np.abs(t) > z_thresh),
        })
    df = pd.DataFrame(rows).set_index("factor")
    df.attrs["bonferroni_t_threshold"] = z_thresh
    df.attrs["train_end"] = str(train_end)
    return df.sort_values("ic_t_stat", ascending=False, key=lambda s: s.abs())


def select_factors_train(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    train_end: str,
    horizon: int = 5,
    t_threshold: float | str = "bonferroni",
    keep_top_k: int | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Return the names of factors that survive the |t|-stat threshold on TRAIN.

    `t_threshold`:
        * "bonferroni"  -- 0.10 family-wise α / N two-sided threshold (~3 for N≈30)
        * float         -- raw |t| cut-off (e.g. 2.0)
    `keep_top_k`:
        if not None, instead of thresholding take the top-K by |t|.
    """
    summary = factor_train_summary(factors, close, train_end, horizon=horizon)
    if keep_top_k is not None:
        keep = summary.iloc[:keep_top_k].index.tolist()
    elif t_threshold == "bonferroni":
        keep = summary[summary["bonferroni_pass"]].index.tolist()
    else:
        keep = summary[summary["ic_t_stat"].abs() > float(t_threshold)].index.tolist()
    return keep, summary
