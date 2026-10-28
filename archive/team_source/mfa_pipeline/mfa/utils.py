"""Small shared helpers used across the package."""
from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "mfa", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_LOG_FMT, datefmt="%H:%M:%S"))
        log.addHandler(h)
    log.setLevel(level)
    log.propagate = False
    return log


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------
def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Per-series winsorize. NaNs stay NaN."""
    if s.dropna().empty:
        return s
    lo, hi = s.quantile([lower, upper])
    return s.clip(lower=lo, upper=hi)


def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd


def neutralize(values: pd.Series, by: pd.DataFrame) -> pd.Series:
    """OLS-residualize `values` against the columns of `by` (per cross-section).

    Both `values` and `by` must share the same index.
    Returns the residuals with the same index.
    """
    df = pd.concat([values.rename("y"), by], axis=1).dropna()
    if df.empty:
        return values * np.nan
    y = df["y"].values
    X = df.drop(columns="y").values
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    res = pd.Series(y - yhat, index=df.index, name=values.name)
    return res.reindex(values.index)


def safe_div(a: pd.Series | float, b: pd.Series | float) -> pd.Series | float:
    return np.where(np.abs(b) < 1e-12, np.nan, a / b)


# ---------------------------------------------------------------------------
# Pandas helpers
# ---------------------------------------------------------------------------
def to_panel(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Convert a (date, symbol) long DF to a wide (date x symbol) panel."""
    return long_df.reset_index().pivot(index="date", columns="symbol", values=value_col)


def to_long(wide: pd.DataFrame, name: str) -> pd.DataFrame:
    """Inverse of `to_panel`."""
    return (
        wide.stack(dropna=False)
        .rename(name)
        .to_frame()
        .rename_axis(["date", "symbol"])
    )


def align_calendar(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Reindex a dict of wide panels to the union of their date indices."""
    if not panels:
        return panels
    cal = sorted(set().union(*[p.index for p in panels.values()]))
    cols = sorted(set().union(*[p.columns for p in panels.values()]))
    return {k: v.reindex(index=cal, columns=cols) for k, v in panels.items()}


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
@contextmanager
def chdir(path: str | os.PathLike[str]):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def chunked(seq: Iterable, size: int) -> Iterable[list]:
    buf: list = []
    for x in seq:
        buf.append(x)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


def ensure_dir(p: str | os.PathLike[str]) -> Path:
    pth = Path(p)
    pth.mkdir(parents=True, exist_ok=True)
    return pth
