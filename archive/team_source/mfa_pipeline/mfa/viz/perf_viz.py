"""Strategy / portfolio performance plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..evaluation.stats import drawdown_curve, monthly_returns, annual_returns, rolling_sharpe


def plot_equity(returns: pd.Series, benchmark: pd.Series | None = None,
                ax=None, log_y: bool = True, title: str | None = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.5))
    eq = (1 + returns.fillna(0)).cumprod()
    eq.plot(ax=ax, label="strategy", color="#2E86AB", lw=2.0)
    if benchmark is not None:
        bm = benchmark.reindex(returns.index).fillna(0)
        eq_b = (1 + bm).cumprod()
        eq_b.plot(ax=ax, label="benchmark", color="#E63946", lw=1.5, ls="--")
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel("growth of $1")
    ax.set_title(title or "Equity curve")
    ax.legend()
    return ax


def plot_drawdown(returns: pd.Series, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 2.8))
    dd = drawdown_curve(returns)
    ax.fill_between(dd.index, dd.values, 0, color="#E63946", alpha=0.6, lw=0)
    ax.set_ylabel("drawdown")
    ax.set_ylim(min(-0.01, dd.min() * 1.1), 0.01)
    ax.set_title("Drawdown")
    return ax


def plot_rolling_sharpe(returns: pd.Series, window: int = 63, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 2.8))
    rs = rolling_sharpe(returns, window=window)
    rs.plot(ax=ax, color="#2E86AB", lw=1.6)
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(1.0, color="#06A77D", lw=0.8, ls="--", label="Sharpe = 1")
    ax.set_ylabel("rolling Sharpe")
    ax.set_title(f"Rolling Sharpe ({window}d)")
    ax.legend()
    return ax


def plot_monthly_heatmap(returns: pd.Series, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))
    grid = monthly_returns(returns)
    if grid.empty:
        ax.set_title("Monthly returns (no data)")
        return ax
    sns.heatmap(
        grid * 100, annot=True, fmt=".1f", cmap="RdYlGn", center=0.0,
        cbar_kws={"label": "%"}, ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Monthly returns (%)")
    ax.set_xlabel("month")
    ax.set_ylabel("year")
    return ax


def plot_annual_returns(returns: pd.Series, benchmark: pd.Series | None = None, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))
    a = annual_returns(returns).rename("strategy")
    df = a.to_frame()
    if benchmark is not None:
        b = annual_returns(benchmark).rename("benchmark")
        df = df.join(b, how="outer")
    df.index = df.index.year
    df.plot(kind="bar", ax=ax, color=["#2E86AB", "#E63946"][: df.shape[1]])
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title("Annual returns")
    ax.set_ylabel("return")
    ax.set_xlabel("")
    return ax


def plot_turnover(turnover: pd.Series, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 2.5))
    turnover.plot(ax=ax, color="#8E44AD", lw=1.0)
    ax.axhline(turnover.mean(), color="black", ls="--", lw=0.8,
               label=f"mean = {turnover.mean():.2f}")
    ax.set_ylabel("turnover")
    ax.set_title("Per-rebalance turnover")
    ax.legend()
    return ax
