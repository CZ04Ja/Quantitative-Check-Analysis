"""Factor-level diagnostic plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_ic_series(ic: pd.Series, rolling_window: int = 21, ax=None, title: str | None = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.5))
    ic.plot(ax=ax, color="#bbbbbb", lw=0.8, label="daily IC")
    ic.rolling(rolling_window).mean().plot(ax=ax, color="#2E86AB", lw=1.8,
                                           label=f"{rolling_window}d rolling mean")
    ax.axhline(0, color="black", lw=0.7)
    ax.axhline(ic.mean(), color="#E63946", lw=1.0, ls="--",
               label=f"mean = {ic.mean():.3f}")
    ax.set_ylabel("IC")
    ax.set_title(title or "Information Coefficient")
    ax.legend(loc="upper left")
    return ax


def plot_ic_decay(decay: pd.Series, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    decay.plot(kind="bar", ax=ax, color="#2E86AB", edgecolor="black")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("forward horizon (days)")
    ax.set_ylabel("mean IC")
    ax.set_title("IC decay")
    return ax


def plot_quantile_returns(qret: pd.DataFrame, ax=None, title: str | None = None):
    """Cumulative-return plot per quantile bucket."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.5))
    eq = (1 + qret.fillna(0)).cumprod()
    palette = sns.color_palette("RdYlGn", eq.shape[1])
    for i, c in enumerate(eq.columns):
        eq[c].plot(ax=ax, label=c, color=palette[i], lw=1.8)
    ax.set_yscale("log")
    ax.set_ylabel("growth of $1 (log)")
    ax.set_title(title or "Quantile cumulative returns")
    ax.legend(ncol=eq.shape[1])
    return ax


def plot_factor_correlation(corr: pd.DataFrame, ax=None):
    if ax is None:
        size = max(6, 0.5 * len(corr))
        _, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        corr, annot=True, fmt=".2f", center=0.0, cmap="RdBu_r",
        vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.7}, ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Mean cross-sectional factor correlation")
    return ax
