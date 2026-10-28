"""Composite multi-panel tearsheets."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from .factor_viz import (
    plot_ic_series, plot_ic_decay, plot_quantile_returns,
)
from .perf_viz import (
    plot_equity, plot_drawdown, plot_rolling_sharpe,
    plot_monthly_heatmap, plot_annual_returns, plot_turnover,
)


def factor_tearsheet(report: dict, savepath: str | None = None):
    """One-page summary for a single factor (output of evaluation.factor_report)."""
    fig = plt.figure(figsize=(13, 11))
    gs = gridspec.GridSpec(4, 3, hspace=0.55, wspace=0.30)

    fig.suptitle(f"Factor Tearsheet -- {report['name']}",
                 fontsize=14, fontweight="bold", y=0.995)

    # Row 1: IC time series spanning full width
    ax_ic = fig.add_subplot(gs[0, :])
    plot_ic_series(report["ic_series"], ax=ax_ic, title="Daily IC + 21d rolling mean")

    # Row 2 left: IC decay
    if report.get("ic_decay") is not None:
        ax_dec = fig.add_subplot(gs[1, 0])
        plot_ic_decay(report["ic_decay"], ax=ax_dec)
    # Row 2 mid+right: quantile cumulative
    ax_q = fig.add_subplot(gs[1, 1:])
    plot_quantile_returns(report["q_returns"], ax=ax_q,
                          title="Quantile cumulative returns")

    # Row 3: long-short equity + drawdown
    ax_ls = fig.add_subplot(gs[2, :2])
    plot_equity(report["ls_returns"], ax=ax_ls, log_y=False,
                title="Long-Short cumulative return")
    ax_dd = fig.add_subplot(gs[2, 2])
    plot_drawdown(report["ls_returns"], ax=ax_dd)

    # Row 4: stats text
    ax_txt = fig.add_subplot(gs[3, :])
    ax_txt.axis("off")
    ic = report["ic_summary"]
    ls = report["ls_summary"]
    text = (
        f"IC mean = {ic['ic_mean']:+.4f}    IR = {ic['ic_ir']:+.3f}"
        f"    t-stat = {ic['ic_t_stat']:+.2f}    hit-rate = {ic['ic_hit_rate']:.2%}\n"
        f"L-S Sharpe = {ls['sharpe']:+.2f}    ann.return = {ls['ann_return']:+.2%}"
        f"    max DD = {ls['max_drawdown']:.2%}    Calmar = {ls['calmar']:+.2f}"
        f"    avg turnover = {report['turnover'].mean():.2%}"
    )
    ax_txt.text(0.0, 0.5, text, fontsize=11, family="monospace", va="center")

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


def strategy_tearsheet(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    turnover: pd.Series | None = None,
    perf: dict | None = None,
    title: str = "Multi-Factor Strategy Tearsheet",
    savepath: str | None = None,
):
    fig = plt.figure(figsize=(13, 14))
    gs = gridspec.GridSpec(6, 3, hspace=0.55, wspace=0.30)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    ax1 = fig.add_subplot(gs[0:2, :])
    plot_equity(returns, benchmark=benchmark, ax=ax1, log_y=True,
                title="Equity curve (log scale)")

    ax2 = fig.add_subplot(gs[2, :])
    plot_drawdown(returns, ax=ax2)

    ax3 = fig.add_subplot(gs[3, :])
    plot_rolling_sharpe(returns, window=63, ax=ax3)

    ax4 = fig.add_subplot(gs[4, :2])
    plot_monthly_heatmap(returns, ax=ax4)

    ax5 = fig.add_subplot(gs[4, 2])
    plot_annual_returns(returns, benchmark=benchmark, ax=ax5)

    if turnover is not None:
        ax6 = fig.add_subplot(gs[5, :2])
        plot_turnover(turnover, ax=ax6)

    ax7 = fig.add_subplot(gs[5, 2])
    ax7.axis("off")
    if perf:
        text = "\n".join([
            f"ann. return     {perf['ann_return']:+.2%}",
            f"ann. vol        {perf['ann_vol']:+.2%}",
            f"Sharpe          {perf['sharpe']:+.2f}",
            f"Sortino         {perf['sortino']:+.2f}",
            f"Calmar          {perf['calmar']:+.2f}",
            f"max drawdown    {perf['max_drawdown']:.2%}",
            f"max DD duration {perf['max_dd_duration']} bars",
            f"hit rate        {perf['hit_rate']:.2%}",
            f"VaR 95%         {perf['var_95']:.2%}",
            f"CVaR 95%        {perf['cvar_95']:.2%}",
        ])
        ax7.text(0.0, 1.0, text, fontsize=10, family="monospace", va="top")

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig
