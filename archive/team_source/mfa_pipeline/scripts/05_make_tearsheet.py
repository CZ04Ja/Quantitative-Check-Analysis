"""Render PNG tearsheets for the latest backtest outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mfa.config import load_config
from mfa.evaluation.stats import perf_stats
from mfa.viz import apply_style, strategy_tearsheet
from mfa.utils import get_logger

log = get_logger("scripts.tearsheet")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    out = cfg.results_root
    rets = pd.read_csv(out / "returns_net.csv", index_col=0, parse_dates=True).iloc[:, 0]
    turn = pd.read_csv(out / "turnover.csv", index_col=0, parse_dates=True).iloc[:, 0]
    bm_path = out / "benchmark_returns.csv"
    bm = pd.read_csv(bm_path, index_col=0, parse_dates=True).iloc[:, 0] if bm_path.exists() else None
    perf = perf_stats(rets)

    apply_style()
    fig = strategy_tearsheet(
        rets, benchmark=bm, turnover=turn, perf=perf,
        title="Multi-Factor Strategy -- Net of Costs",
        savepath=str(out / "tearsheet.png"),
    )
    log.info("tearsheet saved to %s", out / "tearsheet.png")


if __name__ == "__main__":
    main()
