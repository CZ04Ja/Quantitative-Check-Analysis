"""Evaluate every factor: IC summary, quantile L-S Sharpe, turnover, etc."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mfa.config import load_config
from mfa.data import build_panels, forward_returns
from mfa.evaluation import (
    information_coefficient, ic_summary, quantile_returns,
    long_short_returns, factor_turnover,
)
from mfa.evaluation.stats import perf_stats
from mfa.utils import get_logger

log = get_logger("scripts.evaluate")


def _load_panels(cfg) -> dict:
    out = {}
    for k in ["open", "high", "low", "close", "volume", "vwap", "dollar_volume"]:
        p = cfg.processed_dir / f"panel_{k}.parquet"
        if p.exists():
            out[k] = pd.read_parquet(p)
            out[k].index = pd.to_datetime(out[k].index, utc=True)
    return out


def _load_factors(cfg) -> dict[str, pd.DataFrame]:
    d = cfg.processed_dir / "factors"
    out = {}
    for p in sorted(d.glob("*.parquet")):
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True)
        out[p.stem] = df
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--horizon", type=int, default=5,
                   help="forward return horizon (days) used for IC + quantile L-S")
    args = p.parse_args()

    cfg = load_config(args.config)
    bm = cfg.section("universe").get("benchmark", "SPY")
    panels = build_panels(_load_panels(cfg), benchmark=bm, min_history_days=252)
    factors = _load_factors(cfg)
    if not factors:
        raise RuntimeError("No factor parquet files. Run 02_build_factors.py first.")

    fwd = forward_returns(panels["close"], horizons=[args.horizon])[args.horizon]

    rows = []
    for name, f in factors.items():
        ic = information_coefficient(f, fwd)
        ics = ic_summary(ic)
        qret = quantile_returns(f, fwd, n_quantiles=5)
        ls = long_short_returns(qret)
        # Convert h-day non-overlapping returns into per-step stats
        ls_stats = perf_stats(ls, periods_per_year=252 / args.horizon)
        to = factor_turnover(f, n_quantiles=5).mean()
        rows.append({
            "factor": name,
            "ic_mean": ics["ic_mean"],
            "ic_ir": ics["ic_ir"],
            "ic_t_stat": ics["ic_t_stat"],
            "ic_hit_rate": ics["ic_hit_rate"],
            "ls_sharpe": ls_stats["sharpe"],
            "ls_ann_ret": ls_stats["ann_return"],
            "ls_max_dd": ls_stats["max_drawdown"],
            "ls_calmar": ls_stats["calmar"],
            "avg_turnover": float(to),
        })

    res = pd.DataFrame(rows).set_index("factor").sort_values("ic_ir", ascending=False)
    out = cfg.results_root / f"factor_summary_h{args.horizon}.csv"
    res.to_csv(out)
    log.info("saved %s", out)
    print(res.to_string(float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
