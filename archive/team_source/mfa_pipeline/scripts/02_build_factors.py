"""Compute the full factor zoo on the cached data and save to disk."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mfa.config import load_config
from mfa.data import build_panels
from mfa.factors import compute_all, REGISTRY
from mfa.utils import get_logger

log = get_logger("scripts.factors")


def _load_panels(cfg) -> dict:
    keys = ["open", "high", "low", "close", "volume", "vwap", "dollar_volume"]
    out = {}
    for k in keys:
        p = cfg.processed_dir / f"panel_{k}.parquet"
        if p.exists():
            out[k] = pd.read_parquet(p)
            out[k].index = pd.to_datetime(out[k].index, utc=True)
    if not out:
        raise RuntimeError("No cached panels found. Run 01_download_data.py first.")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    raw_panels = _load_panels(cfg)
    bm = cfg.section("universe").get("benchmark", "SPY")
    panels = build_panels(raw_panels, benchmark=bm, min_history_days=252)

    log.info("universe = %d names, %d trading days",
             panels["close"].shape[1], panels["close"].shape[0])
    log.info("Registered factors: %d", len(REGISTRY))
    for fam, names in REGISTRY.by_family().items():
        log.info("  %-16s -> %s", fam, names)

    factors = compute_all(panels)

    out_dir = cfg.processed_dir / "factors"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, f in factors.items():
        f.to_parquet(out_dir / f"{name}.parquet")
    log.info("saved %d factor panels to %s", len(factors), out_dir)


if __name__ == "__main__":
    main()
