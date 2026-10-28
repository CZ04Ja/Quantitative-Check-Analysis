"""Download daily bars for the configured universe via Alpaca and cache them
under data/raw/alpaca/iex/1Day/.

Usage:
    python scripts/01_download_data.py [--config configs/default.yaml] [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mfa.config import load_config
from mfa.data import AlpacaLoader, AlpacaConfig, Universe
from mfa.utils import get_logger

log = get_logger("scripts.download")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--force", action="store_true",
                   help="Bypass cache and re-download from scratch.")
    args = p.parse_args()

    cfg = load_config(args.config)
    period = cfg.section("period")
    uni = Universe.from_config(cfg.section("universe"))
    alp_cfg = AlpacaConfig.from_dict(cfg.section("alpaca"))

    loader = AlpacaLoader(alp_cfg, cache_dir=cfg.raw_dir)
    panels = loader.load_panel(
        symbols=uni.all_symbols,
        start=period["start"],
        end=period["end"],
        timeframe=period.get("timeframe", "1Day"),
        force_refresh=args.force,
    )

    out = cfg.processed_dir / "panels.parquet"
    # Save each panel as a separate parquet
    for k, v in panels.items():
        target = cfg.processed_dir / f"panel_{k}.parquet"
        v.to_parquet(target)
        log.info("[%s] -> %s  shape=%s", k, target.name, v.shape)
    log.info("done. universe=%d names. dates=%s..%s",
             panels["close"].shape[1],
             panels["close"].index.min().date(),
             panels["close"].index.max().date())


if __name__ == "__main__":
    main()
