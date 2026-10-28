"""One-shot end-to-end runner.

Runs the four step scripts in order. Each step writes artifacts under
data/processed and data/results, so steps can be resumed independently.

Usage:
    python run_pipeline.py [--config configs/default.yaml]
                           [--skip-download] [--force]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(script: str, *cli_args: str):
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / script), *cli_args]
    print(f"\n{'=' * 72}\n>>> {' '.join(cmd[1:])}\n{'=' * 72}", flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="bypass cache when downloading")
    args = p.parse_args()
    cfg = ["--config", args.config]

    if not args.skip_download:
        extra = ["--force"] if args.force else []
        _run("01_download_data.py", *cfg, *extra)

    _run("02_build_factors.py", *cfg)
    _run("03_evaluate_factors.py", *cfg, "--horizon", "5")
    _run("04_combine_and_backtest.py", *cfg)
    _run("05_make_tearsheet.py", *cfg)


if __name__ == "__main__":
    main()
