"""Helpers to compute many factors at once and store them as a 3D dict."""
from __future__ import annotations

import time
from typing import Iterable

import pandas as pd

from ..utils import get_logger
from .base import REGISTRY

log = get_logger("mfa.factors")


def compute_all(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute every registered factor on the given panels.

    Returns: { factor_name: DataFrame(date x symbol) of raw values }.
    Failed factors are skipped (and logged) so one bug doesn't break the run.
    """
    out: dict[str, pd.DataFrame] = {}
    for f in REGISTRY:
        t0 = time.time()
        try:
            out[f.name] = f.compute(panels)
            log.info("[%s] ok  family=%s  shape=%s  %.2fs",
                     f.name, f.family, out[f.name].shape, time.time() - t0)
        except Exception as e:  # noqa: BLE001
            log.exception("[%s] FAILED: %s", f.name, e)
    return out


def compute_subset(panels: dict[str, pd.DataFrame], names: Iterable[str]
                   ) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for n in names:
        f = REGISTRY.get(n)
        out[n] = f.compute(panels)
    return out
