"""Factor base class + global registry."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

import pandas as pd


# Factor signature: (panels: dict[str, DataFrame], **params) -> DataFrame
FactorFn = Callable[..., pd.DataFrame]


@dataclass
class Factor:
    """A factor specification."""

    name: str
    fn: FactorFn
    family: str = "misc"
    description: str = ""
    default_params: Dict = field(default_factory=dict)

    def compute(self, panels: dict[str, pd.DataFrame], **overrides) -> pd.DataFrame:
        params = {**self.default_params, **overrides}
        out = self.fn(panels, **params)
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"Factor {self.name} must return a DataFrame, got {type(out)}")
        out.name = self.name
        return out


class FactorRegistry:
    """Global registry. Use `@REGISTRY.register(...)` to add a factor."""

    def __init__(self) -> None:
        self._factors: dict[str, Factor] = {}

    def register(
        self,
        name: str,
        family: str = "misc",
        description: str = "",
        default_params: dict | None = None,
    ):
        def deco(fn: FactorFn) -> FactorFn:
            f = Factor(
                name=name,
                fn=fn,
                family=family,
                description=description,
                default_params=default_params or {},
            )
            if name in self._factors:
                raise ValueError(f"factor {name!r} already registered")
            self._factors[name] = f
            return fn
        return deco

    def get(self, name: str) -> Factor:
        if name not in self._factors:
            raise KeyError(f"unknown factor: {name}")
        return self._factors[name]

    def names(self) -> list[str]:
        return sorted(self._factors)

    def by_family(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for f in self._factors.values():
            out.setdefault(f.family, []).append(f.name)
        for v in out.values():
            v.sort()
        return out

    def __iter__(self):
        return iter(self._factors.values())

    def __len__(self) -> int:
        return len(self._factors)


REGISTRY = FactorRegistry()
