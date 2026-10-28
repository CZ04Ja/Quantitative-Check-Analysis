"""Tiny YAML config loader. No third-party config framework needed."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_NAME = "default.yaml"


def _project_root() -> Path:
    """Repo root = parent of the `mfa/` package directory."""
    return Path(__file__).resolve().parent.parent


def _resolve_path(p: str | os.PathLike[str]) -> Path:
    pth = Path(p)
    if not pth.is_absolute():
        pth = _project_root() / pth
    return pth


@dataclass
class Config:
    """Lightweight wrapper around the parsed YAML so callers get attribute access
    AND can still treat sub-sections as plain dicts."""

    raw: dict[str, Any] = field(default_factory=dict)

    # ---- conveniences -----------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def section(self, name: str) -> dict[str, Any]:
        sec = self.raw.get(name, {})
        if not isinstance(sec, dict):
            raise TypeError(f"Config section {name!r} is not a mapping")
        return sec

    # ---- common paths -----------------------------------------------------
    @property
    def project_root(self) -> Path:
        return _project_root()

    @property
    def data_root(self) -> Path:
        return _resolve_path(self.section("project").get("data_root", "data"))

    @property
    def results_root(self) -> Path:
        return _resolve_path(self.section("project").get("results_root", "data/results"))

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    def ensure_dirs(self) -> None:
        for d in (self.data_root, self.results_root, self.raw_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load a YAML config. If `path` is None, use configs/default.yaml."""
    if path is None:
        path = _project_root() / "configs" / _DEFAULT_CONFIG_NAME
    path = _resolve_path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = Config(raw=data)
    cfg.ensure_dirs()
    return cfg
