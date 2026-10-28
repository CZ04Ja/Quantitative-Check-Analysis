"""Centralised matplotlib styling so every chart looks the same."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


_PALETTE = [
    "#2E86AB", "#E63946", "#F4A261", "#2A9D8F", "#8E44AD",
    "#F77F00", "#06A77D", "#D62828", "#003049", "#5C946E",
]


def apply_style() -> None:
    """Apply a clean, dense, professional style. Call once at notebook top."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.figsize": (11, 4.5),
        "figure.dpi": 110,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=_PALETTE),
        "lines.linewidth": 1.6,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })
