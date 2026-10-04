"""House plotting style and output paths, so every notebook looks the same."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

#: Colour-blind-safe series colours, used in the order listed.
PALETTE = ["#0b6e4f", "#b5591f", "#1f4e79", "#7b3294", "#8c6d1f"]


def apply_style() -> None:
    """Set readable defaults and make sure the output folders exist."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "font.size": 9,
        "legend.frameon": False,
    })
    DATA_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def save(fig, name: str) -> Path:
    """Save a figure into ``figures/`` and return where it landed."""
    FIG_DIR.mkdir(exist_ok=True)
    target = FIG_DIR / name
    fig.savefig(target)
    return target
