"""Shared figure style for the debate analysis report.

Import this at the top of every figure script:
    from style import apply_style, COLORS, annotate_insight
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# Palette (Tableau colorblind-safe)
COLORS = {
    "blue": "#4E79A7",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "red": "#E15759",
    "teal": "#76B7B2",
    "purple": "#B07AA1",
    "brown": "#9C755F",
    "gray": "#BAB0AC",
    "dark": "#404040",
    "light": "#888888",
    "grid": "#E8E8E8",
    "accent": "#D55E00",  # for callouts/warnings
    # Rung-specific
    "rung1": "#4E79A7",
    "rung2": "#F28E2B",
    # Semantic
    "good": "#59A14F",
    "bad": "#E15759",
    "neutral": "#BAB0AC",
}


def apply_style():
    """Apply consistent rcParams. Call once at script start."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": COLORS["dark"],
        "axes.titlepad": 10,
        "axes.labelsize": 11,
        "axes.labelcolor": COLORS["dark"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#DDDDDD",
        "figure.facecolor": "white",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def annotate_insight(ax, text, xy, xytext, color=None):
    """Add a callout annotation with consistent styling."""
    color = color or COLORS["accent"]
    ax.annotate(
        text, xy=xy, xytext=xytext,
        fontsize=9, fontweight="bold", color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9),
    )


def save(fig, name):
    """Save figure with consistent settings."""
    fig.savefig(f"reports/figures/{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    print(f"Saved {name}.png")
