"""TicTacToe self-play ablation figure.

Three-arm comparison: Control (no subgroups), Treatment MC (per-role + mean_center),
Treatment MaxRL (per-role + maxrl). Shows reward trajectory and verbosity drift.
n=1 per arm. Entropy collapse was uniform across arms (not shown).
"""

import matplotlib.pyplot as plt
import numpy as np

batches = np.array([0, 5, 10, 15, 20, 25, 30])

reward = {
    "Control": [-0.406, 0.266, 0.141, -0.062, 0.031, 0.125, 0.156],
    "Per-role MC": [-0.344, 0.359, -0.250, -0.188, -0.188, 0.141, 0.375],
    "Per-role MaxRL": [-0.469, 0.328, 0.156, 0.078, 0.156, 0.188, 0.141],
}
tokens = {
    "Control": [26.7, 16.7, 16.8, 17.7, 18.8, 16.8, 14.2],
    "Per-role MC": [25.2, 17.1, 26.6, 39.5, 33.3, 28.5, 20.1],
    "Per-role MaxRL": [25.2, 15.6, 14.8, 15.5, 15.7, 14.8, 21.3],
}

# Colorblind-safe (Tol bright): distinct hue + linestyle + marker
styles = {
    "Control": {"color": "#4477AA", "linestyle": "-", "marker": "s"},
    "Per-role MC": {"color": "#EE6677", "linestyle": "--", "marker": "o"},
    "Per-role MaxRL": {"color": "#228833", "linestyle": "-.", "marker": "^"},
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"wspace": 0.35})

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.2, lw=0.5)
    ax.tick_params(direction="out", length=3, labelsize=8)
    ax.set_xticks(batches)

fig.supxlabel("Training Batch", fontsize=9)

line_kw = dict(lw=1.6, markersize=4.5, markeredgewidth=0.7, markeredgecolor="white", zorder=3)

handles = []
for name in reward:
    s = styles[name]
    (h,) = ax1.plot(
        batches, reward[name], color=s["color"], linestyle=s["linestyle"],
        marker=s["marker"], label=name, **line_kw,
    )
    handles.append(h)
    ax2.plot(
        batches, tokens[name], color=s["color"], linestyle=s["linestyle"],
        marker=s["marker"], **line_kw,
    )

# --- Panel A: reward ---
ax1.axhline(0, color="#999999", lw=0.5, zorder=1)
ax1.set_ylabel("Test Reward", fontsize=9)
ax1.set_ylim(-0.55, 0.5)

# --- Panel B: tokens ---
# Neutral shading for drift window
ax2.axvspan(5, 20, alpha=0.12, color="#888888", zorder=0)

# Baseline reference: mean of Control tokens (batches 5-30, post-init)
ctrl_baseline = np.mean(tokens["Control"][1:])
ax2.axhline(ctrl_baseline, color="#999999", lw=0.7, linestyle=":", zorder=1)
ax2.text(30.5, ctrl_baseline - 1.5, "ctrl mean", fontsize=7, color="#777777",
         va="top", ha="right")

# Drift window label with white background for contrast
ax2.text(
    12.5, 42, "drift window", fontsize=8, color="#444444", ha="center",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
)

ax2.set_ylabel("Action Tokens / Turn", fontsize=9)
ax2.set_ylim(10, 45)
ax2.set_xlim(-1.5, 31.5)

# Panel labels
for label, ax in [("A", ax1), ("B", ax2)]:
    ax.text(0.03, 0.96, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", ha="left", va="top")

# Legend above panels
fig.legend(
    handles,
    list(reward.keys()),
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.05),
    frameon=False,
    fontsize=8.5,
    columnspacing=2.0,
    handlelength=2.5,  # longer swatches so linestyles are distinguishable
)

fig.tight_layout(rect=[0, 0.04, 1, 0.92])

import os

outdir = "tinker_cookbook/recipes/multiplayer_rl/debate/figures"
os.makedirs(outdir, exist_ok=True)
plt.savefig(f"{outdir}/ttt_ablation.png", bbox_inches="tight", dpi=600)
plt.savefig(f"{outdir}/ttt_ablation.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved ttt_ablation.png and ttt_ablation.pdf")
