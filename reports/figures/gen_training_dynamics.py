"""Training Dynamics — ONE figure, 2 panels, 3 metrics overlaid.

Tufte redesign: accuracy, disagreement, draw rate on shared [0,1] axes.
Mean of debater A+B for accuracy. Collapse region shaded in Rung 1.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

r1 = load("logs/thinking-experiment/rung1-no-think/metrics.jsonl")
r2 = load("logs/thinking-experiment/rung2-private-think/metrics.jsonl")

def extract(rows, key_a, key_b=None):
    if key_b:
        return [((r.get(key_a, 0) or 0) + (r.get(key_b, 0) or 0)) / 2 for r in rows]
    return [r.get(key_a, 0) or 0 for r in rows]

# Extract metrics
acc1 = extract(r1, "env/all/accuracy.debater_a", "env/all/accuracy.debater_b")
dis1 = extract(r1, "env/all/disagreement")
drw1 = extract(r1, "env/all/draw_rate")
acc1_a = [r.get("env/all/accuracy.debater_a", 0) or 0 for r in r1]
acc1_b = [r.get("env/all/accuracy.debater_b", 0) or 0 for r in r1]

acc2 = extract(r2, "env/all/accuracy.debater_a", "env/all/accuracy.debater_b")
dis2 = extract(r2, "env/all/disagreement")
drw2 = extract(r2, "env/all/draw_rate")

steps1 = np.arange(len(r1))
steps2 = np.arange(len(r2))
max_steps = max(len(r1), len(r2))

# Colors
C_ACC = "#4E79A7"
C_DIS = "#59A14F"
C_DRW = "#E15759"
C_GRAY = "#595959"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
fig.subplots_adjust(wspace=0.08)

for ax, steps, acc, dis, drw, acc_a_vals, acc_b_vals, title, n_steps in [
    (ax1, steps1, acc1, dis1, drw1, acc1_a, acc1_b, "Rung 1: No Think", len(r1)),
    (ax2, steps2, acc2, dis2, drw2,
     [r.get("env/all/accuracy.debater_a", 0) or 0 for r in r2],
     [r.get("env/all/accuracy.debater_b", 0) or 0 for r in r2],
     "Rung 2: Private Think", len(r2)),
]:
    # Accuracy: mean line + A/B band
    ax.fill_between(steps, acc_a_vals, acc_b_vals, alpha=0.12, color=C_ACC, linewidth=0)
    ax.plot(steps, acc, "o-", color=C_ACC, markersize=4, linewidth=1.8, label="Accuracy (mean A,B)", zorder=3)

    # Disagreement
    ax.plot(steps, dis, "s-", color=C_DIS, markersize=4, linewidth=1.8, label="Disagreement", zorder=3)

    # Draw rate
    ax.plot(steps, drw, "^-", color=C_DRW, markersize=4, linewidth=1.8, label="Draw rate", zorder=3)

    ax.set_title(title, fontsize=12, fontweight="bold", color=C_GRAY, pad=8)
    ax.set_xlabel("Training step", fontsize=10)
    ax.set_xlim(-0.5, max_steps - 0.5)
    ax.set_ylim(0, 0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.6)
    ax.set_axisbelow(True)

# Rung 1: shade collapse region
ax1.axvspan(4, 9, color="#FFE0E0", alpha=0.5, zorder=0)
ax1.text(6.5, 0.68, "collapse\nregion", ha="center", va="top", fontsize=8,
         color="#D55E00", style="italic", alpha=0.8)

# Rung 1: annotate the convergence
ax1.annotate("All three converge:\naccuracy, disagreement,\ndraw rate cluster near 0.2-0.3",
             xy=(10, 0.25), xytext=(1, 0.08),
             fontsize=7.5, color=C_GRAY,
             arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=0.8),
             ha="left", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9))

# Rung 2: annotate stability
ax2.annotate("Metrics stay separated:\ndebate structure preserved",
             xy=(6, 0.42), xytext=(1, 0.08),
             fontsize=7.5, color=C_GRAY,
             arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=0.8),
             ha="left", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9))

ax1.set_ylabel("Rate", fontsize=10)

# Single legend — upper right of Rung 2 panel (less data there)
ax2.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="white",
           edgecolor="#DDDDDD", handlelength=1.5)

fig.suptitle("Without thinking, all training signals collapse together",
             fontsize=13, fontweight="bold", color=C_GRAY, y=1.0)

plt.savefig("reports/figures/fig_training_dynamics.png", dpi=600, bbox_inches="tight",
            facecolor="white")
print("Saved fig_training_dynamics.png")
