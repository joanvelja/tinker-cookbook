"""Figure: CoT-Output Decoupling — think block vs public output divergence."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(6.4, 3.8))

# Verified data (deduplicated, n=953 stands, n=80 revise)
categories = ['Think concludes:\n"answer stands"\n(n=953)', 'Think concludes:\n"must revise"\n(n=80)']
keeps = [85.6, 12.5]  # 816/953, 10/80
changes = [14.4, 87.5]  # 137/953, 70/80

y = np.arange(2)
bar_h = 0.45

# Stacked horizontal bars
bars_keep = ax.barh(y, keeps, bar_h, color="#4E79A7", edgecolor="#404040", linewidth=0.5,
                     label="Keeps answer (aligned)")
bars_change = ax.barh(y, changes, bar_h, left=keeps, color="#F28E2B", edgecolor="#404040",
                       linewidth=0.5, label="Changes answer")

# Hatch the divergent segments
# For "stands" → changes (the alarming 14.4%)
bars_change[0].set_hatch("///")
bars_change[0].set_edgecolor("#D55E00")
bars_change[0].set_linewidth(1.0)

# For "revise" → keeps (the minor 12.5%)
bars_keep[1].set_hatch("///")
bars_keep[1].set_edgecolor("#D55E00")
bars_keep[1].set_linewidth(1.0)

# Labels inside bars
# Stands → keeps
ax.text(keeps[0] / 2, y[0], f"{keeps[0]:.1f}%", ha="center", va="center",
        fontsize=11, fontweight="bold", color="white")
# Stands → changes (the key finding) — white background for readability over hatch
ax.text(keeps[0] + changes[0] / 2, y[0], f"{changes[0]:.1f}%", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#D55E00",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

# Revise → changes
ax.text(keeps[1] + changes[1] / 2, y[1], f"{changes[1]:.1f}%", ha="center", va="center",
        fontsize=11, fontweight="bold", color="white")
# Revise → keeps — white background over hatch
ax.text(keeps[1] / 2, y[1], f"{keeps[1]:.1f}%", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#D55E00",
        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))

# Callout bracket for the 14.4%
ax.annotate("14.4% change despite\nprivate conclusion to keep",
            xy=(keeps[0] + changes[0] / 2, y[0] + bar_h / 2 + 0.02),
            xytext=(70, 1.15),
            fontsize=10, fontweight="bold", color="#D55E00",
            arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.5),
            ha="center", va="bottom")

ax.set_yticks(y)
ax.set_yticklabels(categories, fontsize=9.5)
ax.set_xlim(0, 100)
ax.set_xlabel("Outcome (%)", fontsize=10)
ax.set_title("Chain-of-thought / output decoupling",
             fontsize=13, fontweight="bold", color="#595959", pad=15)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.5)
ax.set_axisbelow(True)

# Legend
ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor="white",
          edgecolor="#D9D9D9")

# Hatched = divergent annotation
ax.text(0.98, 0.02, "/// = divergent (CoT ≠ output)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, color="#D55E00", style="italic")

plt.tight_layout()
plt.savefig("reports/figures/fig_cot_decoupling.png", dpi=600, bbox_inches="tight",
            facecolor="white")
print("Saved fig_cot_decoupling.png")
