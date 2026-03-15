"""Figure: Peak at Step 1 — composite metric showing immediate degradation."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load metrics
def load_metrics(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

r1 = load_metrics("logs/thinking-experiment/rung1-no-think/metrics.jsonl")
r2 = load_metrics("logs/thinking-experiment/rung2-private-think/metrics.jsonl")

def composite(rows):
    """Mean of accuracy_mean, truth_surfaced, judge_quality — normalized to [0,1]."""
    vals = []
    for r in rows:
        acc = ((r.get("env/all/accuracy.debater_a", 0) or 0) +
               (r.get("env/all/accuracy.debater_b", 0) or 0)) / 2
        ts = r.get("env/all/truth_surfaced", 0) or 0
        jq = r.get("env/all/judge_quality", 0) or 0
        vals.append((acc + ts + jq) / 3)
    return vals

c1 = composite(r1)
c2 = composite(r2)
steps1 = list(range(len(c1)))
steps2 = list(range(len(c2)))

fig, ax = plt.subplots(figsize=(6.8, 4.2))

# Destructive region (after step 1) — lighter tint
ax.axvspan(1.5, max(len(c1), len(c2)) - 0.5, color="#FDF2F2", zorder=0)

# Lines
ax.plot(steps1, c1, "o-", color="#4E79A7", markersize=6, linewidth=2, label="Rung 1 (no think)", zorder=3)
ax.plot(steps2, c2, "s-", color="#F28E2B", markersize=6, linewidth=2, label="Rung 2 (private think)", zorder=3)

# Vertical line at step 1
ax.axvline(x=1, color="#595959", linewidth=1, linestyle="--", alpha=0.5, zorder=1)

# Annotate peak
peak1 = max(c1)
peak2 = max(c2)
ax.annotate(f"R1 peak: {c1[1]:.3f}", xy=(1, c1[1]), xytext=(4, 0.49),
            fontsize=9, fontweight="bold", color="#4E79A7",
            arrowprops=dict(arrowstyle="->", color="#4E79A7", lw=1.2))
ax.annotate(f"R2 peak: {c2[1]:.3f}", xy=(1, c2[1]), xytext=(4, 0.55),
            fontsize=9, fontweight="bold", color="#F28E2B",
            arrowprops=dict(arrowstyle="->", color="#F28E2B", lw=1.2))

# "Net destructive" label — positioned in the mid-chart area
mid_step = (1.5 + len(c1) - 0.5) / 2
ax.text(mid_step, 0.28, "Training is net-destructive after step 1",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="#D55E00", alpha=0.7)

# Step 1 label
ax.text(1, ax.get_ylim()[0] - 0.01, "step 1", ha="center", va="top",
        fontsize=8, color="#595959")

ax.set_xlabel("Training step", fontsize=11)
ax.set_ylabel("Composite metric", fontsize=10)
ax.set_title("Both runs peak at step 1, then degrade monotonically",
             fontsize=12, fontweight="bold", color="#595959", pad=10)

ax.set_xlim(-0.5, len(c1) - 0.5)
ax.set_ylim(0.20, max(peak1, peak2) + 0.06)

# Caption for composite definition
ax.text(0.02, 0.02, "Composite = mean(accuracy, truth_surfaced, judge_quality)",
        transform=ax.transAxes, fontsize=7.5, color="#999999", va="bottom")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.8)
ax.set_axisbelow(True)

ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="#D9D9D9")

plt.tight_layout()
plt.savefig("reports/figures/fig_peak_step1.png", dpi=600, bbox_inches="tight",
            facecolor="white")
print("Saved fig_peak_step1.png")
