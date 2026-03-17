"""Figure: Agreement Collapse — debate outcome composition over training."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open("logs/thinking-experiment/rung1-no-think/episodes/episodes.jsonl") as f:
    episodes = [json.loads(l) for l in f if l.strip()]

# Deduplicate by debate_id (each debate has 2 episodes)
seen = set()
debates = []
for ep in episodes:
    did = ep["debate_id"]
    if did not in seen:
        seen.add(did)
        debates.append(ep)

n = len(debates)
quintile_size = n // 5
quintiles = [debates[i * quintile_size:(i + 1) * quintile_size] for i in range(5)]

# Compute categories per quintile
labels = ["Q1\n(early)", "Q2", "Q3", "Q4", "Q5\n(late)"]
both_correct, both_wrong, contested = [], [], []

for q in quintiles:
    bc = bw = ct = 0
    for d in q:
        acc_a = d["signals"].get("accuracy.debater_a")
        acc_b = d["signals"].get("accuracy.debater_b")
        if acc_a is None or acc_b is None:
            continue
        a_correct = acc_a > 0.5
        b_correct = acc_b > 0.5
        if a_correct and b_correct:
            bc += 1
        elif not a_correct and not b_correct:
            bw += 1
        else:
            ct += 1
    total = bc + bw + ct
    both_correct.append(100 * bc / total if total else 0)
    both_wrong.append(100 * bw / total if total else 0)
    contested.append(100 * ct / total if total else 0)

# Plot
fig, ax = plt.subplots(figsize=(6.6, 4.4))

x = np.arange(5)
width = 0.6

# Stack: both_wrong at bottom, contested in middle, both_correct on top
bars_bw = ax.bar(x, both_wrong, width, color="#D55E00", edgecolor="#404040",
                 linewidth=0.5, label="Both wrong")
bars_ct = ax.bar(x, contested, width, bottom=both_wrong, color="#EDC948",
                 edgecolor="#404040", linewidth=0.5, label="Contested")
bars_bc = ax.bar(x, both_correct, width,
                 bottom=[bw + ct for bw, ct in zip(both_wrong, contested)],
                 color="#59A14F", edgecolor="#404040", linewidth=0.5, label="Both correct")

# Labels inside bars (skip if too thin)
for i in range(5):
    # Both wrong
    if both_wrong[i] > 8:
        ax.text(x[i], both_wrong[i] / 2, f"{both_wrong[i]:.0f}%",
                ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    # Contested — use leader line if too thin
    mid_ct = both_wrong[i] + contested[i] / 2
    if contested[i] > 5:
        ax.text(x[i], mid_ct, f"{contested[i]:.1f}%",
                ha="center", va="center", fontsize=8, color="#404040")
    elif contested[i] > 0:
        ax.annotate(f"{contested[i]:.1f}%", xy=(x[i] + width / 2, mid_ct),
                    xytext=(x[i] + width / 2 + 0.4, mid_ct),
                    fontsize=7, color="#404040",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7),
                    va="center")
    # Both correct
    mid_bc = both_wrong[i] + contested[i] + both_correct[i] / 2
    if both_correct[i] > 8:
        ax.text(x[i], mid_bc, f"{both_correct[i]:.0f}%",
                ha="center", va="center", fontsize=9, fontweight="bold", color="white")

# Callout on Q5 red block — placed above the bars entirely
ax.annotate(f"{both_wrong[-1]:.0f}% both wrong",
            xy=(x[-1], both_wrong[-1] * 0.6), xytext=(x[-1], 108),
            fontsize=11, fontweight="bold", color="#D55E00",
            arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.5),
            ha="center")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 115)
ax.set_ylabel("Debate outcome (%)", fontsize=11)
ax.set_title("Wrong-answer consensus expands\nwhile real disagreement disappears",
             fontsize=12, fontweight="bold", color="#595959", pad=12)

# Y gridlines at 20% intervals
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.8)
ax.set_axisbelow(True)

# Remove top/right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend below
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
          fontsize=10, frameon=False)

plt.savefig("reports/figures/fig_agreement_collapse.png", dpi=600, bbox_inches="tight",
            facecolor="white")
print("Saved fig_agreement_collapse.png")
